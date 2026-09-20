"""Command-line interface for safe setup, automation, and knowledge tools."""

from __future__ import annotations

import argparse
import asyncio
from importlib import resources
import logging
import json
import math
import os
from pathlib import Path
import re
import sys
import sqlite3
import time
import warnings

import mss
import yaml

from autoykt.calibration import Calibrator, ImageSelections
from autoykt.core.config import (
    AppConfig,
    ConfigError,
    default_user_config_path,
    discover_config_path,
    load_config,
)
from autoykt.core.event_bus import Event, EventBus, EventType
from autoykt.core.logger import setup_logger
from autoykt.core.run_guard import keep_awake, run_lease
from autoykt.monitor.class_session import (
    ClassSession,
    read_session_status,
    request_session_stop,
    session_directory,
)
from autoykt.knowledge.ingest import ingest_courseware
from autoykt.knowledge.recorder import LiveKnowledgeRecorder
from autoykt.knowledge.store import KnowledgeStore
from autoykt.monitor.detector import ImageTemplateMatcher
from autoykt.monitor.ocr_engine import create_ocr_engine
from autoykt.monitor.image_inspection import inspect_image
from autoykt.monitor.calibration_checks import calibration_sample_issues
from autoykt.monitor.screen_watcher import ScreenWatcher
from autoykt.monitor.rehearsal import run_rehearsal
from autoykt.monitor.preflight import (
    live_configuration_issues,
    require_live_configuration,
)
from autoykt.monitor.page_tools import (
    capture_state,
    preview_option,
    preview_step,
    recover_current,
    show_windows,
)
from autoykt.notifier.factory import create_notifiers
from autoykt.notifier.qq_bot import QQNotifier


logger = logging.getLogger("autoykt")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autoykt",
        description="Configurable visual quiz automation",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run_parser = commands.add_parser("run", help="monitor and answer questions")
    _add_run_arguments(run_parser)
    _add_session_arguments(
        commands.add_parser(
            "session",
            help="start a timed classroom test (default: 120 minutes)",
        )
    )
    commands.add_parser(
        "windows", help="list visible Windows window selectors and client sizes"
    )
    for name, help_text in (
        ("capture-state", "save a labeled page state without AI or clicks"),
        ("preview", "preview a configured option without AI or clicks"),
        (
            "recover",
            "skip a manually reviewed current page after stopping all runs",
        ),
    ):
        page_parser = commands.add_parser(name, help=help_text)
        _add_config_argument(page_parser)
        page_parser.add_argument("--profile", required=True)
        page_parser.add_argument(
            "--delay",
            type=_parse_positive_float,
            default=3.0,
            help="seconds to bring the target window forward",
        )
        if name == "capture-state":
            page_parser.add_argument(
                "--label",
                required=True,
                choices=[
                    "waiting",
                    "question",
                    "selected",
                    "submitting",
                    "success",
                    "failure",
                    "next",
                    "loading",
                    "manual",
                    "finished",
                ],
            )
        elif name == "preview":
            selection = page_parser.add_mutually_exclusive_group(required=True)
            selection.add_argument("--option", help="answer option to preview")
            selection.add_argument(
                "--step", help="configured after_submit action name"
            )
        else:
            page_parser.add_argument(
                "--skip-current",
                action="store_true",
                required=True,
                help=(
                    "confirm manual review; wait for the page to change "
                    "before answering again"
                ),
            )
    commands.add_parser("monitors", help="list MSS monitor coordinates")

    check_parser = commands.add_parser(
        "check", help="validate config, assets, and environment"
    )
    _add_config_argument(check_parser)
    check_parser.add_argument("--live", action="store_true")
    check_parser.add_argument("--profile")

    init_parser = commands.add_parser(
        "init", help="create a private v2 configuration"
    )
    init_parser.add_argument("--path", default=None)

    migrate_parser = commands.add_parser(
        "migrate", help="write a secret-free v2 copy of a legacy config"
    )
    _add_config_argument(migrate_parser)
    migrate_parser.add_argument("--output", default=None)

    calibrate_parser = commands.add_parser(
        "calibrate", help="interactively calibrate one page profile"
    )
    _add_calibration_arguments(calibrate_parser)
    for name, help_text in (
        ("inspect-image", "inspect a local image without AI or desktop access"),
        (
            "rehearse",
            "replay images with real models and virtual input/feedback",
        ),
    ):
        _add_image_arguments(
            commands.add_parser(name, help=help_text),
            multiple=name == "rehearse",
        )

    qq_parser = commands.add_parser(
        "test-qq", help="send one QQ connectivity message"
    )
    _add_config_argument(qq_parser)

    knowledge_parser = commands.add_parser(
        "knowledge", help="manage local course knowledge"
    )
    knowledge_commands = knowledge_parser.add_subparsers(
        dest="knowledge_command", required=True
    )
    ingest_parser = knowledge_commands.add_parser(
        "ingest", help="ingest courseware files or directories"
    )
    _add_config_argument(ingest_parser)
    ingest_parser.add_argument("--course", required=True, type=_parse_course)
    ingest_parser.add_argument("paths", nargs="+", type=Path)

    search_parser = knowledge_commands.add_parser(
        "search", help="test local retrieval"
    )
    _add_config_argument(search_parser)
    search_parser.add_argument("--course", required=True, type=_parse_course)
    search_parser.add_argument("query")

    record_parser = knowledge_commands.add_parser(
        "record", help="OCR changed lecture slides until interrupted"
    )
    _add_config_argument(record_parser)
    record_parser.add_argument("--course", required=True, type=_parse_course)
    record_parser.add_argument("--monitor", required=True, type=int)
    record_parser.add_argument(
        "--region",
        required=True,
        type=_parse_region,
        help="monitor-relative x,y,width,height",
    )
    record_parser.add_argument(
        "--interval", type=_parse_positive_float, default=2.0
    )
    return parser


def _add_session_arguments(parser: argparse.ArgumentParser) -> None:
    _add_config_argument(parser)
    parser.add_argument("--profile")
    parser.add_argument("--minutes", type=_parse_positive_float, default=120.0)
    parser.add_argument("--dry-run", action="store_true")
    control = parser.add_mutually_exclusive_group()
    control.add_argument(
        "--check",
        action="store_true",
        help="check setup without desktop or API access",
    )
    control.add_argument(
        "--status", action="store_true", help="read the current session status"
    )
    control.add_argument(
        "--stop",
        action="store_true",
        help="request immediate cancellation of the current session",
    )


def _add_image_arguments(
    parser: argparse.ArgumentParser, *, multiple: bool
) -> None:
    _add_config_argument(parser)
    parser.add_argument("--profile", required=True)
    parser.add_argument(
        "--image",
        type=Path,
        action="append" if multiple else "store",
        required=True,
    )
    if multiple:
        parser.add_argument("--output", type=Path)


def _check_command(config: AppConfig, args: argparse.Namespace) -> int:
    assets = config.model_copy(deep=True)
    if args.profile:
        assets.page_profile(args.profile).enabled = True
        assets.runtime.active_profiles = [args.profile]
    result = _check_config(assets)
    if not args.live:
        return result
    issues = live_configuration_issues(config, args.profile)
    for issue in issues:
        print(f"Live setup: {issue}")
    if issues:
        return 1
    print(
        "Live configuration complete; confirm the actual page before running."
    )
    return result


def _add_calibration_arguments(parser: argparse.ArgumentParser) -> None:
    _add_config_argument(parser)
    parser.add_argument("--profile", required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--from-state",
        type=Path,
        help="calibrate a private capture-state report offline",
    )
    parser.add_argument("--delay", type=_parse_positive_float, default=3.0)
    source.add_argument(
        "--from-image",
        type=Path,
        help="calibrate a supplied image offline; the profile stays disabled",
    )
    parser.add_argument(
        "--selections",
        type=Path,
        help="apply JSON crops without a GUI; requires --from-image",
    )


def _add_run_arguments(run_parser: argparse.ArgumentParser) -> None:
    _add_config_argument(run_parser)
    run_parser.add_argument(
        "--detect-only",
        action="store_true",
        help="detect and capture without calling models or clicking",
    )

    run_parser.add_argument("--profile", help="run one enabled profile")
    run_parser.add_argument(
        "--once",
        action="store_true",
        help="stop after one attempt, including preview or failure",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="override config to preview without any clicks",
    )


def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        default=None,
        help="private YAML config; defaults to AUTOYKT_CONFIG/user config",
    )


def _parse_region(value: str) -> tuple[int, int, int, int]:
    try:
        items = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "region must contain four integers"
        ) from error
    if len(items) != 4 or items[2] <= 0 or items[3] <= 0:
        raise argparse.ArgumentTypeError(
            "region must be x,y,width,height with positive dimensions"
        )
    return items


def _parse_positive_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be a number") from error
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return number


def _parse_course(value: str) -> str:
    normalized = value.strip()
    if not re.fullmatch(r"[a-zA-Z0-9_.-]{1,64}", normalized):
        raise argparse.ArgumentTypeError(
            "course must contain at most 64 letters, numbers, '.', '_', or '-'"
        )
    return normalized


def _translate_legacy_arguments(arguments: list[str]) -> list[str]:
    commands = {
        "run",
        "session",
        "monitors",
        "windows",
        "capture-state",
        "inspect-image",
        "rehearse",
        "preview",
        "check",
        "init",
        "migrate",
        "calibrate",
        "test-qq",
        "knowledge",
    }
    if arguments and arguments[0] in commands:
        return arguments
    if arguments in (["-h"], ["--help"]):
        return arguments
    if "--monitors" in arguments:
        return ["monitors"]
    translated = list(arguments)
    if "--calibrate" in translated:
        translated.remove("--calibrate")
        if "--profile" not in translated:
            translated.extend(["--profile", "legacy"])
        return ["calibrate", *translated]
    if "--test-qq" in translated:
        translated.remove("--test-qq")
        return ["test-qq", *translated]
    return ["run", *translated]


def _load_from_argument(path: str | None) -> AppConfig:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        config = load_config(discover_config_path(path))
    for warning in caught:
        print(f"WARNING: {warning.message}", file=sys.stderr)
    return config


async def _run_automation(
    config: AppConfig,
    detect_only: bool,
    *,
    once: bool = False,
    minutes: float | None = None,
) -> int:
    log_dir = config.resolve_path(config.storage.log_dir)
    setup_logger(config.logging.level, log_dir)
    event_bus = EventBus()

    async def log_event(event: Event) -> None:
        if event.type == EventType.ERROR:
            logger.error(
                "Event %s [%s]: %s",
                event.type.value,
                event.profile_id or "system",
                event.payload,
            )
        else:
            logger.info(
                "Event %s [%s]",
                event.type.value,
                event.profile_id or "system",
            )

    for event_type in EventType:
        event_bus.subscribe(event_type, log_event)
    notifiers = create_notifiers(config, event_bus)
    watcher = None
    bus_task = None
    session = None
    try:
        if minutes is not None:
            session = ClassSession(config, event_bus, minutes)
        watcher = ScreenWatcher(
            config, event_bus, detect_only=detect_only, once=once
        )
        bus_task = asyncio.create_task(event_bus.start())
        if session is not None:
            return await session.run(watcher)
        await watcher.start()
        return watcher.exit_code
    finally:
        if session is not None and watcher is None:
            session.finish("startup_error", 1, None)
        try:
            if watcher is not None:
                await watcher.close()
        finally:
            try:
                if bus_task is not None:
                    # Drain notifications before closing their clients.
                    event_bus.stop()
                    await bus_task
            finally:
                await asyncio.gather(
                    *(notifier.close() for notifier in notifiers),
                    return_exceptions=True,
                )
                if session is not None:
                    # Include observer errors drained during shutdown without
                    # replacing the final profile states with STOPPED.
                    session.refresh()
                    print(
                        f"Class session report: {session.report_path}",
                        flush=True,
                    )


def _init_config(destination: Path) -> None:
    if destination.exists():
        raise ConfigError(
            f"refusing to overwrite existing config: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    template = resources.files("autoykt.resources").joinpath(
        "config.example.yaml"
    )
    with destination.open("x", encoding="utf-8") as config_file:
        config_file.write(template.read_text(encoding="utf-8"))
    (destination.parent / "templates" / "options").mkdir(
        parents=True,
        exist_ok=True,
    )
    print(f"Created private config: {destination}")
    print("Next: edit it, add templates, then run 'autoykt check'.")


def _migrate_config(config: AppConfig, output: Path) -> None:
    if output.exists():
        raise ConfigError(f"refusing to overwrite existing file: {output}")
    document = config.model_dump(mode="json")
    # Moving a config must not silently retarget assets or the knowledge DB.
    for key, value in document["storage"].items():
        document["storage"][key] = str(config.resolve_path(value))
    document["knowledge"]["database_path"] = str(
        config.resolve_path(config.knowledge.database_path)
    )
    if config.answering.prompt_template:
        document["answering"]["prompt_template"] = str(
            config.resolve_path(config.answering.prompt_template)
        )
    for page in document["pages"]:
        for template in (
            page["triggers"]
            + page["verification"]["success_templates"]
            + page["verification"]["failure_templates"]
            + ([page["page_guard"]] if page["page_guard"] else [])
            + [step["when"] for step in page["page_flow"]["before_question"]]
            + [step["when"] for step in page["page_flow"]["after_submit"]]
            + page["page_flow"]["loading"]
            + page["page_flow"]["manual"]
            + page["page_flow"]["finished"]
            + page["calibration_samples"]
        ):
            template["path"] = str(config.resolve_path(template["path"]))
        templates = page["answer_style"]["option_templates"]
        for option, value in templates.items():
            templates[option] = str(config.resolve_path(value))
    exported_providers = document["answering"]["providers"]
    for provider, exported in zip(
        config.answering.providers,
        exported_providers,
        strict=True,
    ):
        if not exported.get("api_key_env"):
            exported["api_key_env"] = _migration_api_key_environment(
                provider.name
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as output_file:
        yaml.safe_dump(
            document,
            output_file,
            allow_unicode=True,
            sort_keys=False,
        )
    print(f"Wrote secret-free v2 config: {output}")
    print("Review api_key_env and notifier environment names before use.")


def _migration_api_key_environment(provider_name: str) -> str:
    if provider_name.casefold() == "openai":
        return "OPENAI_API_KEY"
    identifier = re.sub(r"[^A-Za-z0-9]+", "_", provider_name)
    identifier = identifier.strip("_").upper() or "MODEL"
    return f"AUTOYKT_{identifier}_API_KEY"


# Asset checks are also used on an in-memory calibration candidate.
# pylint: disable-next=too-many-branches
def _check_config(config: AppConfig, *, announce_success: bool = True) -> int:
    errors: list[str] = []
    warning_messages: list[str] = []
    profiles = config.active_page_profiles()
    if not profiles:
        warning_messages.append("no enabled page profiles are active")
    interactions_enabled = (
        not config.runtime.dry_run and config.answering.auto_apply
    )
    if config.runtime.dry_run != (not config.answering.auto_apply):
        warning_messages.append(
            "mouse interaction remains disabled until runtime.dry_run=false "
            "and answering.auto_apply=true"
        )
    if config.answering.prompt_template:
        _check_file(
            config.resolve_path(config.answering.prompt_template),
            "answering prompt template",
            errors,
        )
    for profile in profiles:
        errors.extend(calibration_sample_issues(config, profile))
        for trigger in profile.triggers:
            _check_file(
                config.resolve_path(trigger.path),
                f"profile {profile.id} trigger {trigger.name}",
                errors,
                is_image=True,
            )
        for option, path in profile.answer_style.option_templates.items():
            _check_file(
                config.resolve_path(path),
                f"profile {profile.id} option {option}",
                errors,
                is_image=True,
            )
        for template in [
            *profile.verification.success_templates,
            *profile.verification.failure_templates,
            *profile.page_flow.templates,
            *([profile.page_guard] if profile.page_guard else []),
        ]:
            _check_file(
                config.resolve_path(template.path),
                f"profile {profile.id} operation template",
                errors,
                is_image=True,
            )
        if (
            interactions_enabled
            and profile.target_window
            and not profile.page_guard
        ):
            errors.append(
                f"profile {profile.id} live window operation requires "
                f"page_guard"
            )
        if interactions_enabled and not profile.verification.success_templates:
            if (
                profile.target_window
                or profile.verification.require_success_template
                or bool(profile.page_flow.after_submit)
            ):
                errors.append(
                    f"profile {profile.id} requires explicit success templates"
                )
            warning_messages.append(
                f"profile {profile.id} verifies by visual change only; "
                "a success template is safer"
            )
    for provider in config.answering.providers:
        if provider.api_key is None:
            if not provider.api_key_env:
                errors.append(
                    f"provider {provider.name} has no API key environment"
                )
            elif not os.environ.get(provider.api_key_env, "").strip():
                warning_messages.append(
                    f"provider {provider.name} environment "
                    f"{provider.api_key_env} is not set"
                )
    required_notifier_environment = []
    if "qq" in config.notifier.enabled:
        required_notifier_environment.append(config.notifier.qq.target_env)
    if "telegram" in config.notifier.enabled:
        required_notifier_environment.extend(
            [
                config.notifier.telegram.token_env,
                config.notifier.telegram.chat_id_env,
            ]
        )
    for variable in required_notifier_environment:
        if not os.environ.get(variable, "").strip():
            errors.append(f"notifier environment {variable} is not set")
    for item in errors:
        print(f"ERROR: {item}")
    for item in warning_messages:
        print(f"WARNING: {item}")
    if not errors and announce_success:
        print(
            f"Config OK: {config.source_path} "
            f"({len(profiles)} active profile(s))"
        )
    return 1 if errors else 0


def _check_file(
    path: Path, label: str, errors: list[str], *, is_image: bool = False
) -> None:
    if not path.is_file():
        errors.append(f"{label} file is missing: {path}")
    elif is_image:
        try:
            ImageTemplateMatcher([(label, str(path), 1.0)])
        except (OSError, ValueError) as error:
            errors.append(f"{label}: {error}")


def _show_monitors() -> None:
    with mss.mss() as screen:
        for index, monitor in enumerate(screen.monitors):
            label = "virtual desktop" if index == 0 else f"monitor {index}"
            print(
                f"[{index}] {label}: left={monitor['left']}, "
                f"top={monitor['top']}, "
                f"size={monitor['width']}x{monitor['height']}"
            )


def _knowledge_store(config: AppConfig) -> KnowledgeStore:
    settings = config.knowledge
    return KnowledgeStore(
        config.resolve_path(settings.database_path),
        chunk_size=settings.chunk_size_characters,
        chunk_overlap=settings.chunk_overlap_characters,
    )


def _ingest_knowledge(config: AppConfig, args: argparse.Namespace) -> None:
    with _knowledge_store(config) as store:
        results = ingest_courseware(store, args.course, args.paths)
    for item in results:
        status = "added" if item.result.added else "unchanged"
        print(f"{status}: {item.path} ({item.result.chunks} chunks)")


def _search_knowledge(config: AppConfig, args: argparse.Namespace) -> None:
    settings = config.knowledge
    with _knowledge_store(config) as store:
        matches = store.search(
            args.course,
            args.query,
            settings.maximum_results,
            settings.minimum_score,
        )
    for index, match in enumerate(matches, start=1):
        print(f"[{index}] score={match.score:.3f} source={match.source}")
        print(match.content)
        print()


async def _record_knowledge(
    config: AppConfig, args: argparse.Namespace
) -> None:
    with _knowledge_store(config) as store:
        recorder = LiveKnowledgeRecorder(
            store=store,
            ocr_engine=create_ocr_engine(),
            course_id=args.course,
            monitor_index=args.monitor,
            region=args.region,
            interval_seconds=args.interval,
        )
        try:
            await recorder.run()
        finally:
            recorder.stop()


async def _test_qq(config: AppConfig) -> None:
    settings = config.notifier.qq
    target = os.environ.get(settings.target_env, "").strip()
    if not target:
        raise ConfigError(
            f"required environment variable is missing: {settings.target_env}"
        )
    notifier = QQNotifier(
        EventBus(),
        settings.onebot_url,
        target,
        os.environ.get(settings.access_token_env, ""),
    )
    try:
        await notifier.send_text("AutoYKT QQ 通知连接正常")
    finally:
        await notifier.close()


def _run_command(config: AppConfig, args: argparse.Namespace) -> int:
    if args.profile:
        profile = config.page_profile(args.profile)
        if not profile.enabled:
            raise ConfigError(
                "selected profile is disabled; enable it in the "
                "private config"
            )
        config.runtime.active_profiles = [args.profile]
    if args.dry_run:
        config.runtime.dry_run = True
    if (
        not args.detect_only
        and not config.runtime.dry_run
        and config.answering.auto_apply
    ):
        require_live_configuration(config)
        if _check_config(config, announce_success=False):
            raise ConfigError("live calibration checks failed")
    with run_lease(session_directory(config)):
        return asyncio.run(
            _run_automation(config, args.detect_only, once=args.once)
        )


def _session_command(config: AppConfig, args: argparse.Namespace) -> int:
    if args.status:
        print(
            json.dumps(
                read_session_status(config), ensure_ascii=False, indent=2
            )
        )
        return 0
    if args.stop:
        print(f"Stop requested: {request_session_stop(config)}")
        print("Use --status to confirm the session has ended.")
        return 0
    # The session command explicitly requests input; overrides stay in memory.
    selected = config.model_copy(deep=True)
    selected.runtime.dry_run = args.dry_run
    selected.answering.auto_apply = not args.dry_run
    if args.profile:
        selected.runtime.active_profiles = [args.profile]
    profile_id = args.profile
    if (
        profile_id is None
        and not selected.active_page_profiles()
        and len(selected.pages) == 1
    ):
        profile_id = selected.pages[0].id
    issues = live_configuration_issues(selected, profile_id)
    # Inspect disabled candidate assets as well, without enabling real input.
    assets = selected.model_copy(deep=True)
    candidate_ids = (
        [profile_id] if profile_id else assets.runtime.active_profiles
    )
    if candidate_ids:
        for key in candidate_ids:
            assets.page_profile(key).enabled = True
        assets.runtime.active_profiles = candidate_ids
    assets_ok = _check_config(assets, announce_success=False) == 0
    if assets_ok:
        sample_count = sum(
            len(profile.calibration_samples)
            for profile in assets.active_page_profiles()
        )
        print(
            "Calibration assets are readable; "
            f"{sample_count} labeled screenshot(s) passed."
        )
    for issue in issues:
        print(f"Live setup: {issue}")
    print(f"Config: {selected.source_path}")
    print(f"Session length: {args.minutes:g} minutes")
    for provider in selected.answering.providers:
        print(
            f"Models: {provider.name}: {', '.join(provider.models)} "
            f"({provider.input_mode})"
        )
    if args.check:
        return int(bool(issues) or not assets_ok)
    if issues or not assets_ok:
        raise ConfigError(
            "class session setup is incomplete; "
            "fix the listed items before starting"
        )
    with run_lease(session_directory(selected)), keep_awake():
        return asyncio.run(
            _run_automation(selected, False, minutes=args.minutes)
        )


def _page_command(config: AppConfig, args: argparse.Namespace) -> int:
    print(
        f"Bring the target window forward; capturing in "
        f"{args.delay:g} seconds.",
        flush=True,
    )
    time.sleep(args.delay)
    if args.command == "capture-state":
        report = capture_state(config, args.profile, args.label)
        image_name = "full.png"
    elif args.command == "preview":
        report = (
            preview_step(config, args.profile, args.step)
            if args.step
            else preview_option(config, args.profile, args.option)
        )
        image_name = "plan_preview.png"
    else:
        with run_lease(session_directory(config)):
            report = recover_current(config, args.profile)
        image_name = "rearm_before.png"
        print(
            "Current page marked for skipping. "
            "Restart run; it waits for a new page."
        )
    print(f"Report: {report}")
    print(f"Image: {report.parent / image_name}")
    return 0


def _knowledge_command(config: AppConfig, args: argparse.Namespace) -> int:
    if args.knowledge_command == "ingest":
        _ingest_knowledge(config, args)
    elif args.knowledge_command == "search":
        _search_knowledge(config, args)
    elif args.knowledge_command == "record":
        asyncio.run(_record_knowledge(config, args))
    return 0


def _calibrate_command(config: AppConfig, args: argparse.Namespace) -> int:
    if args.selections and not args.from_image:
        raise ConfigError("--selections requires --from-image")
    if not args.from_state and not args.from_image:
        print(
            f"Bring the target window forward; capturing in "
            f"{args.delay:g} seconds.",
            flush=True,
        )
        time.sleep(args.delay)
    calibrator = Calibrator(
        config.source_path,
        args.profile,
        from_state=args.from_state,
        from_image=args.from_image,
    )
    if args.selections:
        selections = ImageSelections.model_validate_json(
            args.selections.read_text(encoding="utf-8")
        )
        calibrator.apply_selections(selections)
    else:
        calibrator.run()
    return 0


def _dispatch(  # pylint: disable=too-many-return-statements
    args: argparse.Namespace,
) -> int:
    if args.command == "windows":
        show_windows()
        return 0
    if args.command == "monitors":
        _show_monitors()
        return 0
    if args.command == "init":
        destination = (
            Path(args.path).expanduser().resolve()
            if args.path
            else default_user_config_path()
        )
        _init_config(destination)
        return 0

    config = _load_from_argument(getattr(args, "config", None))
    if args.command in {"session", "run"}:
        handler = {"session": _session_command, "run": _run_command}[
            args.command
        ]
        return handler(config, args)
    if args.command == "check":
        return _check_command(config, args)
    if args.command == "migrate":
        output = (
            Path(args.output).expanduser().resolve()
            if args.output
            else config.source_path.with_name("config.v2.yaml")
        )
        _migrate_config(config, output)
        return 0
    if args.command in {"capture-state", "preview", "recover"}:
        return _page_command(config, args)
    if args.command == "calibrate":
        return _calibrate_command(config, args)
    if args.command == "inspect-image":
        report = inspect_image(config, args.profile, args.image)
        print(f"Report: {report}")
        print(f"Image: {report.parent / 'annotated.png'}")
        return 0
    if args.command == "rehearse":
        report = asyncio.run(
            run_rehearsal(config, args.profile, args.image, args.output)
        )
        print(f"Simulated workflow report: {report}")
        return (
            0 if json.loads(report.read_text(encoding="utf-8"))["passed"] else 1
        )
    if args.command == "test-qq":
        asyncio.run(_test_qq(config))
        return 0
    if args.command == "knowledge":
        return _knowledge_command(config, args)
    raise AssertionError(f"unhandled command: {args.command}")


def main(arguments: list[str] | None = None) -> int:
    """Parse arguments and return a process exit code."""
    raw_arguments = list(sys.argv[1:] if arguments is None else arguments)
    parser = _build_parser()
    try:
        args = parser.parse_args(_translate_legacy_arguments(raw_arguments))
        return _dispatch(args)
    except ConfigError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2
    except (OSError, RuntimeError, ValueError, sqlite3.Error) as error:
        print(f"AutoYKT error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Stopped.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
