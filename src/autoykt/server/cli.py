"""Standalone CLI that never imports screen capture or desktop automation."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
from importlib import resources
import json
import logging
import os
from pathlib import Path
import signal
import sys
import time
from uuid import uuid4

from autoykt.server.cache import ClassroomCache
from autoykt.server.client import YuketangClient
from autoykt.server.config import load_server_config
from autoykt.server.control import ControlServer
from autoykt.server.courseware import export_courseware
from autoykt.server.events import EventWriter
from autoykt.server.service import ClassroomService
from autoykt.server.login import publish_ticket
from autoykt.server.solver import AssetStore


def build_parser() -> argparse.ArgumentParser:
    """Expose read-only probing separately from active classroom control."""
    parser = argparse.ArgumentParser(prog="autoykt-server")
    parser.add_argument(
        "--config",
        default=os.environ.get("AUTOYKT_SERVER_CONFIG", "config.server.yaml"),
    )
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="create a server configuration")
    init.add_argument("--path", default="config.server.yaml")
    commands.add_parser(
        "check", help="validate configuration without connecting"
    )
    commands.add_parser(
        "probe", help="read session and active-classroom endpoints"
    )
    commands.add_parser(
        "login", help="save a QR login session without a browser"
    )
    commands.add_parser(
        "run", help="discover classrooms and run the configured answer mode"
    )
    commands.add_parser("health", help="check the local service heartbeat")
    commands.add_parser(
        "notify-test", help="send a test through the configured webhook"
    )
    cache = commands.add_parser(
        "cache", help="import/export slide-indexed answers"
    )
    cache.add_argument("action", choices=["import", "export"])
    cache.add_argument("file")
    cache.add_argument("--course", required=True)
    cache.add_argument("--presentation", required=True)
    courseware = commands.add_parser(
        "courseware", help="export cached courseware and prepared answers"
    )
    courseware.add_argument("action", choices=["export"])
    courseware.add_argument("directory")
    courseware.add_argument("--course", required=True)
    return parser


async def execute(
    args,
    config,
    *,
    client_type=YuketangClient,
    service_type=ClassroomService,
    control_type=ControlServer,
) -> int:
    """Connect only for login, probe and run."""
    events = EventWriter(config)
    await events.start()
    try:
        if args.command == "notify-test":
            if not config.webhook.url:
                raise ValueError(
                    "configure webhook.url before testing phone delivery"
                )
            if (
                config.webhook.events
                and "notification_test" not in config.webhook.events
            ):
                raise ValueError("webhook.events excludes notification_test")
            event = await events.emit(
                "notification_test",
                payload={
                    "message": "AutoYKT 通知链路测试，请在手机上确认收到。"
                },
            )
            await events.wait_delivered(
                event["event_id"],
                timeout=max(10, config.webhook.timeout_seconds * 2 + 2),
            )
            print(
                json.dumps(
                    {
                        "event_id": event["event_id"],
                        "webhook_accepted": True,
                        "phone_receipt": "confirm_on_phone",
                    }
                )
            )
            return 0
        async with client_type(config) as client:
            if args.command == "probe":
                result = await client.probe()
                await events.emit("probe_result", payload=result)
                print(json.dumps(result, ensure_ascii=False))
                return 0 if result["ok"] else 1
            if args.command == "login":

                async def on_ticket(url):
                    await publish_ticket(
                        config,
                        client,
                        events,
                        url,
                        attempt_id=attempt,
                        expires_at=expires,
                    )
                    print(f"Scan QR: {url}", flush=True)

                attempt, expires = uuid4().hex, time.time() + 120
                await client.login(on_ticket)
                await events.emit("login_succeeded")
                print(f"Session saved: {client.session_path}")
                return 0
            service = service_type(config, client, events)
            controller = control_type(service)
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                try:
                    loop.add_signal_handler(sig, service.stop.set)
                except NotImplementedError:
                    pass
            try:
                await controller.start()
                await service.run()
            finally:
                await controller.close()
                await service.close()
                await events.emit("service_stopped")
        return 0
    finally:
        await events.close()


# Each CLI command has its own exit result.
# pylint: disable-next=too-many-return-statements
def main(argv: list[str] | None = None) -> int:
    """Execute the service CLI without exposing configuration secrets."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        if args.command == "init":
            target = Path(args.path).expanduser().resolve()
            target.parent.mkdir(parents=True, exist_ok=True)
            template = resources.files("autoykt.resources").joinpath(
                "config.server.example.yaml"
            )
            with target.open("x", encoding="utf-8") as stream:
                stream.write(template.read_text(encoding="utf-8"))
            print(f"Config: {target}")
            return 0
        config = load_server_config(args.config)
        if args.command == "check":
            print(
                json.dumps(
                    {
                        "valid": True,
                        "mode": config.answering.mode,
                        "entry_mode": "web",
                        "domain": config.platform.base_url,
                        "webhook": bool(config.webhook.url),
                        "courseware": config.courseware.enabled,
                        "precompute": config.answering.precompute,
                        "control": config.control.enabled,
                    },
                    ensure_ascii=False,
                )
            )
            return 0
        if args.command == "health":
            path = config.storage / "status.json"
            status = json.loads(path.read_text(encoding="utf-8"))
            age = (
                datetime.now(timezone.utc)
                - datetime.fromisoformat(status["updated_at"])
            ).total_seconds()
            print(json.dumps(status))
            return (
                0
                if status["status"] == "running"
                and age
                < max(
                    90,
                    config.platform.poll_seconds * 3
                    + config.platform.request_timeout * 2,
                )
                else 1
            )
        if args.command in {"cache", "courseware"}:
            cache = ClassroomCache(
                config.storage / "classroom.sqlite3", config.platform.base_url
            )
            try:
                if args.command == "courseware":
                    manifest = export_courseware(
                        cache,
                        AssetStore(config.storage / "images", None),
                        args.course,
                        Path(args.directory),
                    )
                    print(
                        json.dumps(
                            {
                                "presentations": len(manifest["presentations"]),
                                "directory": str(
                                    Path(args.directory).resolve()
                                ),
                            }
                        )
                    )
                    return 0
                file = Path(args.file)
                if args.action == "import":
                    count = cache.import_answers(
                        args.course,
                        args.presentation,
                        json.loads(file.read_text(encoding="utf-8")),
                    )
                    print(f"Imported answers: {count}")
                else:
                    file.write_text(
                        json.dumps(
                            cache.export_answers(
                                args.course, args.presentation
                            ),
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
            finally:
                cache.close()
            return 0
        return asyncio.run(execute(args, config))
    except KeyboardInterrupt:
        return 130
    except Exception as error:  # pylint: disable=broad-exception-caught
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1
