"""Typed configuration loading and legacy configuration migration."""

from __future__ import annotations

import os
import re
import warnings
from pathlib import Path
from typing import Annotated, Any, Literal, cast

import yaml
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    SecretStr,
    field_validator,
    model_validator,
)

from autoykt.core.legacy_config import (
    LegacyConfigError,
    migrate_legacy_config,
)


class ConfigError(ValueError):
    """Raised when an AutoYKT configuration cannot be loaded."""


def _validate_region(
    region: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    if region[2] <= 0 or region[3] <= 0:
        raise ValueError("region width and height must be greater than zero")
    return region


Region = Annotated[
    tuple[int, int, int, int],
    AfterValidator(_validate_region),
]
Point = tuple[int, int]
CoordinateSpace = Literal["monitor", "screen", "window"]
DetectionAction = Literal["answer", "notify"]
QuestionType = Literal["single", "multiple"]


def _validate_environment_name(value: str) -> str:
    normalized = value.strip()
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", normalized):
        raise ValueError("value must be an environment variable name")
    return normalized


EnvironmentName = Annotated[str, AfterValidator(_validate_environment_name)]


class ConfigModel(BaseModel):
    """Base model shared by strict v2 configuration sections."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        allow_inf_nan=False,
        hide_input_in_errors=True,
    )


class StorageConfig(ConfigModel):
    """Paths for private runtime artifacts."""

    data_dir: str = "data"
    log_dir: str = "logs"
    screenshot_dir: str = "screenshots"


class RuntimeConfig(ConfigModel):
    """Process-wide runtime settings."""

    active_profiles: list[str] = Field(default_factory=list)
    poll_interval_seconds: float = Field(default=0.5, gt=0)
    dry_run: bool = True


class ImageTemplateConfig(ConfigModel):
    """One image template used for a single-frame state check."""

    path: str
    threshold: float = Field(default=0.85, ge=0.0, le=1.0)

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("template path cannot be empty")
        return normalized


class WindowTargetConfig(ConfigModel):
    """A private Windows window selector and calibrated client dimensions."""

    title_pattern: str = Field(min_length=1)
    class_name: str | None = None
    client_size: tuple[int, int]
    background_monitoring: bool = False
    focus_point: Point | None = None

    @field_validator("title_pattern")
    @classmethod
    def _validate_pattern(cls, value: str) -> str:
        try:
            re.compile(value)
        except re.error as error:
            raise ValueError(
                "invalid window title regular expression"
            ) from error
        return value

    @field_validator("client_size")
    @classmethod
    def _validate_size(cls, value: tuple[int, int]) -> tuple[int, int]:
        if min(value) <= 0:
            raise ValueError("window client dimensions must be positive")
        return value

    @model_validator(mode="after")
    def _validate_focus_point(self) -> "WindowTargetConfig":
        if self.focus_point is not None:
            x, y = self.focus_point
            width, height = self.client_size
            if not (0 <= x < width and 0 <= y < height):
                raise ValueError(
                    "focus point lies outside the window client area"
                )
            if not self.background_monitoring:
                raise ValueError("focus point requires background_monitoring")
        return self


class PageGuardConfig(ImageTemplateConfig):
    """A stable page identity crop, relative to the capture surface."""

    region: Region
    match_grayscale: bool = False
    scales: list[Annotated[float, Field(gt=0, le=4)]] = Field(
        default_factory=lambda: [1.0], min_length=1, max_length=16
    )


class TriggerTemplateConfig(ImageTemplateConfig):
    """A debounced template that starts an automation action."""

    name: str
    consecutive_hits: int = Field(default=3, ge=1)
    action: DetectionAction = "answer"
    question_type: QuestionType = "single"

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        normalized = value.strip()
        if not re.fullmatch(r"[a-zA-Z0-9_.-]{1,64}", normalized):
            raise ValueError(
                "template name may contain at most 64 letters, numbers, "
                "'.', '_', and '-'"
            )
        return normalized


class ClickTargetConfig(ConfigModel):
    """A click point or region center in a declared coordinate space."""

    point: Point | None = None
    region: Region | None = None
    coordinate_space: CoordinateSpace = "monitor"

    @model_validator(mode="after")
    def _validate_target(self) -> "ClickTargetConfig":
        if (self.point is None) == (self.region is None):
            raise ValueError(
                "exactly one of point or region must be configured"
            )
        return self


class PageRegionsConfig(ConfigModel):
    """Regions relative to the monitor or bound window client area."""

    detection: Region
    question: Region
    answers: Region | None = None
    verification: Region | None = None
    rearm: Region | None = None

    @property
    def answer_region(self) -> Region:
        """Return the answer-search region, falling back to the question."""
        return self.answers or self.question

    @property
    def verification_region(self) -> Region:
        """Return the post-submit region, falling back to the question."""
        return self.verification or self.question

    @property
    def rearm_region(self) -> Region:
        """Return the next-question region, falling back to the question."""
        return self.rearm or self.question


def _normalize_option_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("option mapping must be a dictionary")
    normalized: dict[str, Any] = {}
    for raw_key, item in value.items():
        key = str(raw_key).strip().upper()
        if not re.fullmatch(r"[A-Z0-9_-]{1,16}", key):
            raise ValueError(
                "option key may contain at most 16 letters, numbers, "
                "'_', and '-'"
            )
        if key in normalized:
            raise ValueError(f"duplicate option key: {key}")
        normalized[key] = item
    return normalized


ColorChannel = Annotated[int, Field(ge=0, le=255)]
RGBColor = tuple[ColorChannel, ColorChannel, ColorChannel]


class ButtonColorsConfig(ConfigModel):
    """Calibrated background colors for letter buttons with white glyphs."""

    selected_rgb: RGBColor
    unselected_rgb: RGBColor
    tolerance: int = Field(default=20, ge=0, le=127)
    button_size: tuple[int, int] | None = None

    @model_validator(mode="after")
    def _validate_separation(self) -> "ButtonColorsConfig":
        if self.button_size is not None and min(self.button_size) <= 0:
            raise ValueError("button dimensions must be positive")
        if (
            max(
                abs(first - second)
                for first, second in zip(self.selected_rgb, self.unselected_rgb)
            )
            <= 2 * self.tolerance
        ):
            raise ValueError("button color tolerance ranges overlap")
        return self


class AnswerStyleConfig(ConfigModel):
    """How answer options are located for one page profile."""

    option_templates: dict[str, str] = Field(default_factory=dict)
    option_match_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    selection_timeout_seconds: float = Field(default=2.0, gt=0)
    match_grayscale: bool = False
    button_colors: ButtonColorsConfig | None = None
    fallback_positions: dict[str, Point] = Field(default_factory=dict)
    fallback_coordinate_space: CoordinateSpace = "monitor"

    @field_validator("option_templates", "fallback_positions", mode="before")
    @classmethod
    def _normalize_options(cls, value: Any) -> dict[str, Any]:
        return _normalize_option_mapping(value)

    @model_validator(mode="after")
    def _validate_available_options(self) -> "AnswerStyleConfig":
        if not self.option_templates and not self.fallback_positions:
            raise ValueError(
                "at least one option template or fallback position is required"
            )
        return self

    @property
    def option_keys(self) -> frozenset[str]:
        """Return every configured answer key."""
        return frozenset(self.option_templates) | frozenset(
            self.fallback_positions
        )


class QuestionReadyConfig(ConfigModel):
    """Settling rules used after an optional entry click."""

    delay_seconds: float = Field(default=0.3, ge=0.0)
    timeout_seconds: float = Field(default=3.0, gt=0.0)
    stable_frames: int = Field(default=2, ge=1)
    maximum_change_ratio: float = Field(default=0.01, ge=0.0, le=1.0)


class VerificationConfig(ConfigModel):
    """Rules for proving that an answer submission succeeded."""

    success_templates: list[ImageTemplateConfig] = Field(default_factory=list)
    failure_templates: list[ImageTemplateConfig] = Field(default_factory=list)
    require_success_template: bool = False
    timeout_seconds: float = Field(default=5.0, gt=0.0)
    poll_interval_seconds: float = Field(default=0.25, gt=0.0)
    stable_hits: int = Field(default=2, ge=1)
    minimum_change_ratio: float = Field(default=0.02, gt=0.0, le=1.0)


class RearmConfig(ConfigModel):
    """Rules for recognizing that another question is ready."""

    timeout_seconds: float = Field(default=30.0, gt=0.0)
    stable_hits: int = Field(default=2, ge=1)
    minimum_change_ratio: float = Field(default=0.02, gt=0.0, le=1.0)


class PageAdvanceConfig(ConfigModel):
    """One recognized result-page button, clicked at most once per question."""

    name: str = Field(pattern=r"^[a-zA-Z0-9_.-]{1,64}$")
    when: PageGuardConfig
    target: ClickTargetConfig | None = None
    click_match: bool = False
    stable_hits: int = Field(default=2, ge=1)

    @model_validator(mode="after")
    def _validate_click(self) -> "PageAdvanceConfig":
        if self.click_match == (self.target is not None):
            raise ValueError("choose exactly one of target or click_match")
        return self


class PageFlowConfig(ConfigModel):
    """Visual states outside the answer and submit controls."""

    before_question: list[PageAdvanceConfig] = Field(default_factory=list)
    after_submit: list[PageAdvanceConfig] = Field(default_factory=list)
    loading: list[PageGuardConfig] = Field(default_factory=list)
    manual: list[PageGuardConfig] = Field(default_factory=list)
    finished: list[PageGuardConfig] = Field(default_factory=list)
    transition_timeout_seconds: float = Field(default=10.0, gt=0)

    @model_validator(mode="after")
    def _validate_names(self) -> "PageFlowConfig":
        names = [
            step.name for step in [*self.before_question, *self.after_submit]
        ]
        if len(names) != len(set(names)):
            raise ValueError("page action names must be unique")
        return self

    @property
    def templates(self) -> list[PageGuardConfig]:
        """Return all regional state templates for validation and migration."""
        return [
            *(step.when for step in self.before_question),
            *(step.when for step in self.after_submit),
            *self.loading,
            *self.manual,
            *self.finished,
        ]


class QuestionBudgetConfig(ConfigModel):
    """Time limits from trigger detection to verified submission."""

    total_seconds: float = Field(default=45.0, gt=0)
    context_seconds: float = Field(default=5.0, gt=0)
    answer_seconds: float = Field(default=25.0, gt=0)


class RecoveryConfig(ConfigModel):
    """Bounded retries before any answer-selection input has been attempted."""

    maximum_attempts: int = Field(default=2, ge=1, le=5)
    delay_seconds: float = Field(default=1.0, ge=0)


class CalibrationSampleConfig(ConfigModel):
    """Private labeled screenshots checked before an unattended run."""

    path: str = Field(min_length=1)
    expected_state: Literal[
        "question",
        "completed",
        "failure",
        "unrecognized",
        "notice",
        "finished",
        "loading",
        "manual",
    ]
    question_type: QuestionType | None = None
    expected_options: list[str] | None = None
    selected_options: list[str] | None = None

    @field_validator("expected_options", "selected_options")
    @classmethod
    def _validate_keys(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        keys = [key.strip().upper() for key in value]
        if len(keys) != len(set(keys)) or any(
            not re.fullmatch(r"[A-Z0-9]", key) for key in keys
        ):
            raise ValueError(
                "sample option keys must be unique letters or digits"
            )
        return keys

    @model_validator(mode="after")
    def _validate_selected(self) -> "CalibrationSampleConfig":
        if self.selected_options is not None and (
            self.expected_options is None
            or not set(self.selected_options) <= set(self.expected_options)
        ):
            raise ValueError(
                "selected sample options require matching expected_options"
            )
        return self


class PageProfileConfig(ConfigModel):
    """Automation settings for one quiz website or visual layout."""

    id: str
    display_name: str
    course_id: str = "default"
    enabled: bool = True
    monitor_index: int = Field(default=1, ge=0)
    target_window: WindowTargetConfig | None = None
    page_guard: PageGuardConfig | None = None
    regions: PageRegionsConfig
    triggers: list[TriggerTemplateConfig]
    answer_style: AnswerStyleConfig
    calibration_samples: list[CalibrationSampleConfig] = Field(
        default_factory=list
    )
    entry_target: ClickTargetConfig | None = None
    submit_target: ClickTargetConfig | None = None
    submit_on_select: bool = False
    question_ready: QuestionReadyConfig = Field(
        default_factory=QuestionReadyConfig
    )
    verification: VerificationConfig = Field(default_factory=VerificationConfig)
    rearm: RearmConfig = Field(default_factory=RearmConfig)
    page_flow: PageFlowConfig = Field(default_factory=PageFlowConfig)
    question_budget: QuestionBudgetConfig = Field(
        default_factory=QuestionBudgetConfig
    )
    recovery: RecoveryConfig = Field(default_factory=RecoveryConfig)

    @field_validator("id", "course_id")
    @classmethod
    def _validate_identifier(cls, value: str) -> str:
        normalized = value.strip()
        if not re.fullmatch(r"[a-zA-Z0-9_.-]{1,64}", normalized):
            raise ValueError(
                "identifier may contain at most 64 letters, numbers, '.', "
                "'_', and '-'"
            )
        return normalized

    @field_validator("display_name")
    @classmethod
    def _validate_display_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("display_name cannot be empty")
        return normalized

    @model_validator(mode="after")
    def _validate_triggers(self) -> "PageProfileConfig":
        if not self.triggers:
            raise ValueError("at least one trigger template is required")
        names = [trigger.name for trigger in self.triggers]
        if len(names) != len(set(names)):
            raise ValueError("trigger template names must be unique")
        flow = self.page_flow
        # Pylint sees FieldInfo for this default factory inside validators.
        steps = [*flow.before_question, *flow.after_submit]  # pylint: disable=no-member
        state_templates = flow.templates  # pylint: disable=no-member
        targets = [self.entry_target, self.submit_target]
        targets.extend(step.target for step in steps)
        spaces = [
            target.coordinate_space for target in targets if target is not None
        ]
        if self.answer_style.fallback_positions:
            spaces.append(self.answer_style.fallback_coordinate_space)
        if self.target_window:
            if any(space != "window" for space in spaces):
                raise ValueError(
                    "bound window click targets must use window coordinates"
                )
            width, height = self.target_window.client_size
            regions = [
                value for value in self.regions.model_dump().values() if value
            ]
            if self.page_guard:
                regions.append(self.page_guard.region)
            regions.extend(template.region for template in state_templates)
            regions.extend(
                target.region for target in targets if target and target.region
            )
            if any(
                x < 0 or y < 0 or x + w > width or y + h > height
                for x, y, w, h in regions
            ):
                raise ValueError(
                    "region lies outside the calibrated window client area"
                )
            points = list(self.answer_style.fallback_positions.values())
            points.extend(
                target.point for target in targets if target and target.point
            )
            if any(not (0 <= x < width and 0 <= y < height) for x, y in points):
                raise ValueError(
                    "click point lies outside the calibrated window client area"
                )
        elif "window" in spaces:
            raise ValueError("window coordinates require target_window")
        return self


class ModelProviderConfig(ConfigModel):
    """One OpenAI-compatible endpoint and its configured models."""

    name: str
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str | None = "OPENAI_API_KEY"
    api_key: SecretStr | None = Field(default=None, exclude=True, repr=False)
    models: list[str]
    timeout_seconds: float = Field(default=30.0, gt=0.0)
    max_output_tokens: int = Field(default=2048, ge=1)
    request_json_object: bool = False
    input_mode: Literal["text", "vision"] = "vision"
    thinking: Literal["enabled", "disabled"] | None = None
    reasoning_effort: (
        Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]
        | None
    ) = None

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("provider name cannot be empty")
        return normalized

    @field_validator("models")
    @classmethod
    def _validate_models(cls, value: list[str]) -> list[str]:
        models = [model.strip() for model in value if model.strip()]
        if not models:
            raise ValueError("at least one model must be configured")
        if len(models) != len(set(models)):
            raise ValueError("model names must be unique within a provider")
        return models

    @field_validator("api_key_env")
    @classmethod
    def _validate_environment_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", normalized):
            raise ValueError("api_key_env must be an environment variable name")
        return normalized

    @field_validator("api_key")
    @classmethod
    def _validate_api_key(cls, value: SecretStr | None) -> SecretStr | None:
        if value is None:
            return None
        normalized = value.get_secret_value().strip()
        if not normalized:
            raise ValueError("api_key cannot be empty; use api_key_env instead")
        return SecretStr(normalized)

    def resolve_api_key(self) -> str:
        """Return the configured secret without exposing it in model dumps."""
        if self.api_key is not None:
            api_key = cast(SecretStr, self.api_key)
            return api_key.get_secret_value()  # pylint: disable=no-member
        if self.api_key_env:
            value = os.environ.get(self.api_key_env, "").strip()
            if value:
                return value
            raise ConfigError(
                f"provider '{self.name}' requires environment variable "
                f"{self.api_key_env}"
            )
        raise ConfigError(f"provider '{self.name}' has no API key source")


class AnsweringConfig(ConfigModel):
    """Multi-model answering and consensus policy."""

    providers: list[ModelProviderConfig]
    minimum_responses: int = Field(default=1, ge=1)
    minimum_agreement: int = Field(default=1, ge=1)
    minimum_confidence: float | None = Field(default=0.6, ge=0.0, le=1.0)
    maximum_parallel_requests: int = Field(default=4, ge=1)
    maximum_rounds: int = Field(default=1, ge=1, le=7)
    initial_rounds: int = Field(default=1, ge=1, le=7)
    auto_apply: bool = False
    prompt_template: str | None = None

    @model_validator(mode="after")
    def _validate_consensus(self) -> "AnsweringConfig":
        names = [provider.name for provider in self.providers]
        if len(names) != len(set(names)):
            raise ValueError("provider names must be unique")
        request_count = (
            sum(len(provider.models) for provider in self.providers)
            * self.maximum_rounds
        )
        if self.initial_rounds > self.maximum_rounds:
            raise ValueError("initial_rounds cannot exceed maximum_rounds")
        if request_count == 0:
            raise ValueError("at least one answering model is required")
        if self.minimum_responses > request_count:
            raise ValueError(
                "minimum_responses cannot exceed the configured sample count"
            )
        if self.minimum_agreement > request_count:
            raise ValueError(
                "minimum_agreement cannot exceed the configured sample count"
            )
        return self


class KnowledgeConfig(ConfigModel):
    """Local course knowledge retrieval settings."""

    enabled: bool = False
    database_path: str = "data/knowledge.db"
    question_ocr_enabled: bool = True
    maximum_results: int = Field(default=4, ge=1)
    minimum_score: float = Field(default=0.08, ge=0.0, le=1.0)
    maximum_context_characters: int = Field(default=5000, ge=100)
    chunk_size_characters: int = Field(default=800, ge=100)
    chunk_overlap_characters: int = Field(default=100, ge=0)

    @model_validator(mode="after")
    def _validate_chunk_overlap(self) -> "KnowledgeConfig":
        if self.chunk_overlap_characters >= self.chunk_size_characters:
            raise ValueError("chunk overlap must be smaller than chunk size")
        return self


class QQNotifierConfig(ConfigModel):
    """Private QQ settings resolved from environment variables."""

    onebot_url: str = "http://127.0.0.1:3001"
    target_env: EnvironmentName = "AUTOYKT_QQ_TARGET"
    access_token_env: EnvironmentName = "AUTOYKT_ONEBOT_ACCESS_TOKEN"


class TelegramNotifierConfig(ConfigModel):
    """Private Telegram settings resolved from environment variables."""

    token_env: EnvironmentName = "AUTOYKT_TELEGRAM_TOKEN"
    chat_id_env: EnvironmentName = "AUTOYKT_TELEGRAM_CHAT_ID"


class NotifierConfig(ConfigModel):
    """Enabled remote notification backends."""

    enabled: list[Literal["qq", "telegram"]] = Field(default_factory=list)
    qq: QQNotifierConfig = Field(default_factory=QQNotifierConfig)
    telegram: TelegramNotifierConfig = Field(
        default_factory=TelegramNotifierConfig
    )

    @field_validator("enabled")
    @classmethod
    def _validate_enabled(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("notification backends cannot contain duplicates")
        return value


class LoggingConfig(ConfigModel):
    """Application logging settings."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    @field_validator("level", mode="before")
    @classmethod
    def _normalize_level(cls, value: Any) -> str:
        return str(value).strip().upper()


class AppConfig(ConfigModel):
    """Complete AutoYKT v2 configuration."""

    version: Literal[2] = 2
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    answering: AnsweringConfig
    knowledge: KnowledgeConfig = Field(default_factory=KnowledgeConfig)
    pages: list[PageProfileConfig]
    notifier: NotifierConfig = Field(default_factory=NotifierConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    _source_path: Path = PrivateAttr(default=Path("config.yaml"))
    _migrated_from_legacy: bool = PrivateAttr(default=False)

    @model_validator(mode="after")
    def _validate_profiles(self) -> "AppConfig":
        if not self.pages:
            raise ValueError("at least one page profile is required")
        profile_ids = [profile.id for profile in self.pages]
        if len(profile_ids) != len(set(profile_ids)):
            raise ValueError("page profile ids must be unique")
        runtime = cast(RuntimeConfig, self.runtime)
        active_profiles = runtime.active_profiles  # pylint: disable=no-member
        if len(active_profiles) != len(set(active_profiles)):
            raise ValueError("active_profiles cannot contain duplicates")
        unknown = set(active_profiles) - set(profile_ids)
        if unknown:
            raise ValueError(
                "active_profiles contains unknown profiles: "
                + ", ".join(sorted(unknown))
            )
        return self

    @property
    def source_path(self) -> Path:
        """Return the absolute path from which this config was loaded."""
        return self._source_path

    @property
    def migrated_from_legacy(self) -> bool:
        """Whether an unversioned v1 document was migrated in memory."""
        return self._migrated_from_legacy

    def resolve_path(self, value: str) -> Path:
        """Resolve a configured path relative to the private config file."""
        path = Path(os.path.expandvars(os.path.expanduser(value)))
        if not path.is_absolute():
            path = self._source_path.parent / path
        return path.resolve()

    def active_page_profiles(self) -> list[PageProfileConfig]:
        """Return enabled profiles selected by the runtime configuration."""
        runtime = cast(RuntimeConfig, self.runtime)
        selected = set(runtime.active_profiles)  # pylint: disable=no-member
        return [
            profile
            for profile in self.pages
            if profile.enabled and (not selected or profile.id in selected)
        ]

    def page_profile(self, profile_id: str) -> PageProfileConfig:
        """Return one page profile or raise a configuration error."""
        for profile in self.pages:
            if profile.id == profile_id:
                return profile
        raise ConfigError(f"unknown page profile: {profile_id}")

    def bind_source(self, path: Path, migrated_from_legacy: bool) -> None:
        """Attach loader metadata that is intentionally absent from YAML."""
        self._source_path = path
        self._migrated_from_legacy = migrated_from_legacy


def load_config(path: str | Path) -> AppConfig:
    """Load and validate an AutoYKT configuration.

    Args:
        path: YAML path. Relative paths inside the document are resolved
            against the configuration file's directory.

    Returns:
        The validated v2 application configuration.

    Raises:
        ConfigError: The file is missing, malformed, or invalid.
    """
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigError(f"config file not found: {config_path}")
    try:
        with config_path.open("r", encoding="utf-8") as config_file:
            raw = yaml.safe_load(config_file)
    except yaml.YAMLError as error:
        # YAML's default exception includes private source lines.
        mark = getattr(error, "problem_mark", None)
        location = f" at line {mark.line + 1}" if mark is not None else ""
        raise ConfigError(f"invalid YAML{location}") from None
    except (OSError, UnicodeError) as error:
        raise ConfigError(f"failed to read config: {error}") from error
    if not isinstance(raw, dict):
        raise ConfigError("config root must be a YAML mapping")

    migrated = "version" not in raw and "pages" not in raw
    try:
        document = migrate_legacy_config(raw) if migrated else raw
    except LegacyConfigError as error:
        raise ConfigError(str(error)) from error
    try:
        config = AppConfig.model_validate(document)
    except ValueError as error:
        raise ConfigError(str(error)) from error
    config.bind_source(config_path, migrated)
    if migrated:
        warnings.warn(
            "Loaded a legacy AutoYKT config in compatibility mode. "
            "Run 'autoykt migrate' to create an explicit v2 config.",
            UserWarning,
            stacklevel=2,
        )
    return config


def default_user_config_path() -> Path:
    """Return a platform-appropriate private configuration path."""
    explicit_home = os.environ.get("AUTOYKT_CONFIG_HOME", "").strip()
    if explicit_home:
        return Path(explicit_home).expanduser() / "config.yaml"
    if os.name == "nt":
        app_data = os.environ.get("LOCALAPPDATA", "").strip()
        if app_data:
            return Path(app_data) / "AutoYKT" / "config.yaml"
    xdg_config = os.environ.get("XDG_CONFIG_HOME", "").strip()
    if xdg_config:
        return Path(xdg_config) / "autoykt" / "config.yaml"
    return Path.home() / ".config" / "autoykt" / "config.yaml"


def discover_config_path(explicit_path: str | None = None) -> Path:
    """Find a private config without treating the example as runnable."""
    if explicit_path:
        return Path(explicit_path).expanduser().resolve()
    environment_path = os.environ.get("AUTOYKT_CONFIG", "").strip()
    if environment_path:
        return Path(environment_path).expanduser().resolve()
    user_path = default_user_config_path()
    if user_path.is_file():
        return user_path.resolve()
    raise ConfigError(
        f"no private config found at {user_path}; run 'autoykt init' "
        "or pass --config explicitly; working-directory configs are not "
        "loaded automatically"
    )
