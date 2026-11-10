"""Configuration for the Linux/Docker service."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, PrivateAttr, field_validator, model_validator
import yaml

from autoykt.core.config import ConfigModel, ModelProviderConfig


class PlatformConfig(ConfigModel):
    """One account and one Rain Classroom environment."""

    base_url: str = "https://www.yuketang.cn"
    session_file: str = "session.json"
    poll_seconds: float = Field(default=15, ge=0.1)
    reconnect_seconds: float = Field(default=3, gt=0)
    request_timeout: float = Field(default=10, gt=0)
    classroom_ids: list[str] = Field(default_factory=list)
    ignored_lesson_ids: list[str] = Field(default_factory=list)
    auto_login: bool = True
    login_retry_seconds: float = Field(default=5, gt=0)
    handshake_timeout: float = Field(default=10, gt=0)
    metadata_wait_seconds: float = Field(default=30, gt=0)
    metadata_retry_seconds: float = Field(default=1, gt=0)

    @model_validator(mode="before")
    @classmethod
    def read_older_web_config(cls, data):
        """Accept Web settings from releases that bundled multiple entries."""
        if cls is PlatformConfig and isinstance(data, dict):
            data = dict(data)
            if data.pop("entry_mode", "web") != "web":
                raise ValueError("当前配置需要独立入口扩展。答题服务使用 Web 配置。")
            data.pop("app_session_file", None)
        return data

    @field_validator("base_url")
    @classmethod
    def validate_origin(cls, value: str) -> str:
        """Keep platform requests on a configured HTTP origin."""
        parsed = urlsplit(value)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
        ):
            raise ValueError("base_url must be an HTTP(S) origin")
        if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
            raise ValueError(
                "base_url must not include a path, query or fragment"
            )
        return value.rstrip("/")


class AnswerPolicy(ConfigModel):
    """Favor a timely usable response over repeated confidence voting."""

    mode: Literal["notify", "assist", "auto"] = "notify"
    timeout_seconds: float = Field(default=18, gt=0)
    submit_reserve_seconds: float = Field(default=3, ge=0)
    fallback: Literal["notify", "first_option"] = "notify"
    precompute: bool = True
    previous_slides: int = Field(default=3, ge=0, le=12)
    context_characters: int = Field(default=12000, ge=100)
    maximum_parallel: int = Field(default=2, ge=1, le=16)
    providers: list[ModelProviderConfig] = Field(default_factory=list)
    auto_from: datetime | None = None
    auto_until: datetime | None = None

    @field_validator("auto_from", "auto_until")
    @classmethod
    def aware_automatic_window(cls, value):
        if value is not None and value.utcoffset() is None:
            raise ValueError("automatic answering timestamps require timezone")
        return value

    @property
    def automatic_window_open(self):
        now = datetime.now(timezone.utc)
        return (self.auto_from is None or self.auto_from <= now) and (
            self.auto_until is None or now < self.auto_until
        )


class CoursewareConfig(ConfigModel):
    """Fetch all account-visible decks without delaying classroom events."""

    enabled: bool = True
    download_images: bool = True
    refresh_seconds: float = Field(default=60, ge=1)


class ControlConfig(ConfigModel):
    """Optional loopback-only operations interface."""

    enabled: bool = False
    port: int = Field(default=18767, ge=1, le=65535)
    token_env: str = "AUTOYKT_WEBHOOK_TOKEN"


class WebhookConfig(ConfigModel):
    """Optional generic JSON POST integration."""

    url: str | None = None
    token_env: str = "AUTOYKT_WEBHOOK_TOKEN"
    timeout_seconds: float = Field(default=5, gt=0)
    attempts: int = Field(default=3, ge=1, le=5)
    queue_size: int = Field(default=1000, ge=1)
    events: list[str] = Field(default_factory=list)
    retention_seconds: float = Field(default=300, gt=0)
    inline_question_images: bool = True
    inline_image_max_bytes: int = Field(
        default=2_000_000, ge=1024, le=20_000_000
    )

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str | None) -> str | None:
        """Require an explicit HTTP endpoint without embedded credentials."""
        if value is not None:
            parsed = urlsplit(value)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.hostname
                or parsed.username
                or parsed.password
            ):
                raise ValueError("webhook url must be an HTTP(S) URL")
        return value


class DiagnosticsConfig(ConfigModel):
    """Temporarily retain message shapes for an unattended live test."""

    capture_from: datetime | None = None
    capture_until: datetime | None = None
    max_bytes: int = Field(default=5_000_000, ge=1024, le=20_000_000)

    @field_validator("capture_from", "capture_until")
    @classmethod
    def require_timezone(cls, value: datetime | None) -> datetime | None:
        """Make an unattended test window independent of the host timezone."""
        if value is not None and value.utcoffset() is None:
            raise ValueError("diagnostic timestamps require a timezone")
        return value


class ServerConfig(ConfigModel):
    """Standalone schema, independent of visual page calibration."""

    version: Literal[1] = 1
    data_dir: str = "data"
    platform: PlatformConfig = Field(default_factory=PlatformConfig)
    answering: AnswerPolicy = Field(default_factory=AnswerPolicy)
    courseware: CoursewareConfig = Field(default_factory=CoursewareConfig)
    control: ControlConfig = Field(default_factory=ControlConfig)
    webhook: WebhookConfig = Field(default_factory=WebhookConfig)
    diagnostics: DiagnosticsConfig = Field(default_factory=DiagnosticsConfig)
    _directory: Path = PrivateAttr(default_factory=Path.cwd)

    def resolve(self, path: str) -> Path:
        """Resolve private paths relative to the configuration file."""
        candidate = Path(path).expanduser()
        return (self._directory / candidate).resolve()

    @property
    def storage(self) -> Path:
        """Return the persistent service data directory."""
        return self.resolve(self.data_dir)


def load_server_config(
    path: str | Path, *, schema: type[ServerConfig] = ServerConfig
) -> ServerConfig:
    """Load without desktop dependencies or environment mutation."""
    source = Path(path).expanduser().resolve()
    config = schema.model_validate(
        yaml.safe_load(source.read_text(encoding="utf-8"))
    )
    config._directory = source.parent  # pylint: disable=protected-access
    return config
