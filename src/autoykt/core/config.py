"""Shared validation for service and model configuration."""

from __future__ import annotations

import os
import re
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator


class ConfigError(ValueError):
    """Raised when an AutoYKT configuration cannot be loaded."""


class ConfigModel(BaseModel):
    """Base model shared by strict service configuration sections."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        allow_inf_nan=False,
        hide_input_in_errors=True,
    )


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
