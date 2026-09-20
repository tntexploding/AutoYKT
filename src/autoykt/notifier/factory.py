"""Create configured notification observers without embedded credentials."""

from __future__ import annotations

import os

from autoykt.core.config import AppConfig, ConfigError
from autoykt.core.event_bus import EventBus
from autoykt.notifier.base import BaseNotifier
from autoykt.notifier.qq_bot import QQNotifier
from autoykt.notifier.telegram_bot import TelegramNotifier


def create_notifiers(
    config: AppConfig,
    event_bus: EventBus,
) -> list[BaseNotifier]:
    """Create enabled notifiers after resolving private environment values."""
    notifiers: list[BaseNotifier] = []
    for backend in config.notifier.enabled:
        if backend == "qq":
            settings = config.notifier.qq
            target = _required_environment(settings.target_env)
            token = os.environ.get(settings.access_token_env, "")
            notifiers.append(
                QQNotifier(
                    event_bus=event_bus,
                    onebot_url=settings.onebot_url,
                    target_qq=target,
                    access_token=token,
                )
            )
        elif backend == "telegram":
            settings = config.notifier.telegram
            notifiers.append(
                TelegramNotifier(
                    event_bus=event_bus,
                    token=_required_environment(settings.token_env),
                    chat_id=_required_environment(settings.chat_id_env),
                )
            )
    return notifiers


def _required_environment(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ConfigError(f"required environment variable is missing: {name}")
    return value
