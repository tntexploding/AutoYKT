"""Factory for creating notifier instances from config."""

import logging

from autoykt.core.config import AppConfig
from autoykt.core.event_bus import EventBus
from autoykt.notifier.base import BaseNotifier

logger = logging.getLogger("auto_answer")


def create_notifiers(config: AppConfig, event_bus: EventBus) -> list[BaseNotifier]:
    """Create and return notifier instances based on config.enabled list."""
    notifiers: list[BaseNotifier] = []

    for backend in config.notifier.enabled:
        try:
            if backend == "telegram":
                from autoykt.notifier.telegram_bot import TelegramNotifier
                n = TelegramNotifier(
                    event_bus=event_bus,
                    token=config.notifier.telegram.token,
                    chat_id=config.notifier.telegram.chat_id,
                )
                notifiers.append(n)

            elif backend == "qq":
                from autoykt.notifier.qq_bot import QQNotifier
                n = QQNotifier(
                    event_bus=event_bus,
                    onebot_url=config.notifier.qq.onebot_url,
                    target_qq=config.notifier.qq.target_qq,
                    access_token=config.notifier.qq.access_token,
                )
                notifiers.append(n)

            else:
                logger.warning(f"Unknown notifier backend: {backend}")

        except Exception as e:
            logger.error(f"Failed to create notifier '{backend}': {e}")

    logger.info(f"Created {len(notifiers)} notifier(s): {[n.name for n in notifiers]}")
    return notifiers