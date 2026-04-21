"""Auto-Answer System - Entry Point.

Usage:
    python main.py              # Run the full system
    python main.py --calibrate  # Run calibration tool
"""

import asyncio
import argparse
import logging
import signal
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from autoykt.core.config import load_config
from autoykt.core.event_bus import EventBus, EventType, Event
from autoykt.core.logger import setup_logger
from autoykt.monitor.screen_watcher import ScreenWatcher
from autoykt.agent.answer_agent import AnswerAgent
from autoykt.notifier.factory import create_notifiers


async def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-Answer System")
    parser.add_argument("--calibrate", action="store_true", help="Run calibration tool")
    parser.add_argument("--detect-only", action="store_true", help="Only detect + notify, skip OCR/Agent/Clicker")
    parser.add_argument("--test-qq", action="store_true", help="Send a test message to QQ and exit")
    parser.add_argument("--monitors", action="store_true", help="Show monitor layout info and exit")
    parser.add_argument("--config", default="config.example.yaml", help="Config file path")
    args = parser.parse_args()
    # 检测传入参数，选择普通模式/校准模式，或更新config文件

    if args.monitors:
        import mss
        sct = mss.mss()
        print("=== 显示器布局 ===")
        for i, m in enumerate(sct.monitors):
            label = "虚拟桌面(全部)" if i == 0 else f"显示器 {i}"
            print(f"  [{i}] {label}: left={m['left']}, top={m['top']}, {m['width']}x{m['height']}")
        sct.close()
        return

    if args.calibrate:
        config = load_config(args.config)
        from scripts.calibrate import Calibrator
        Calibrator(monitor_index=config.monitor.monitor_index).run()
        return

    if args.test_qq:
        config = load_config(args.config)
        setup_logger(level=config.logging.level, log_dir=config.logging.log_dir)
        from autoykt.notifier.qq_bot import QQNotifier
        notifier = QQNotifier(
            event_bus=EventBus(),
            onebot_url=config.notifier.qq.onebot_url,
            target_qq=config.notifier.qq.target_qq,
            access_token=config.notifier.qq.access_token,
        )
        await notifier.send_text("✅ AutoYKT 测试消息 - QQ通道连接正常")
        await notifier.close()
        print("测试消息已发送，请检查QQ。")
        return

    # Load config and set up logging
    config = load_config(args.config)
    logger = setup_logger(level=config.logging.level, log_dir=config.logging.log_dir)
    logger.info("Auto-Answer System starting...")
    # 读取config文件，初始化logger并发布首条INFO

    # Create event bus
    bus = EventBus()
    #新的eventbus对象

    # Temporary logging subscriber for all events (until other modules are wired)
    async def log_event(event: Event) -> None:
        logger.info(f"[EventBus] {event.type.name}: {event.payload}")

    for et in EventType:
        bus.subscribe(et, log_event)
    #每个eventtype里的事件类型都挂logevent

    # Create modules
    detect_only = args.detect_only
    watcher = ScreenWatcher(config=config, event_bus=bus, detect_only=detect_only)
    notifiers = create_notifiers(config=config, event_bus=bus)

    agent = None
    if not detect_only:
        agent = AnswerAgent(config=config, event_bus=bus)

    # Graceful shutdown
    async def cleanup() -> None:
        """Async cleanup for resources that need await."""
        for n in notifiers:
            try:
                await n.close()
            except Exception as e:
                logger.warning(f"Error closing notifier {n.name}: {e}")

    def shutdown() -> None:
        logger.info("Shutting down...")
        watcher.stop()
        if agent is not None:
            agent.close()
        asyncio.ensure_future(cleanup())
        bus.stop()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, shutdown)
        except NotImplementedError:
            pass  # Windows doesn't support add_signal_handler
    #ctrlC终止程序

    # Run event bus and watcher concurrently
    await asyncio.gather(
        bus.start(),
        watcher.start(),
    )


if __name__ == "__main__":
    asyncio.run(main())