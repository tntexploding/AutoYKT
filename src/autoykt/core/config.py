"""Configuration loader - reads YAML config and provides typed access."""

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class MonitorConfig(BaseModel):
    poll_interval: float = 0.5
    monitor_index: int = 1  # mss monitor index: 0=全部, 1=主显示器, 2=副显示器...
    roi: list[int] = Field(default=[100, 200, 800, 600])
    match_threshold: float = 0.85
    template_path: str = "assets/templates/question_region.png"
    debounce_frames: int = 3


class OcrConfig(BaseModel):
    engine: str = "rapidocr"
    vision_llm_model: str = "gpt-4o"


class AgentConfig(BaseModel):
    provider: str = "openai"
    model: str = "gpt-4o"
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    prompt_template: str = "src/autoykt/agent/prompts/answer.j2"
    db_similarity_threshold: float = 0.9
    timeout: int = 10


class ClickerConfig(BaseModel):
    options_positions: dict[str, list[int]] = Field(default_factory=dict)
    confirm_delay: float = 1.0


class TelegramConfig(BaseModel):
    token: str = ""
    chat_id: str = ""


class QQConfig(BaseModel):
    onebot_url: str = "http://localhost:3001"
    target_qq: str = ""
    access_token: str = ""


class NotifierConfig(BaseModel):
    enabled: list[str] = Field(default=["telegram"])
    telegram: TelegramConfig = TelegramConfig()
    qq: QQConfig = QQConfig()


class LoggingConfig(BaseModel):
    level: str = "INFO"
    log_dir: str = "storage/logs"


class AppConfig(BaseModel):                                                     #标准config类
    monitor: MonitorConfig = MonitorConfig()
    ocr: OcrConfig = OcrConfig()
    agent: AgentConfig = AgentConfig()
    clicker: ClickerConfig = ClickerConfig()
    notifier: NotifierConfig = NotifierConfig()
    logging: LoggingConfig = LoggingConfig()


def _resolve_env_vars(data: Any) -> Any:                                        #递归解算传入的data
    """Recursively resolve ${ENV_VAR} placeholders in config values."""
    if isinstance(data, str) and data.startswith("${") and data.endswith("}"):
        env_key = data[2:-1]
        return os.environ.get(env_key, data)
    if isinstance(data, dict):
        return {k: _resolve_env_vars(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_resolve_env_vars(item) for item in data]
    return data                                                                 #data中的信息被全部提取并返回


def load_config(path: str = "config.example.yaml") -> AppConfig:                #返回appconfig
    """Load and validate config from YAML file."""
    config_path = Path(path)
    if not config_path.exists():                                                #检查路径合法性
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(config_path, "r", encoding="utf-8") as f:                         #config传给raw，附加安全控制
        raw = yaml.safe_load(f)

    resolved = _resolve_env_vars(raw)                                           #raw提取为resolved
    return AppConfig(**resolved)                                                #自动格式化到新appconfig对象