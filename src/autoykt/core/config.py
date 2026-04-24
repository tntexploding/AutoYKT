"""Configuration loader - reads YAML config and provides typed access."""

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class MonitorConfig(BaseModel):
    poll_interval: float = 0.5
    monitor_index: int = 1  # mss monitor index: 0=全部, 1=主显示器, 2=副显示器...
    # Legacy fields (kept for compatibility)
    roi: list[int] = Field(default=[100, 200, 800, 600])
    question_roi: list[int] | None = None
    # New explicit regions for auto-answer workflow
    feature_roi: list[int] | None = None
    entry_roi: list[int] | None = None
    task_roi: list[int] | None = None
    finish_task_roi: list[int] | None = None
    task_ready_timeout: float = 2.0
    task_ready_stable_frames: int = 2
    finish_click_delay: float = 0.5
    post_answer_resume_by_change_enabled: bool = True
    post_answer_resume_change_ratio: float = 0.18
    post_answer_resume_change_hits: int = 2
    match_threshold: float = 0.85
    template_path: str = "assets/templates/question_region.png"
    debounce_frames: int = 3

    @model_validator(mode="after")
    def _fill_region_fallbacks(self) -> "MonitorConfig":
        if self.feature_roi is None:
            self.feature_roi = self.roi
        if self.task_roi is None:
            self.task_roi = self.question_roi or self.roi
        if self.entry_roi is None:
            self.entry_roi = self.task_roi
        return self


class DetectorConfig(BaseModel):
    question_feature_template_path: str = "assets/templates/question_region.png"
    option_templates: dict[str, str] = Field(default_factory=lambda: {
        "A": "assets/templates/options/A.png",
        "B": "assets/templates/options/B.png",
        "C": "assets/templates/options/C.png",
        "D": "assets/templates/options/D.png",
    })
    option_match_threshold: float = 0.85
    option_match_method: str = "TM_CCOEFF_NORMED"
    option_nms_iou: float = 0.3


class OcrConfig(BaseModel):
    engine: str = "rapidocr"
    vision_llm_model: str = "gpt-4o"


class AgentConfig(BaseModel):
    provider: str = "openai"
    model: str = "gpt-4o"
    models: list[str] = Field(default_factory=list)
    answer_count: int = 1
    min_response_count: int = 1
    auto_click: bool = False
    question_db_enabled: bool = True
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    prompt_template: str = "src/autoykt/agent/prompts/answer.j2"
    db_similarity_threshold: float = 0.9
    timeout: int = 10

    @field_validator("answer_count")
    @classmethod
    def _validate_answer_count(cls, value: int) -> int:
        if value < 1:
            raise ValueError("agent.answer_count must be greater than 0")
        return value

    @field_validator("min_response_count")
    @classmethod
    def _validate_min_response_count(cls, value: int) -> int:
        if value < 1:
            raise ValueError("agent.min_response_count must be greater than 0")
        return value


class ClickerConfig(BaseModel):
    options_positions: dict[str, list[int]] = Field(default_factory=dict)
    confirm_delay: float = 1.0


class TelegramConfig(BaseModel):
    token: str = ""
    chat_id: str = ""


class QQConfig(BaseModel):
    onebot_url: str = "http://localhost:3001"
    target_qq: int = 0
    access_token: str = ""

    @field_validator("target_qq", mode="before")
    @classmethod
    def _parse_target_qq(cls, value: Any) -> int:
        if isinstance(value, int):
            return value

        if isinstance(value, str):
            raw_value = value.strip()
            if raw_value.startswith("${") and raw_value.endswith("}"):
                env_key = raw_value[2:-1]
                raw_value = os.environ.get(env_key, "").strip()

            if raw_value.isdigit():
                return int(raw_value)

        raise ValueError("notifier.qq.target_qq must be a QQ number or a resolvable ${ENV_VAR} placeholder")


class NotifierConfig(BaseModel):
    enabled: list[str] = Field(default=["telegram"])
    telegram: TelegramConfig = TelegramConfig()
    qq: QQConfig = QQConfig()


class LoggingConfig(BaseModel):
    level: str = "INFO"
    log_dir: str = "storage/logs"


class AppConfig(BaseModel):                                                     #标准config类
    monitor: MonitorConfig = MonitorConfig()
    detector: DetectorConfig = DetectorConfig()
    ocr: OcrConfig = OcrConfig()
    agent: AgentConfig = AgentConfig()
    clicker: ClickerConfig = ClickerConfig()
    notifier: NotifierConfig = NotifierConfig()
    logging: LoggingConfig = LoggingConfig()

    @model_validator(mode="after")
    def _sync_legacy_template_path(self) -> "AppConfig":
        # If user only configured legacy monitor.template_path, keep detector path aligned.
        if self.monitor.template_path and (
            not self.detector.question_feature_template_path
            or self.detector.question_feature_template_path == "assets/templates/question_region.png"
        ):
            self.detector.question_feature_template_path = self.monitor.template_path
        return self


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