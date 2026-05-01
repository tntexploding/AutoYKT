# AutoYKT

基于屏幕特征检测 + 视觉大模型答题 + 自动点击 + 远程通知的自动答题脚本。

本项目通过持续截图指定显示器区域，检测题目框出现后将截图提交给兼容 OpenAI 格式的视觉模型，解析出选项答案并自动点击，同时将关键事件通过 QQ/Telegram 推送到远端。

## 功能特性

- 题目出现检测：OpenCV 模板匹配 + 防抖触发（debounce）
- 视觉答题：支持图片输入的 OpenAI 兼容接口
- 自动作答：解析 A/B/C/D 并模拟鼠标点击
- 结果回传：QQ OneBot / Telegram Bot(暂未适配) 通知
- 本地缓存：SQLite 存储历史问答(建议关闭，没用)
- 运行日志：JSONL 结构化日志

## 工作流程

1. `ScreenWatcher` 定时截图 ROI 区域。
2. `QuestionDetector` 识别题目框后发布 `QUESTION_DETECTED` 事件。
3. `AnswerAgent` 调用视觉 LLM，解析答案并发布 `ANSWER_READY`。
4. `Clicker` 根据选项坐标执行点击，发布 `CLICK_DONE`。
5. `Notifier` 监听事件并推送题目/答案/结果截图。

核心事件流：`QUESTION_DETECTED -> ANSWER_READY -> CLICK_DONE`

## 目录结构

```text
AutoYKT/
  src/autoykt/
    agent/                 # LLM 答题与题库缓存
    core/                  # 配置、日志、事件总线
    monitor/               # 截图、检测、点击、OCR
    notifier/              # QQ / Telegram 通知
  scripts/                 # 校准与测试脚本
  assets/templates/        # 题目区域模板图
  storage/
    data/                  # SQLite 数据
    logs/                  # JSONL 日志
    screenshots/           # 题目/结果截图
  config.example.yaml      # 配置模板
  main.py                  # 启动入口
  requirements.txt
  commands.txt
```

## 环境要求

- Windows（已针对 DPI 与多显示器场景做适配）
- Python 3.10+
- 可访问的 OpenAI 兼容视觉模型接口
- QQ 通知需本地 OneBot HTTP 服务可用（如 Lagrange + OneBot）

## 安装步骤

```bash
pip install -r requirements.txt
```

## 配置说明

项目使用 YAML 配置，支持 `${ENV_VAR}` 环境变量占位。

1. 复制配置模板：

```bash
copy config.example.yaml config.yaml
```

1. 设置环境变量（PowerShell 示例）：

```powershell
$env:OPENAI_API_KEY="<your_api_key>"
$env:ONEBOT_ACCESS_TOKEN="<onebot_token>"
$env:QQ_TARGET="<qq_number>"
```

1. 根据你的屏幕和题目页面完成坐标标定（见下一节）。

## 首次标定

```bash
python main.py --calibrate
```

标定时注意显示器的缩放比例。

按键说明：

- `1`：标定题目 ROI（点击左上和右下）
- `2`：标定 A/B/C/D 点击位置
- `3`：截取题目模板图片
- `4`：标定完整题目截图区域（用于提交给 AI）
- `5`：点击进入题目的位置
- `6`：依次标定ABCD的标准样式
- `7`：标定提交答案的位置
- `s`：保存到配置
- `q`：退出

模板会保存到 `assets/templates/question_region.png`。
标定结果会优先写入 `config.yaml`，如果不存在则写入 `config.example.yaml`。

## 运行方式

初次上手：首次标定--测试QQ连接--测试特征检测--正式运行

```bash
python main.py --calibrate --config config.yaml   #标定
python main.py --test-qq --config config.yaml     #测试QQ
python main.py --detect-only --config config.yaml #测试特征
python main.py --config config.yaml               #运行
```

### 1. 查看显示器索引

```bash
python main.py --monitors
```

### 2. 仅检测模式（不调用 AI，不点击）

```bash
python main.py --detect-only --config config.yaml
```

### 3. QQ 通知连通性测试

```bash
python main.py --test-qq --config config.yaml
```

### 4. 正式运行

```bash
python main.py --config config.yaml
```

## 测试脚本

```bash
python -m scripts.test_monitor --all
```

可选参数：

- `--capture` 仅测截图
- `--detect` 仅测检测
- `--click A` 测试点击流程提示

## 关键配置项

- `monitor.roi`：截图区域 `[x, y, width, height]`
- `monitor.template_path`：题目模板路径
- `monitor.match_threshold`：模板匹配阈值（0~1）
- `monitor.post_answer_resume_by_change_enabled`：是否启用“答题后基于 ROI 像素变化恢复检测”
- `monitor.post_answer_resume_change_ratio`：判定恢复所需的像素变化比例阈值
- `monitor.post_answer_resume_change_hits`：连续满足阈值的次数要求
- `agent.model`：视觉模型名
- `agent.models`：并行提交的模型列表
- `agent.answer_count`：本次并发回答的模型数量，默认使用列表前 N 个模型
- `agent.min_response_count`：超时后继续流程所需的最少模型回答条数
- `agent.base_url`：OpenAI 兼容 API 地址
- `agent.auto_click`：是否在拿到回答后继续自动点击；关闭时只发送原始回答到 QQ/TG
- `agent.question_db_enabled`：是否启用过往问答数据库缓存
- `clicker.options_positions`：A/B/C/D 屏幕坐标
- `notifier.enabled`：通知后端列表（`qq` / `telegram`）
- `notifier.qq.target_qq`：QQ 号，支持直接填写数字或使用 `${QQ_TARGET}` 环境变量占位符

## 日志与产物

- 结构化日志：`storage/logs/YYYY-MM-DD.jsonl`
- 题目/结果截图：`storage/screenshots/`
- 问答缓存：`storage/data/questions.db`

## 常见问题

1. 检测不到题目/标定界面显示问题

- 重新标定模板，保证模板来自当前页面样式。
- 降低 `monitor.match_threshold`（例如 0.85 -> 0.80）。
- 确认 `monitor_index` 与 `roi` 对应正确显示器。
- 注意windows屏幕设置中的缩放比例，缩放会影响标定界面的正常显示。

1. 识别答案慢或失败

- 检查 `agent.base_url` 与模型是否支持图片输入。
- 查看日志中 `Vision LLM call failed` 相关错误。

1. 点击位置错误

- 重新运行 `scripts.calibrate` 标定 A/B/C/D 坐标。
- 确认系统缩放设置与 DPI 感知一致。

1. QQ 无消息

- 检查 OneBot 服务地址和 token。
- 先执行 `--test-qq` 单独验证通道。

1. 识别系统多次识别同一题目

- 在标定“3，题目区域”时，在题目PPT未被选中的情况下标定，要包含一小部分PPT灰色边框。
- 原理：识别-答题流程会自动点击PPT页面，此时灰色边框变为蓝色，题目特征和标定特征显著不同，避免重复识别。

## 开发说明

- 事件总线定义在 `src/autoykt/core/event_bus.py`。
- 业务入口在 `main.py`。
- 新增通知后端时，继承 `BaseNotifier` 并在 `notifier/factory.py` 注册。

## 安全建议

- 不要将真实 API Key、QQ 号、Token 写入版本库。
- 使用环境变量注入敏感信息。
- 提交前检查 `config.yaml` 与日志截图是否包含隐私数据。

## 开源协议

本项目采用 [MIT License](LICENSE) 开源协议。
