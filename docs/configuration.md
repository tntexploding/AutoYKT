# 配置手册

AutoYKT v2 使用严格 YAML。未知字段、错误类型、非法区域、重复 profile/model 或不可能的共识门槛会在启动前报错。完整安全示例位于仓库根目录 `config.example.yaml`。

## 配置发现顺序

未传 `--config` 时依次查找：

1. `AUTOYKT_CONFIG` 指向的文件；
2. 平台私有配置（Windows 为 `%LOCALAPPDATA%\AutoYKT\config.yaml`）；

`AUTOYKT_CONFIG_HOME` 可以改变平台私有配置目录。公共 `config.example.yaml` 永远不会被当作默认运行配置。

当前目录的 `config.yaml` 不再自动加载；迁移旧配置时必须显式传入 `--config`。

## 顶层结构

| 区域 | 说明 |
|---|---|
| `version` | 固定为 `2` |
| `runtime` | 活动页面、轮询间隔与总干运行开关 |
| `storage` | 数据、日志、截图目录 |
| `answering` | 模型端点、并发与共识 |
| `knowledge` | 本地课程知识参数 |
| `pages` | 一个或多个页面视觉适配器 |
| `notifier` | 远程通知及环境变量名称 |
| `logging` | 日志级别 |

## runtime

| 字段 | 默认 | 说明 |
|---|---:|---|
| `active_profiles` | `[]` | 空表示所有 `enabled: true` 页面；否则只选择列出的 ID |
| `poll_interval_seconds` | `0.5` | 完成一轮 profile 轮询后的等待时间，必须大于 0 |
| `dry_run` | `true` | 为 true 时禁止鼠标交互 |

## storage

`data_dir`、`log_dir`、`screenshot_dir` 均可为绝对路径或相对私有配置目录的路径。知识库使用自己的 `database_path`；`data_dir` 保存 `runs` 操作证据、`sessions` 会话报告和离线回放。

## answering

### provider

| 字段 | 说明 |
|---|---|
| `name` | 本地唯一名称，用于日志与投票明细 |
| `base_url` | OpenAI 兼容 API 根地址 |
| `api_key_env` | 保存密钥的环境变量名 |
| `models` | 至少一个模型名 |
| `timeout_seconds` | 单请求总超时 |
| `max_output_tokens` | 回答长度上限 |
| `request_json_object` | 端点支持 JSON response format 时才开启 |
| `input_mode` | `text` 只发送 OCR 文字；`vision` 还发送题图 |
| `thinking` / `reasoning_effort` | 可选提供方参数；默认 Flash 关闭思考以控制时延 |

`api_key_env` 从启动进程的环境读取密钥。也支持在仓库外的私有文件中设置 `api_key`；存在时优先使用它，去除首尾空白并拒绝空值，配置导出会去除该字段。标定保存会保留原私有文件中的密钥。程序不自动加载 `.env`。

### 共识字段

| 字段 | 说明 |
|---|---|
| `minimum_responses` | 至少需要多少条成功解析的模型回答 |
| `minimum_agreement` | 获胜选项至少需要多少票，且不能平票 |
| `minimum_confidence` | 模型自报置信度下限；不是多数票比例，也不是正确率 |
| `maximum_parallel_requests` | 同时进行的最大模型请求数 |
| `initial_rounds` | 每个模型初次并发安排的请求轮数，默认 1；Flash 模板为 3 |
| `maximum_rounds` | 每个模型最多采样轮数，默认 1；Flash 模板为 5，所有轮数共用截止时间 |
| `auto_apply` | 是否允许共识结果进入鼠标阶段 |
| `prompt_template` | 可选 Jinja2 文件；空值使用内置严格 JSON 提示词 |

自定义提示词可使用 `allowed_options`、`question_text`、`knowledge_context`、`multiple`，并兼容旧变量 `question` 与 `options`。

## pages

### 基本字段

| 字段 | 说明 |
|---|---|
| `id` | 只允许字母、数字、点、下划线和连字符；全局唯一 |
| `display_name` | 日志与通知中显示的名称 |
| `course_id` | 知识库隔离键 |
| `enabled` | 是否可被调度 |
| `monitor_index` | `autoykt monitors` 显示的 MSS 编号 |

### regions

每个区域格式都是 `[x, y, width, height]`，无窗口绑定时相对指定显示器左上角；绑定 `target_window` 后相对窗口客户区左上角。宽高必须大于零。

| 字段 | 用途 |
|---|---|
| `detection` | 搜索触发模板，范围越小越快 |
| `question` | 稳定检查、截图、OCR 与模型输入 |
| `answers` | 搜索 A/B/C/D 等选项模板；省略时使用 question |
| `verification` | 点击/提交后的状态证明；省略时使用 question |
| `rearm` | 判断当前题是否已经离开；省略时使用 question |

### triggers

每个触发器包含唯一 `name`、模板 `path`、0–1 的 `threshold`、至少 1 的 `consecutive_hits` 和 `action`；`question_type` 为 `single` 或 `multiple`，默认 `single`。

- `answer`：进入完整答题流程。
- `notify`：保存页面截图并通知，不调用模型或鼠标。

模板每次连续出现只触发一次，消失后才会自然重新武装；完成题目后状态机还有第二层 `rearm` 保护。

### answer_style

`option_templates` 将选项键映射到图片模板；匹配成功后点击模板中心。`fallback_positions` 将选项键映射到点坐标。配置结构允许两者至少配置一种，键会统一转为大写；正式启动要求模板、按钮颜色齐全且 `fallback_positions: {}`，不能用固定坐标代替动态定位。

当单选圆形按钮与多选方形按钮使用相同字母时，可以只截取中央字母作为选项模板，并在 `button_colors.button_size` 中配置完整按钮的 `[宽, 高]`。程序以实际颜色轮廓确定完整按钮范围，用于选中反馈遮罩和未识别按钮检查；省略该配置时沿用模板尺寸。单选与多选仍需各自的题型触发模板。

`fallback_coordinate_space`：

- `monitor`：相对 `monitor_index` 的左上角，适用于未绑定窗口的兼容配置。
- `window`：相对窗口客户区左上角；绑定窗口时标定器输出此格式。
- `screen`：虚拟桌面绝对坐标，副屏位于左侧时可以为负值。

### entry_target 与 submit_target

入口可为空；正式页面需要提交目标，只有明确确认点击即提交的单选页可改用 `submit_on_select: true`。点击目标可为点或区域：

```yaml
submit_target:
  region: [1500, 900, 120, 50]
  coordinate_space: window  # 已绑定 target_window 的正式页面
```

`point` 与 `region` 必须二选一；区域会点击中心。

### question_ready

- `delay_seconds`：进入题目后的固定等待。
- `timeout_seconds`：等待题面稳定的最长时间。
- `stable_frames`：连续稳定帧数。
- `maximum_change_ratio`：低于此像素变化比例视为稳定；模型返回后、选项点击前和提交前也用同一上限确认题目仍未改变。请将区域和阈值标定到能够容忍正常选中反馈，同时识别换题。

### verification

`success_templates` 是成功状态模板列表，每项有 `path` 与 `threshold`。配置后只接受模板证据；列表为空时，以相对提交前基线的 `minimum_change_ratio` 作为兜底。

`stable_hits` 要求证据连续出现；`timeout_seconds` 和 `poll_interval_seconds` 控制验证窗口。开始点击前已有成功标记时，记录为已完成并等待下一题；已有失败标记（例如已过期）时，记录为不可作答，跳过模型调用与点击；选项点击后已出现成功标记时，系统直接验证，不会额外点击提交按钮。正式启动必须使用明确的成功模板；纯画面变化兜底仅保留给旧版内部兼容流程。

### rearm

系统在题目截图时记录 `rearm` 区域，保留到本题完成或答案被拒绝。因此模型等待期间或提交后立即出现的新题不会被当成已处理题目的基线。该区域相对基线达到 `minimum_change_ratio`，或本次触发的模板不再存在，并连续满足 `stable_hits` 后，才接受下一题；成功状态模板仍存在时继续等待。尚未输入时失败会保留原题基线；选择或提交已尝试且无法确认结果时暂停并要求人工处理。已验证的结果页动作由 `page_flow` 管理。普通 `rearm` 超时只记一次信息日志，继续等待，不把老师尚未换题记成运行错误。

`rearm` 应圈定题号或稳定题干等能区分题目的区域，避开倒计时、选中高亮、动画与提交反馈。`minimum_change_ratio` 必须大于零。纯画面变化不能从业务上证明提交成功；需要可靠确认时应配置独立成功模板。

## 页面连续运行配置

`pages[].page_flow.after_submit` 配置关闭结果、继续或下一题。每项包含唯一 `name`、带 `region` 的状态模板 `when`、点击目标 `target`、连续命中次数 `stable_hits`。有这些动作的真实运行必须使用明确成功模板；每个动作每题最多一次，且输入前重验状态。所有新增区域和点击目标同样遵循窗口坐标及边界检查。

`page_flow.loading`、`manual`、`finished` 均为 `{path, region, threshold}` 列表，分别表示临时等待、人工处理和课堂结束。`transition_timeout_seconds` 限制加载与单次结果页动作后的等待。

`question_budget.total_seconds` 限制触发至提交验证的总时间（默认 45 秒），`context_seconds` 限制 OCR 与知识检索（默认 5 秒），`answer_seconds` 限制包含排队、重试的模型阶段（默认 25 秒）。模型阶段还会预留提交验证时间；到期只使用及时返回且满足原共识门槛的响应。

`recovery.maximum_attempts` 为临时模型错误的总尝试次数（默认 2，范围 1–5），`delay_seconds` 为重试间隔（默认 1 秒）。重试前重新核对原题，不重放入口或提交动作。

私有 `storage.data_dir/runs/<profile_id>/pending.json` 记录未离开的题目。已验证记录可恢复等待；未确认输入阻止真实运行，人工检查后使用 `recover --skip-current` 记录当前页跳过基线。详情和独立按钮预览命令见 [正式页面操作与验收](formal-page-operations.md)。

## knowledge

| 字段 | 说明 |
|---|---|
| `enabled` | 是否在答题时查询课程知识库 |
| `database_path` | 私有 SQLite 文件 |
| `question_ocr_enabled` | 是否本地 OCR 题目；即使知识库关闭，文本也可进入模型提示 |
| `maximum_results` | 最多返回资料块数 |
| `minimum_score` | 本地相似度下限 |
| `maximum_context_characters` | 发送给模型的资料字符上限 |
| `chunk_size_characters` | 入库分块大小 |
| `chunk_overlap_characters` | 相邻分块重叠，必须小于块大小 |

## notifier

`enabled` 可包含 `qq`、`telegram`。配置只保存变量名：

- QQ 必需 `target_env`；`access_token_env` 对无鉴权 OneBot 服务可以为空值。
- Telegram 必需 `token_env` 和 `chat_id_env`。
- `onebot_url` 可指向本机 OneBot HTTP 服务。

通知属于观察层；发送失败会记录日志，但不会让状态机点击、重试或改变答案。

## logging

`level` 只能是 `DEBUG`、`INFO`、`WARNING`、`ERROR`、`CRITICAL`。控制台使用可读文本，文件使用每行一个 JSON 对象的 UTC 时间日志。

## 阈值调试顺序

1. 固定浏览器缩放、系统 DPI、窗口位置和主题。
2. 缩小 detection 区域，模板保留足够独特的边缘与文字。
3. 从较高阈值（如 0.90）开始，在 `run --detect-only` 中观察。
4. 若漏检，小幅降低；若误检，提高阈值或重新裁剪模板。
5. 提高 `consecutive_hits` 可抑制瞬时误检，但会增加触发延迟。
6. 最后验证答案坐标、成功模板和下一题重置，再开启真实交互。

## 旧配置迁移

旧版 `monitor/detector/agent/clicker/notifier` 配置仍可直接读取，但会发出兼容警告。建议导出后人工复核：

```powershell
autoykt migrate --config .\config.yaml --output "$env:LOCALAPPDATA\AutoYKT\config.v2.yaml"
```

迁移不会修改源文件，导出不会包含旧版内嵌 API Key；缺失的密钥来源会替换为 `OPENAI_API_KEY` 或按提供方名称生成的环境变量名。旧版通知中的直接账号/Token 也不会复制为明文；请改为相应环境变量。

## 配置保存与检查

- 校准截图写入 `templates/<profile_id>/`，不同页面不会覆盖彼此的模板；部分后备坐标修改保留其他选项。
- 配置迁移到其他目录时，模板、提示词、数据库和运行目录改写为原目标的绝对路径。迁移不会搬动文件；如需迁移数据目录，应单独复制文件并调整目标配置。
- `check --profile <id>` 只检查指定页面的资源，即使该页面还未启用；不会启用或改写它。`--live` 额外要求该页已启用且满足正式操作条件。
- `check` 会解码图片并拒绝没有空间纹理的纯色模板，避免归一化匹配产生无意义的满分。中文目录和文件名使用 Unicode 安全的读写方式。
- 无穷大、NaN、零变化门槛会被拒绝；旧版 `auto_click` 必须为 YAML 布尔值。
- `run --detect-only` 不加载答案/成功模板、自定义模型提示词或知识数据库，可先单独完成触发检测标定。


## 正式页面配置扩展

- `pages[].target_window`：可选 Windows 窗口绑定，包含标题正则 `title_pattern`、可选窗口类 `class_name`、标定客户区大小 `client_size: [width, height]`。
- 绑定后，所有截图区域改为客户区相对坐标；入口、提交和备用点必须显式使用 `coordinate_space: window`。
- `pages[].page_guard`：页面标识，包含 `path`、`region`、`threshold`。每次真实输入前再次检查。
- `verification.failure_templates`：明确失败状态；命中后停止当前尝试。
- `verification.require_success_template`：显示器配置也可强制明确成功模板。绑定窗口的真实操作始终要求成功模板和页面标识。
- 新增运行报告保存在 `storage.data_dir/runs/<profile_id>/`。`--once`、`--profile`、`--dry-run` 为单次命令参数，不写回个人配置。

详细配置、采集和标定命令见 [正式页面操作与验收](formal-page-operations.md)。


## 正式启动与离线回放

`check --live --profile <id>` 检查窗口绑定、页面身份、动态选项模板和颜色、提交行为、成功反馈及文本模型的 OCR 前提。`run` 在允许真实输入时也执行该检查。`submit_on_select: true` 仅用于已经确认“点击即提交”的单选页面，不能和 `submit_target` 同时设置；多选仍要求明确提交目标。

`rehearse --profile <id> --image <path> [--image <path> ...] [--output <新目录>]` 不要求启用 profile，也不修改源配置。它用文件截图和虚拟鼠标运行正式状态机，会调用已配置的模型。虚拟提交控件与成功反馈不能代替真实网站的提交验收。详见 [课前回放与正式测试](preflight-rehearsal.md)。


## Flash 的限时请求

默认模板使用 `deepseek-v4-flash`、`input_mode: text`，本地 OCR 提供题干与完整选项。
`thinking: disabled` 通过兼容接口的 `extra_body` 显式关闭思考模式，适用于约一分钟的课堂答题；
`reasoning_effort` 在该模式下不会发送。可以在私有配置中改为 `thinking: enabled` 和
`reasoning_effort: low`，但应重新实测时延与准确率。`thinking: null` 不发送此扩展参数，
用于未配置该能力的其他兼容服务。参见 [DeepSeek 官方模式说明](https://api-docs.deepseek.com/guides/thinking_mode/)。

采样上限、至少两份有效回复和整题时间预算独立于思考模式，不会为赶时间降低投票门槛。


### 后台窗口和新题入口

`target_window.background_monitoring` 默认关闭；开启后只读监测可见的后台窗口。`focus_point` 是可选的窗口客户区空白点，用于系统拒绝直接激活时的普通点击；必须在私有配置标定，并位于客户区内。被遮挡、最小化或尺寸变化仍阻止操作。

`page_flow.before_question` 是新题提示入口列表，支持 `name`、`when`、`stable_hits`，以及二选一的 `target` 或 `click_match: true`。后者根据匹配图案动态定位中心，模板不得含关闭按钮。`when` 可带 `scales`（默认 `[1.0]`）和 `match_grayscale`（默认 `false`）。提示、模型、选项及提交验证共享整题时间预算。


## 私有截图回归检查

在每个页面的 `calibration_samples` 中登记原尺寸截图及预期结果。截图存入私有配置目录；路径相对此目录解析。`check`、`session --check` 和正式启动都会核验，样本不匹配时阻止输入。没有配置样本时只检查配置和模板文件，不代表识别已经通过验证。

```yaml
calibration_samples:
  - path: calibration/single-active.png
    expected_state: question
    question_type: single
    expected_options: [A, B, C, D]
    selected_options: []
  - path: calibration/multiple-completed.png
    expected_state: completed
    question_type: multiple
    expected_options: [A, B, C, D, E]
    selected_options: [A, C]
  - path: calibration/ordinary-slide.png
    expected_state: unrecognized
    expected_options: []
```

`expected_state` 支持 `question`、`completed`、`failure`（过期/失败）、`unrecognized`、`notice`、`finished`、`loading`、`manual`。题型、选项和选中集合可按样本情况省略；显式空列表要求不出现任何选项/选择。选项要求集合完全一致，并且字母定位唯一、颜色状态已知。绑定窗口时图片尺寸必须等于标定客户区尺寸，页面身份模板也必须匹配。移动配置时 `migrate` 会保留样本的实际路径。

建议至少覆盖单选、多选、不同选项数量、已完成、已过期、普通课件和下课画面，并加入真实新题提示的正例与无提示的反例。样本仅证明这些像素下的识别结果；虚拟回放只能验证本地状态机，仍需真实新题完成“选中、提交、平台确认、下一题”的验收。旧截图中的蓝色选择不是标准答案。


换题检测使用 `rearm.minimum_change_ratio`，默认 0.02。白底题面即使内容全换，变化像素也可能少于 10%；请用连续两题的原尺寸截图回放确认阈值。配置按钮颜色后，比较会排除已识别按钮的完整区域，避免灰蓝切换被当成换题；完成或过期反馈仍可阻止重复答题。阈值不宜靠任意降低来追求通过，应同时检查同一道题选中前后的反例。


## 实时课程录入

`knowledge ingest` 支持 UTF-8 TXT/MD/RST/CSV、文本型 PDF、PPTX 和 DOCX；图片内容不会自动 OCR。
`knowledge record` 在单独进程中持续采集指定显示器区域，画面变化且 OCR 有足够文字后写入该课程；Ctrl+C 停止：

```powershell
autoykt monitors
autoykt knowledge record --course example_course --monitor 1 --region '100,100,800,600' --interval 2
```

上述坐标是示例，必须替换为课件区域。录入不绑定浏览器窗口，不要让其他窗口遮挡采集区域。答题中的检索按 `course_id` 限定资料，将匹配片段送给模型，不以历史选中项当作标准答案。
