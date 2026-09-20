# AutoYKT

AutoYKT 是运行在 Windows 可见桌面上的课堂答题工具：识别新题或新题提示，截取题目，用本地 OCR 取得题干和选项，调用 Flash 多次采样，定位字母按钮，选择、提交并检查平台的完成反馈，然后等待下一题。

**首次使用需要按自己的正式页面标定一次；完成后，日常启动只需 `start-class.cmd`，默认运行两小时。** 程序使用本机截图和鼠标，不依赖 ChatGPT / Codex / Edge 连接插件，也不需要浏览器调试端口。

当前阶段以实际使用和问题修复为主。支持单选、多选及不同选项数量/位置；提供多页面配置、课程资料检索、私有运行报告和中断恢复。窗口必须保持可见且未被遮挡，不能在最小化窗口或隐藏标签页中操作。默认 Flash 为文本模式，依赖图片、公式或电路图细节的题目仍有识别与理解限制。

## 全新安装

准备 **Windows 10/11、64 位 Python 3.12、Git** 和可用的 DeepSeek API 密钥。支持范围是 Python 3.10–3.12，推荐 3.12；当前 [RapidOCR 依赖](https://pypi.org/project/rapidocr-onnxruntime/) 不支持在 Python 3.13/3.14 中正常全新安装。

在 PowerShell 中执行，不需要激活虚拟环境，也不需要更改执行策略：

```powershell
git clone https://github.com/tntexploding/AutoYKT.git
cd AutoYKT
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m autoykt init
```

`init` 会显示实际配置路径，默认是 `%LOCALAPPDATA%\AutoYKT\config.yaml`，并创建私有模板目录。已有文件不会被覆盖。后续请编辑这份私有文件；仓库中的 `config.example.yaml` 仅为模板，不能直接投入运行。

设置密钥有两种方式，任选一种：

- 简单方式：在私有配置的 `answering.providers[0]` 下添加 `api_key: "你的密钥"`。仅保存在仓库外的私有文件中，程序导出配置时会去除该字段。
- 环境变量方式：在 Windows“编辑账户的环境变量”中添加 `DEEPSEEK_API_KEY`，保存后重新打开终端/启动器。仅运行 `$env:DEEPSEEK_API_KEY = '...'` 只对当前 PowerShell 及其子进程有效，双击启动器不会继承。

`.env.example` 只列出可用变量；**程序不会自动读取 `.env` 文件**。默认模型为 `deepseek-v4-flash`，纯文本输入、关闭思考模式，先采样 3 次，必要时补到 5 次。至少两份有效回复、胜者至少两票；平票或不满足条件时不输入。模型自报置信度用于筛选回答，投票结果不等于正确率。

## 首次标定与启动

1. 在浏览器登录并打开正式课堂页面，固定浏览器缩放、窗口大小和侧栏布局。
2. 按 [首次部署与标定](docs/deployment.md) 配置窗口、题型、字母按钮、颜色、提交按钮、完成反馈和新题提示。新安装的示例页面默认禁用；完成后启用自己的页面配置。
3. 执行启动检查，再进行有人监控的首次运行：

```powershell
.\start-class.cmd --check
.\start-class.cmd
```

只有 `--check` 无缺项并返回成功，才能进入正式操作。该检查读取配置、模板和已登记的截图；不点击、不调用 API，也不能代替实际网页提交验收。若旧题已经显示“已完成”，程序会跳过它，等待新题。

启动后让课堂页完整可见，将终端放到其他显示器或最小化。程序不会自动切换浏览器标签页。需要失焦后继续监测时，配置 `background_monitoring: true` 和 [新题提示入口](docs/deployment.md#新题提示与其他状态)。

## 日常使用

在仓库目录运行，也可以双击 `start-class.cmd`。启动器使用仓库的 `.venv`，不会临时安装或升级依赖。

```powershell
.\start-class.cmd                         # 正式运行 120 分钟，会选择并提交
.\start-class.cmd --minutes 60            # 修改本次时长
.\start-class.cmd --profile example_yuketang
.\start-class.cmd --status                # 查看状态、暂停原因和报告路径
.\start-class.cmd --stop                  # 请求停止，再查状态确认 ended
```

`session` / 启动器明确启用本次真实输入，不改写私有 YAML 的演练开关。`--dry-run` 才仅调用模型与生成计划，不点击。更多计时、退出状态和恢复方法见 [课堂运行](docs/class-session.md)。

如果从不同终端、桌面应用启动时读取到不同配置，把实际私有配置的绝对路径设为 Windows 用户环境变量 `AUTOYKT_CONFIG`，或在命令中传 `--config '私有配置路径'`；以启动输出的 `Config:` 为准。

| 现象 | 处理 |
| --- | --- |
| 安装失败 / 找不到 OCR 依赖 | 用 `py -0p` 确认已安装 64 位 Python 3.12，使用它创建 `.venv` |
| `profile is disabled` / 模板缺失 | 按部署指南完成标定；配置不是模板文件；启用目标 profile |
| `window ... foreground/covered/size changed` | 恢复目标窗口、移开遮挡；尺寸或缩放改变后重新标定 |
| 提示密钥缺失 | 检查启动进程的环境变量或私有 `api_key`，不要把密钥写入仓库 |
| `previous input needs manual review` | 先停止运行，检查实际页面，再按下方恢复步骤处理 |
| 有心跳但没有题目 | 同时检查 `monitoring.detection_scans`、最近扫描时间和暂停原因；心跳不代表正在扫描 |

无法确认提交的题目会保留检查点，重启也不会盲目重复点击。人工确认当前页后，跳过它并恢复等待：

```powershell
.\start-class.cmd --stop
.\start-class.cmd --status
# 确认 ended，检查实际页面后执行；example_yuketang 替换为自己的页面 ID。
.\.venv\Scripts\python.exe -m autoykt recover --profile example_yuketang --skip-current
.\start-class.cmd
```

## 可选：课程知识库

支持 UTF-8 TXT/MD/RST/CSV、文本型 PDF、PPTX 和 DOCX。扫描 PDF、图片型课件不会自动提取图片内容。检索按 `course_id` 隔离，把相关课程片段与题目一起交给模型；没有直接复用历史答案的捷径。

```powershell
.\.venv\Scripts\python.exe -m autoykt knowledge ingest --course example_course 'D:\课程资料'
.\.venv\Scripts\python.exe -m autoykt knowledge search --course example_course '传感器的分类'
```

在私有配置设 `knowledge.enabled: true`，并使页面的 `course_id` 与导入课程一致。`knowledge record` 可在单独进程中 OCR 指定课件区域，参见 [配置手册](docs/configuration.md)。

## 更新、数据与验证

更新前停止程序，备份仓库外的整个私有配置目录（配置、模板、知识库和运行记录一起备份）：

```powershell
git pull --ff-only
.\.venv\Scripts\python.exe -m pip install -e .
.\start-class.cmd --check
```

不要覆盖自己的配置文件；新增字段和默认值查看 [变更记录](CHANGELOG.md)。从原型配置升级需先运行 `migrate --config '旧配置路径' --output '私有新配置路径'`，再补齐正式页面标定；迁移不会导出旧密钥。

配置、模板、题图、日志和 SQLite 默认都位于仓库外。默认只向配置的模型端点发送 OCR 文本与检索片段；改为视觉模型时才发送题图。QQ / Telegram 通知默认关闭，启用后会向配置的收件人发送题图、答案及结果。运行证据可能含课程内容，分享前先检查；截图和会话报告不自动清理，请定期归档。

项目按 [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) 的规范方向维护，使用 Pyink、Pylint 和 Pyright 检查。开发检查不需要真实密钥或桌面点击：

```powershell
.\.venv\Scripts\python.exe -m pip install -e '.[dev]'
.\.venv\Scripts\python.exe -m unittest discover -s tests -t .
.\.venv\Scripts\python.exe -m pyink --check src tests scripts main.py
.\.venv\Scripts\python.exe -m pylint src/autoykt scripts main.py
.\.venv\Scripts\python.exe -m pyright
```

自动化测试、真实题图回放和实际平台提交是不同证据。[本轮审计记录](docs/audit-2026-09-20.md) 记录验证范围；在完成真实新题的提交验收前，不宣称已通过真实连续两小时运行或保证每题正确。

| 文档 | 内容 |
| --- | --- |
| [首次部署与标定](docs/deployment.md) | 从空环境到首次课堂的操作顺序 |
| [课堂运行](docs/class-session.md) | 两小时启动、状态、停止、报告含义 |
| [配置手册](docs/configuration.md) | 全部配置字段及私有样本检查 |
| [正式页面操作](docs/formal-page-operations.md) | 状态采集、操作预览、翻页及恢复 |
| [离线回放](docs/preflight-rehearsal.md) | 已有题图测试及其边界 |
| [架构](docs/architecture.md) | 模块和答题流程 |

许可证：[MIT](LICENSE)。
