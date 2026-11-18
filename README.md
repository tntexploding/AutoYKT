# AutoYKT

AutoYKT 是雨课堂 Web 的课堂提醒和辅助作答工具。它使用普通 Web 会话发现课堂，在老师发题时记录题目，也可以用 AI 生成答案建议。

[官方仓库 · tntexploding/AutoYKT](https://github.com/tntexploding/AutoYKT) · [下载源码 ZIP](https://github.com/tntexploding/AutoYKT/archive/refs/heads/main.zip)

## 提醒与手动作答

| `answering.mode` | 用法 |
| --- | --- |
| `notify`（默认） | 记录课堂状态和新题。在雨课堂自行作答并提交。 |
| `assist` | 生成 AI 答案建议。核对后在雨课堂手动提交。 |

## Windows 快速开始

安装 Python 3.10–3.14。下载并解压源码 ZIP，或运行 `git clone https://github.com/tntexploding/AutoYKT.git`。在包含 `start-server.cmd` 的目录打开 PowerShell。

```powershell
py -3 -m venv .venv-server
.\.venv-server\Scripts\python.exe -m pip install .
.\start-server.cmd init
notepad "$env:LOCALAPPDATA\AutoYKT\server\config.server.yaml"
```

把 `platform.base_url` 设为你平时登录的雨课堂站点地址，例如 `https://changjiang.yuketang.cn`，并保留 `answering.mode: notify`。

```powershell
.\start-server.cmd check
.\start-server.cmd login
.\start-server.cmd probe
.\start-server.cmd run
```

按 `login` 的输出完成本人账号的 Web 扫码登录，再用 `probe` 查看活动课堂。以后可以用 `run` 或双击 `start-server.cmd` 启动，用 `Ctrl+C` 停止。

## 查看提醒与 AI 建议

服务启动后，在另一个 PowerShell 窗口跟踪本地事件。

```powershell
Get-Content "$env:LOCALAPPDATA\AutoYKT\server\data\events.jsonl" -Encoding utf8 -Tail 20 -Wait
```

`question_detected` 是新题提醒，`question_ready` 包含已读取的题面。需要把提醒发给自己的通知服务时，配置 `webhook.url`。

需要 AI 建议时，把 `answering.mode` 改为 `assist`。在 `answering.providers` 中配置兼容 OpenAI API 的服务地址、模型名和密钥环境变量，图片题需使用支持图片输入的模型。重启服务后，在 `answer_ready` 事件中查看建议，再到雨课堂手动提交。

## 自动提交（可选）

确认课程允许自动提交后，显式把 `answering.mode` 设为 `auto`，配置模型并重启服务。程序会尝试提交已发布且仍可作答的题目，`submission_result` 记录平台回执。需要恢复手动作答时，把模式改回 `notify` 或 `assist` 并重启。

## Docker 与命令行

[部署与配置](docs/server.md)包含 Docker Compose、命令行运行、模型密钥、Webhook 和课件导出的说明。

## 许可证

[MIT](LICENSE)
