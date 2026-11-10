# AutoYKT

AutoYKT 是雨课堂 Web 自动答题服务。它通过 HTTP 发现课堂、读取课件和题面，通过 WebSocket 接收课堂事件。服务可提取账号能访问的课件、用 AI 提前作答并缓存答案，也可在老师发题后实时求解。

| 模式 | 行为 |
| --- | --- |
| `notify` | 记录新题和课堂状态 |
| `assist` | 生成备答和答案建议，由本人提交 |
| `auto` | 复用备答或实时求解，提交当前可作答的题目并回读结果 |

用 `answering.mode` 切换模式。课件目录不可用时，服务继续从课堂事件发现课件和题目。

## Windows

安装 Python 3.10–3.14，在仓库目录运行。

```powershell
py -3 -m venv .venv-server
.\.venv-server\Scripts\python.exe -m pip install .
.\start-server.cmd init
notepad "$env:LOCALAPPDATA\AutoYKT\server\config.server.yaml"
```

填写雨课堂站点、答题模式和模型设置。依次执行 `start-server.cmd check`、`start-server.cmd login` 和 `start-server.cmd probe`，核对配置、完成 Web 扫码并检查课堂列表。之后双击 `start-server.cmd` 启动。

## Docker 与命令行

[部署与配置](docs/server.md)包含 Docker Compose、Windows 启动、Web 会话、课件导出和运行状态的操作说明。
