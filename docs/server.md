# 独立答题服务

服务使用普通雨课堂 Web 会话，直接通过 HTTP 发现课堂、读取课件，通过 WebSocket 接收新题。可用题面交给配置中的模型，已备答的题目按内容版本命中缓存。自动模式只提交当前已发布且尚未关闭的题目，并保存平台回执和回读结果。

## 直接启动

```sh
python -m pip install .
python -m autoykt.server init
python -m autoykt.server check
python -m autoykt.server login
python -m autoykt.server probe
python -m autoykt.server run
```

在 `init` 后编辑私有的 `config.server.yaml`。将 `platform.base_url` 指向自己上课的站点，配置 `answering.providers` 中的模型、地址和密钥环境变量。终端命令从当前进程读取环境变量。用 `platform.classroom_ids` 限定课程，空列表监听当前账号的全部活动课堂。

旧 Web 配置及其会话可沿用。旧组合配置由对应入口模块读取，独立答题使用 Web 配置与 Web 会话。

| 模式 | 行为 |
| --- | --- |
| `notify` | 在本地记录发题、题面和状态 |
| `assist` | 生成备答或建议，由本人提交 |
| `auto` | 收到新题后按当前题面复用备答或求解，自动提交并回读 |

`login` 给出普通 Web 登录二维码地址，并尽可能在数据目录保存二维码文件。扫码后会话存入 `platform.session_file`。课堂若要求额外入班确认，按雨课堂页面完成，再继续监听。默认的 `webhook.url: null` 和 `control.enabled: false` 适用于单独运行。

## Windows 启动

在仓库目录打开 PowerShell。首次使用时准备 Python 3.10–3.14，并创建供 Web 答题服务使用的独立环境。安装了多个版本时，可以用 `py -3.12` 或 `py -3.14` 指定版本。

```powershell
py -3 -m venv .venv-server
.\.venv-server\Scripts\python.exe -m pip install .
.\start-server.cmd init
notepad "$env:LOCALAPPDATA\AutoYKT\server\config.server.yaml"
```

在配置中填写站点、模型及密钥来源。初次核对可使用默认的 `notify` 模式。需要自动提交时，把 `answering.mode` 改为 `auto`。双击启动时，模型密钥可写在这份私有配置的 `answering.providers[0].api_key`，也可使用 Windows 用户环境变量。

```powershell
.\start-server.cmd check
.\start-server.cmd login
.\start-server.cmd probe
.\start-server.cmd run
```

`login` 完成普通 Web 扫码。已有有效 Web 会话时，可以在配置中指向该会话文件，直接执行 `probe`。日常双击 `start-server.cmd` 即可运行，按 Ctrl+C 停止。在另一个终端运行 `.\start-server.cmd health` 可查看当前状态。

配置默认位于 `%LOCALAPPDATA%\AutoYKT\server\config.server.yaml`，会话和运行记录位于同目录的 `data\`。如果已有私有配置，可把其绝对路径设为 `AUTOYKT_SERVER_CONFIG`。Windows 桌面视觉模式继续使用原有 `start-class.cmd` 和自己的配置。

## Docker

```sh
cp config.server.example.yaml config.server.yaml
cp .env.example .env
# 编辑配置和模型密钥。
docker compose build
docker compose run --rm autoykt login
docker compose run --rm autoykt probe
docker compose up -d
docker compose logs -f autoykt
```

Docker 示例使用 `/data` 持久卷，直接运行的示例使用配置所在目录下的 `data/`。该目录保存 `status.json`、`events.jsonl`、SQLite 缓存和下载的图片。课件可离线导出。

```sh
python -m autoykt.server courseware export ./export --course 课程ID
```

## 课件目录与备答

- 目录请求使用课堂页面的 Cookie 和 Referer。目录失败时保留 HTTP 状态或业务码，相同失败只记一次，并在至少五分钟后重试。实时课堂事件继续发现课件。
- 已取得的完整课件在收到课件更新，或目录显示还有更多页时刷新。内容相同的状态合并，缺页和图片下载失败的课件仍会补取。目录不可用时，已发现的课件会保留，提取状态标为部分。
- 备答按课程、课件、题目和内容版本保存，兼容旧 SQLite 缓存。内容回到已知版本时可复用对应答案。发题后仍读取当前题面和作答状态，提交前再次校验内容、截止时间和平台状态。缺少备答时走实时 AI。

服务提供可选的 JSON Webhook 和回环控制接口。目录失败降级、备答复用和提交回读已通过本地 HTTP/WebSocket 验证。真实站点能否返回完整目录仍需课堂复测。
