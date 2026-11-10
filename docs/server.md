# 部署与配置

AutoYKT 使用普通雨课堂 Web 会话。HTTP 用于发现课堂、读取课件和题目、提交答案与回读结果。WebSocket 用于接收课堂事件。

## 配置

使用 [配置示例](../config.server.example.yaml)或运行 `autoykt init` 生成私有配置。相对路径以配置文件所在目录为基准。

| 配置 | 用法 |
| --- | --- |
| `platform.base_url` | 雨课堂站点的来源地址，例如 `https://changjiang.yuketang.cn` |
| `platform.classroom_ids` | 限定监听的课程，空列表监听当前账号的活动课堂 |
| `platform.session_file` | 普通 Web 扫码会话文件 |
| `answering.mode` | `notify` 记录新题，`assist` 生成建议，`auto` 提交并回读 |
| `answering.providers` | 配置兼容 OpenAI API 的模型地址、模型名和密钥 |
| `answering.precompute` | 为可读取的内嵌题目提前生成并缓存答案 |
| `data_dir` | 会话之外的运行数据目录，保存心跳、事件、SQLite 缓存和图片 |

模型密钥可从 `api_key_env` 指定的进程环境变量读取，也可填在私有配置的 `api_key` 中。启用 `input_mode: vision` 时使用能接收图片输入的模型。独立运行可使用默认的 `webhook.url: null` 和 `control.enabled: false`。

## Windows

在仓库目录打开 PowerShell，使用 Python 3.10–3.14 创建虚拟环境。

```powershell
py -3 -m venv .venv-server
.\.venv-server\Scripts\python.exe -m pip install .
.\start-server.cmd init
notepad "$env:LOCALAPPDATA\AutoYKT\server\config.server.yaml"
```

编辑配置后运行以下命令。

```powershell
.\start-server.cmd check
.\start-server.cmd login
.\start-server.cmd probe
.\start-server.cmd run
```

`login` 显示 Web 登录二维码地址，并在可下载时保存二维码图片。扫码后，后续运行可复用有效会话，直接执行 `probe`。日常双击 [start-server.cmd](../start-server.cmd) 即可启动，按 Ctrl+C 停止。可在另一个终端执行 `.\start-server.cmd health` 查看状态。

默认配置文件位于 `%LOCALAPPDATA%\AutoYKT\server\config.server.yaml`，会话和运行数据默认位于旁边的 `data\`。已有配置时，可将其绝对路径设为 `AUTOYKT_SERVER_CONFIG`。

## Docker Compose

在仓库目录运行以下命令。

```sh
cp config.server.example.yaml config.server.yaml
cp .env.example .env
# 编辑站点、答题模式和模型设置，并设置密钥。
docker compose build
docker compose run --rm autoykt check
docker compose run --rm autoykt login
docker compose run --rm autoykt probe
docker compose up -d
docker compose logs -f autoykt
```

[Compose 配置](../compose.yaml)将 `config.server.yaml` 以只读方式挂载，并把会话和运行数据保存到 `autoykt_data` 持久卷。用 `docker compose exec autoykt python -m autoykt.server health` 查看运行状态。

## 直接运行

在已激活的 Python 虚拟环境中运行以下命令。

```sh
python -m pip install .
autoykt init
# 编辑生成的 config.server.yaml。
autoykt check
autoykt login
autoykt probe
autoykt run
```

`python -m autoykt.server` 和 `autoykt-server` 提供同一组命令。`probe` 只读取账号会话和活动课堂列表。需要额外入班确认的课堂可在雨课堂完成确认。

## 课件、备答和提交

目录接口失败时，服务记录失败状态并稍后重试，同时继续从课堂事件发现课件。已提取的页面和题目保存在本地。预先拿不到题面时，服务在发题后读取题面并实时求解。

备答按课程、课件、题目和内容版本缓存。`auto` 模式在提交前核对当前题面、题目是否已发布以及是否还可作答。平台回执和回读结果写入运行事件。

可按课程导出已取得的课件页和备答。

```sh
autoykt courseware export ./export --course 课程ID
```

`status.json` 保存运行心跳，`events.jsonl` 记录课堂、答案和提交事件。配置 `webhook.url` 后可把选定的事件发送到自己的服务。
