"""Persistent runtime choices and a private HTTP interface for the Bot worker."""

from __future__ import annotations

import asyncio
import hmac
import json
import os
import re
import sqlite3

from aiohttp import web


MODE_NAMES = {"auto": "全自动", "assist": "半自动", "notify": "仅提醒"}


class SubmissionPaused(RuntimeError):
    """A user changed mode or paused before the answer POST was sent."""


class RuntimeSettings:
    """Restore only explicit owner overrides; leave private YAML untouched."""

    def __init__(self, config):
        self.config = config
        self.domain = config.platform.base_url
        self.db = sqlite3.connect(config.storage / "control.sqlite3")
        self.db.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS settings (domain TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE IF NOT EXISTS commands (id TEXT PRIMARY KEY, response TEXT);
        """
        )
        row = self.db.execute(
            "SELECT value FROM settings WHERE domain=?", (self.domain,)
        ).fetchone()
        self.saved = json.loads(row[0]) if row else {}
        self.paused = bool(self.saved.get("paused", False))
        self.generation = 0
        if "mode" in self.saved:
            config.answering.mode = self.saved["mode"]
        if "precompute" in self.saved:
            config.answering.precompute = self.saved["precompute"]

    def save(self, **changes):
        if "mode" in changes or "paused" in changes:
            self.generation += 1
        self.saved.update(changes)
        self.paused = bool(self.saved.get("paused", False))
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO settings VALUES (?,?)",
                (self.domain, json.dumps(self.saved)),
            )

    def claim(self, request_id):
        row = self.db.execute(
            "SELECT response FROM commands WHERE id=?", (request_id,)
        ).fetchone()
        if row:
            return json.loads(row[0])
        self.finish(
            request_id,
            {
                "ok": False,
                "message": "上次指令结果待核对；不会自动重做，请查询状态。",
            },
        )
        return None

    def finish(self, request_id, response):
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO commands VALUES (?,?)",
                (request_id, json.dumps(response, ensure_ascii=False)),
            )

    def close(self):
        self.db.close()


class ControlServer:
    """Authenticated loopback operations; no arbitrary code or shell commands."""

    def __init__(self, service):
        self.service = service
        self.token = os.environ.get(service.config.control.token_env, "")
        self.runner = None
        self.lock = asyncio.Lock()

    async def start(self):
        if not self.service.config.control.enabled:
            return
        if not self.token:
            raise ValueError("Bot control requires a nonempty control token")
        app = web.Application(client_max_size=4096)
        app.router.add_post("/control/command", self.command)
        self.runner = web.AppRunner(app, access_log=None)
        await self.runner.setup()
        await web.TCPSite(
            self.runner, "127.0.0.1", self.service.config.control.port
        ).start()

    async def close(self):
        if self.runner:
            await self.runner.cleanup()

    async def command(self, request):
        if not self.token or not hmac.compare_digest(
            request.headers.get("Authorization", "").encode(),
            ("Bearer " + self.token).encode(),
        ):
            return web.json_response(
                {"ok": False, "message": "控制鉴权失败。"}, status=401
            )
        try:
            body = await request.json()
            request_id, action = body["request_id"], body["action"]
            if (
                not isinstance(request_id, str)
                or not re.fullmatch(r"[A-Za-z0-9:_-]{1,80}", request_id)
                or not isinstance(action, str)
            ):
                raise ValueError("invalid command")
        except (ValueError, KeyError, TypeError):
            return web.json_response(
                {"ok": False, "message": "指令格式无效。"}, status=400
            )
        async with self.lock:
            prior = self.service.runtime.claim(request_id)
            if prior is not None:
                return web.json_response(prior)
            try:
                response = await self.apply(action)
            except (ValueError, RuntimeError) as error:
                response = {"ok": False, "message": str(error)}
            self.service.runtime.finish(request_id, response)
            return web.json_response(response)

    def require_models(self):
        for provider in self.service.config.answering.providers:
            try:
                if provider.resolve_api_key():
                    return
            except ValueError:
                continue
        raise ValueError("模型或密钥尚未配置，模式保持原值。")

    async def apply(self, action):
        service, policy = self.service, self.service.config.answering
        if action == "status":
            return {
                "ok": True,
                "message": self.describe(),
                "snapshot": service.snapshot("running"),
            }
        if action in {"mode_auto", "mode_assist", "mode_notify"}:
            mode = action.removeprefix("mode_")
            if mode != "notify":
                self.require_models()
            policy.mode = mode
            if mode != "notify":
                policy.precompute = True
            service.runtime.save(
                mode=mode, paused=False, precompute=policy.precompute
            )
            if mode == "notify" and service.warmer:
                service.warmer.cancel()
            else:
                service.start_precompute()
            return {
                "ok": True,
                "message": "已切换为"
                + MODE_NAMES[mode]
                + (
                    "。仅新发题进入自动提交；已发出的提交等待回执。"
                    if mode == "auto"
                    else "。"
                ),
                "mode": mode,
            }
        if action == "pause":
            service.runtime.save(paused=True)
            return {
                "ok": True,
                "message": "已暂停后续自动提交；题目提醒和 AI 建议继续。已经发出的请求等待回执。",
            }
        if action == "use_web":
            await service.restore_entry("web")
            return {
                "ok": True,
                "entry_mode": "web",
                "message": "已恢复正常 Web 入口。",
            }
        states = [state for state in service.states.values() if not state.ended]
        if action not in {"refresh", "prepare"}:
            raise ValueError("未知控制指令。")
        if not states:
            return {
                "ok": False,
                "message": "目前没有进行中的课堂；发现课堂后会自动提取。",
            }
        if action == "refresh":
            for state in states:
                service.schedule_courseware(state, force=True)
            return {
                "ok": True,
                "message": "已安排重新提取可见课件，完成或失败后会通知。",
            }
        self.require_models()
        if policy.mode == "notify":
            return {
                "ok": False,
                "message": "仅提醒模式不调用 AI，请先选择半自动或全自动。",
            }
        policy.precompute = True
        service.runtime.save(precompute=True)
        service.preparation_failures.clear()
        service.start_precompute()
        return {
            "ok": True,
            "message": "已开始准备可见题目；已有有效答案会复用，进度会自动通知。",
        }

    def describe(self):
        snapshot = self.service.snapshot("running")
        statuses = {
            "running": "运行中",
            "degraded": "部分异常",
            "reconnecting": "重连中",
            "login_required": "等待微信登录",
            "stopped": "已停止",
        }
        extraction = {
            "waiting": "等待可见课件",
            "fetching": "正在提取",
            "all_visible": "全部已发现的可见课件已取回",
            "partial": "仅取回部分内容",
            "unavailable": "暂未取得内容，继续实时捕获",
        }
        connections = {
            "connected": "已连接",
            "connecting": "连接中",
            "discovered": "等待连接",
            "disconnected": "已断开，等待恢复",
            "entry_refused": "进班被拒绝",
            "login_required": "等待微信登录",
        }
        lines = [
            "模式："
            + MODE_NAMES[self.service.config.answering.mode]
            + ("（自动提交已暂停）" if self.service.runtime.paused else ""),
            "服务：" + statuses.get(snapshot["status"], "待核对"),
            "入口：" + snapshot["entry_mode"],
        ]
        for state in snapshot["classrooms"]:
            progress = state["progress"]
            lines.extend(
                [
                    f"课堂：{state['name'] or state['lesson_id']} · {connections.get(state['connection'], '等待连接')}",
                    f"课件 {progress['presentations']} 套 / {progress['slides']} 页；题目 {progress['questions']} 道",
                    f"备答 {progress['prepared']}；失败 {progress['failed']}；本人已答 {progress['answered']}；待准备 {progress['remaining']}",
                    f"提取：{extraction.get(state['extraction'], '待核对')}；未发布内容可见性未知",
                ]
            )
        if not snapshot["classrooms"]:
            lines.append("等待上课课堂。")
        return "\n".join(lines)
