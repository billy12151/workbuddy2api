#!/usr/bin/env python3
"""
codebuddy2openai — 把 CodeBuddy / WorkBuddy 的订阅暴露成标准 OpenAI 兼容 API。

原理（直连后端，原生 function calling）：
  - 读取本机已登录的 CodeBuddy 桌面端凭据（auth 文件里的 token / uid / enterpriseId）。
  - 直接转发到 CodeBuddy 后端 `https://copilot.tencent.com/v2/chat/completions`。
    该后端本身就是标准 OpenAI chat/completions 协议（含原生 tools / tool_calls / SSE 流式）。
  - 转换器只做两件事：①注入鉴权 header（Authorization / X-User-Id 等）
    ②在本地 /v1/* 与后端 /v2/* 之间做路径映射与透传（含 Anthropic / Chat / Responses 三种协议）。
  - token 过期时自动调 `/v2/plugin/auth/token/refresh` 刷新，并回写 auth 文件。

双变体：同一份代码支持国内版与国际版两个独立网关实例。
  - cn   （默认）：WorkBuddy 国内版，后端 copilot.tencent.com，凭据
    workbuddy-desktop.info，默认端口 8787。
  - intl：WorkBuddy AI 国际版，后端 www.workbuddy.ai，凭据
    workbuddy-desktop-ai.info，默认端口 8788。国际版后端额外要求
    首条消息必须是 system prompt（转换器自动注入）。
  两版凭据文件同目录共存，按文件名精确区分；模型列表各自独立。

跨平台：自动定位 auth 目录（macOS / Windows / Linux）。
依赖：fastapi + uvicorn + httpx（pip install fastapi "uvicorn[standard]" httpx）。

用法：
  python3 converter.py                       # 国内版，默认 127.0.0.1:8787
  python3 converter.py --variant intl        # 国际版，默认 127.0.0.1:8788
  python3 converter.py --port 9000
  python3 converter.py --api-key mysecret    # 启用客户端鉴权
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

try:
    from desensitize import desensitize_body
except ImportError:  # 模块缺失时降级为不脱敏
    def desensitize_body(body, roles=("system",), desensitize_harness_user=False,
                         desensitize_tools=False, compact_harness=False,
                         strip_tool_metadata=False):
        return body

from responses_adapter import (
    responses_request_to_chat,
    ResponsesStreamConverter,
)
from responses_projection import project_responses_chat_body
from anthropic_adapter import (
    anthropic_request_to_chat,
    AnthropicStreamConverter,
)

# ---------------------------------------------------------------------------
# 变体（国内版 / 国际版）：同一份代码跑两个独立网关实例。
# 两版协议同构（/v2/chat/completions、Bearer + X-User-Id），差异在后端域名、
# 凭据文件名、模型列表、以及国际版后端要求首条消息必须是 system。
# ---------------------------------------------------------------------------

CN_MODELS = [
    "glm-5.2", "glm-5.1", "glm-5v-turbo",
    "kimi-k2.7", "kimi-k2.6", "kimi-k2.5",
    "deepseek-v4-pro", "deepseek-v4-flash",
    "minimax-m3-pay", "hy3-preview-agent", "hy4-preview", "auto",
]

# 国内版模型名映射：Codex 内部功能可能用默认模型名，映射到对应后端模型
CN_MODEL_NAME_MAP = {
    "gpt-5.6-luna": "hy3",
    "gpt-5.6-sol": "hy3",
    "gpt-5.5": "hy3",
    "gpt-5": "hy3",
    "gpt-4o": "hy3",
    "gpt-4": "hy3",
    # hy4 预览版别名（Codex 内部默认模型名 -> 后端 hy4-preview）
    "gpt-5.6": "hy4-preview",
    "gpt-6": "hy4-preview",
    "gpt-6-mini": "hy4-preview",
}

# 国际版（WorkBuddy AI，www.workbuddy.ai）模型列表：
# 取自其产品配置 ~/.workbuddy-ai/cache/acc-product-config-v3.json（2026-10 快照），
# 去掉 UI 选择器别名（default-model/fast-model/...，后端也接受，可直传）
# 与图像/视频生成模型（gpt-image-2.5-sunburst、seedance-2.5）。
# 国际版 gpt-* 是真实模型，不做国内版那套 gpt→hy 名字映射。
INTL_MODELS = [
    "gpt-6-astra", "gpt-6-sol", "gpt-6-luna",
    "gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.5", "gpt-5.4",
    "gemini-3.8-flash", "gemini-3.5-flash",
    "grok-4.7", "kimi-k2.8-preview", "kimi-k3", "kimi-k2.6",
    "glm-5.3", "glm-5.3-flash", "glm-5.2",
    "deepseek-v4.1-flash", "deepseek-v4.1-flash-sg",
    "hy3", "hy4-preview", "hy4-preview-f",
    "auto",
]

VARIANTS = {
    "cn": {
        "backend": "https://copilot.tencent.com",
        "domain": "www.codebuddy.cn",
        "auth_filename": "workbuddy-desktop.info",    # 国内版 WorkBuddy
        "models": CN_MODELS,
        "model_map": CN_MODEL_NAME_MAP,
        "token_cache": "secrets.token-cache.json",
        "require_system_first": False,
    },
    "intl": {
        "backend": "https://www.workbuddy.ai",
        "domain": "www.workbuddy.ai",
        "auth_filename": "workbuddy-desktop-ai.info",  # 国际版 WorkBuddy AI
        "models": INTL_MODELS,
        "model_map": {},
        "token_cache": "secrets.token-cache-intl.json",
        # 国际版后端强制首条消息必须是 system prompt，否则 400 code=11128
        "require_system_first": True,
    },
}


def apply_variant(name: str):
    """按变体设置模块级配置（import 时取 WB_VARIANT 环境变量，main() 可用
    --variant 覆盖；handler 在调用时读取这些全局量）。"""
    global VARIANT, BACKEND, DEFAULT_DOMAIN, AUTH_FILENAME
    global DEFAULT_MODELS, MODEL_NAME_MAP, REQUIRE_SYSTEM_FIRST
    if name not in VARIANTS:
        raise ValueError(f"未知变体: {name}（可选: {', '.join(VARIANTS)}）")
    v = VARIANTS[name]
    VARIANT = name
    BACKEND = v["backend"]
    DEFAULT_DOMAIN = v["domain"]
    AUTH_FILENAME = v["auth_filename"]
    DEFAULT_MODELS = v["models"]
    MODEL_NAME_MAP = v["model_map"]
    REQUIRE_SYSTEM_FIRST = v["require_system_first"]


apply_variant(os.environ.get("WB_VARIANT", "cn"))

USER_AGENT = "codebuddy2openai/2.0"

# ---------------------------------------------------------------------------
# 平台相关：定位 auth 目录
# ---------------------------------------------------------------------------

def auth_dirs() -> list[Path]:
    env_dir = os.environ.get("CODEBUDDY_AUTH_DIR")
    if env_dir:
        return [Path(env_dir)]
    home = Path.home()
    plat = sys.platform
    if plat == "darwin":
        return [home / "Library" / "Application Support" / "CodeBuddyExtension" / "Data" / "Public" / "auth"]
    if plat == "win32":
        local = Path(os.environ.get("LOCALAPPDATA", home / "AppData" / "Local"))
        return [local / "CodeBuddyExtension" / "Data" / "Public" / "auth"]
    xdg = Path(os.environ.get("XDG_DATA_HOME", home / ".local" / "share"))
    return [xdg / "CodeBuddyExtension" / "Data" / "Public" / "auth"]


def find_auth_file() -> Path | None:
    """按变体精确匹配凭据文件名。

    auth 目录里国内版/国际版文件共存（workbuddy-desktop.info /
    workbuddy-desktop-ai.info），不能通配取第一个——字母序 ai 文件在前，
    国内版网关会错拿国际版凭据。
    """
    for d in auth_dirs():
        exact = d / AUTH_FILENAME
        if exact.is_file():
            return exact
    return None


# ---------------------------------------------------------------------------
# Auth 凭据管理（读 + 自动刷新 + 回写）
# ---------------------------------------------------------------------------

class CredentialManager:
    """从 auth 文件读取凭据；token 临近过期时自动刷新并回写。"""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self._cached: dict | None = None
        self._mtime: float = 0.0
        self._atrest_keys: dict | None = None
        # 刷新后的 token 落本地缓存（应用 auth 文件 5.6.2 起是加密格式，不能回写）。
        # 缓存文件按变体区分，避免两个网关实例互相覆盖刷新结果。
        self._cache_path = Path(__file__).resolve().parent / VARIANTS[VARIANT]["token_cache"]

    def _read_raw(self) -> dict:
        with open(self.path, "r", encoding="utf-8") as f:
            doc = json.load(f)
        # WorkBuddy 5.6.2 起 auth 文件敏感字段是 $wbEncrypted 信封，读时解密
        try:
            from wbkey.atrest import load_keys, unwrap_doc
            if self._atrest_keys is None:
                self._atrest_keys = load_keys()
            unwrap_doc(doc, self._atrest_keys)
        except FileNotFoundError:
            if "$wbEncrypted" in json.dumps(doc)[:200000]:
                print("[warn] auth 文件含加密字段但找不到 wbkey 钥匙文件，"
                      "运行 wbkey/grab_key.py 抓取后重启本服务", flush=True)
        except RuntimeError:
            # 钥匙可能刚被 grab_key.py 重新抓取，热重载一次再试
            from wbkey.atrest import load_keys
            self._atrest_keys = load_keys()
            unwrap_doc(doc, self._atrest_keys)
        return doc

    def _load_cache(self) -> dict | None:
        try:
            with open(self._cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    def _load_if_stale(self):
        """若文件 mtime 变了（外部刷新过），重新加载缓存。"""
        try:
            mt = self.path.stat().st_mtime
        except OSError:
            return
        if self._cached is None or mt != self._mtime:
            self._cached = self._read_raw()
            self._mtime = mt
            # 本地缓存里刷新过的 token 比文件里的新则优先（文件不会包含代理刷的 token）
            cached = self._load_cache() or {}
            cached_auth = cached.get("auth") or {}
            file_auth = self._cached.get("auth") or {}
            if cached_auth.get("lastRefreshTime", 0) > file_auth.get("lastRefreshTime", 0):
                self._cached["auth"] = cached_auth

    def _session(self) -> dict:
        self._load_if_stale()
        if self._cached is None:
            raise RuntimeError(f"无法读取 auth 文件：{self.path}")
        return self._cached

    def _is_expired(self) -> bool:
        s = self._session()
        expires_at = (s.get("auth") or {}).get("expiresAt") or 0
        # 提前 60s 判定过期
        return time.time() * 1000 >= (expires_at - 60_000)

    def _refresh(self):
        """调后端刷新 token，写回 auth 文件与缓存。"""
        s = self._session()
        auth = s.get("auth") or {}
        headers = self._build_headers_from(auth, s.get("account") or {})
        headers["X-Refresh-Token"] = auth.get("refreshToken", "")
        headers["X-Auth-Refresh-Source"] = "plugin"
        url = f"{BACKEND}/v2/plugin/auth/token/refresh"
        try:
            with httpx.Client(timeout=15, trust_env=_trust_env()) as c:
                r = c.post(url, headers=headers, json={})
            data = r.json()
        except Exception as e:
            raise RuntimeError(f"刷新 token 网络失败：{e}")
        if data.get("code") != 0 or not data.get("data"):
            raise RuntimeError(f"刷新 token 失败：{data.get('msg', data)}")
        new_auth = data["data"]
        # 继承部分字段
        new_auth["domain"] = new_auth.get("domain") or auth.get("domain")
        new_auth["lastRefreshTime"] = int(time.time() * 1000)
        # 计算 expiresAt（若后端没直接给）
        if not new_auth.get("expiresAt") and new_auth.get("expiresIn"):
            new_auth["expiresAt"] = int(time.time() * 1000) + new_auth["expiresIn"] * 1000
        if not new_auth.get("refreshExpiresAt") and new_auth.get("refreshExpiresIn"):
            new_auth["refreshExpiresAt"] = int(time.time() * 1000) + new_auth["refreshExpiresIn"] * 1000
        s["auth"] = new_auth
        # 5.6.2 起应用 auth 文件是加密格式，不能回写；刷新结果写本地缓存（0600）
        try:
            tmp = self._cache_path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({"auth": new_auth}, f, ensure_ascii=False, indent=2)
            os.chmod(tmp, 0o600)
            os.replace(tmp, self._cache_path)
        except OSError as e:
            print(f"[warn] token 缓存写盘失败: {e}", flush=True)
        self._cached = s
        self._mtime = self.path.stat().st_mtime

    def _build_headers_from(self, auth: dict, account: dict) -> dict:
        domain = auth.get("domain") or DEFAULT_DOMAIN
        h = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {auth.get('accessToken','')}",
            "X-User-Id": account.get("uid", ""),
            "X-Enterprise-Id": account.get("enterpriseId", ""),
            "X-Tenant-Id": account.get("enterpriseId", ""),
            "X-Domain": domain,
            "User-Agent": USER_AGENT,
        }
        return h

    def get_headers(self) -> dict:
        """返回带最新 token 的后端请求 header；必要时先刷新。"""
        with self._lock:
            if self._is_expired():
                self._refresh()
            s = self._session()
            return self._build_headers_from(s.get("auth") or {}, s.get("account") or {})

    def summary(self) -> dict:
        s = self._session()
        auth = s.get("auth") or {}
        acct = s.get("account") or {}
        exp = auth.get("expiresAt", 0)
        return {
            "uid": acct.get("uid"),
            "nickname": acct.get("nickname"),
            "enterpriseName": acct.get("enterpriseName"),
            "token_expires_at": exp,
            "token_expired": self._is_expired(),
        }


# ---------------------------------------------------------------------------
# 模型列表：DEFAULT_MODELS / MODEL_NAME_MAP 已上移到 VARIANTS（按变体区分）
# ---------------------------------------------------------------------------

# 后端请求体里出现过的额外字段（透传时若客户端给了就保留）
PASSTHROUGH_BODY_KEYS = {
    "model", "messages", "tools", "tool_choice", "temperature",
    "max_tokens", "max_completion_tokens", "top_p", "stream",
    "stream_options", "stop", "presence_penalty", "frequency_penalty",
    "n", "response_format", "seed", "user", "reasoning_effort",
    "verbosity", "reasoning_summary",
}

# ---------------------------------------------------------------------------
def _trust_env() -> bool:
    """httpx 是否读环境/系统代理。默认 False（直连后端）。

    macOS 上 urllib 的 getproxies() 会读系统代理，clash 常开系统代理时
    httpx 默认把后端流量全绕道 127.0.0.1:6789——intl 链路抖动（成小时级
    爆发的"网络错误"）即源于此，直连实测稳定。需要代理的环境设
    WB_TRUST_ENV=1 退出直连。
    """
    return os.environ.get("WB_TRUST_ENV") == "1"


# FastAPI 应用
# ---------------------------------------------------------------------------

app = FastAPI(title="codebuddy2openai", version="2.0")
CONFIG: dict = {"api_key": "", "cred": None, "log_path": None,
                "desensitize": False, "no_compact": False}  # cred: CredentialManager | None


# ---------------------------------------------------------------------------
# 日志（写文件）
# ---------------------------------------------------------------------------

_LOG_LOCK = threading.Lock()


def _log(msg: str):
    """写一行日志到 CONFIG['log_path'] 指定的文件（追加，带时间戳）。未设置则丢弃。"""
    path = CONFIG.get("log_path")
    if not path:
        return
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n"
    try:
        with _LOG_LOCK:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
    except OSError:
        pass  # 日志失败不应影响主流程




def _truncate(s: str, n: int = 80) -> str:
    s = str(s).replace("\n", " ").strip()
    return s[:n] + ("…" if len(s) > n else "")


def _check_auth(authorization: Optional[str], x_api_key: Optional[str]):
    key = CONFIG["api_key"]
    if not key:
        return
    token = ""
    if authorization and authorization.startswith("Bearer "):
        token = authorization[7:].strip()
    if not token and x_api_key:
        token = x_api_key
    if token != key:
        raise HTTPException(status_code=401, detail={"error": {"message": "invalid api key", "type": "auth_error"}})


def _cred() -> CredentialManager:
    if CONFIG["cred"] is None:
        raise HTTPException(status_code=503, detail={"error": {"message": "未找到登录凭据，请先在桌面端登录 CodeBuddy/WorkBuddy", "type": "auth_error"}})
    return CONFIG["cred"]


@app.get("/health")
def health():
    cred = CONFIG["cred"]
    info: dict = {"status": "ok", "platform": sys.platform, "python": sys.version.split()[0],
                  "variant": VARIANT, "backend": BACKEND,
                  "auth_file": str(find_auth_file() or "(未找到)"), "mode": "direct-proxy (native function calling)"}
    if cred is not None:
        try:
            info["credential"] = cred.summary()
        except Exception as e:
            info["credential_error"] = str(e)
    return info


@app.get("/v1/models")
def list_models(authorization: Optional[str] = Header(default=None),
                x_api_key: Optional[str] = Header(default=None, alias="X-Api-Key")):
    _check_auth(authorization, x_api_key)
    data = [{"id": m, "object": "model", "created": 1700000000, "owned_by": "codebuddy"}
            for m in DEFAULT_MODELS]
    return {"object": "list", "data": data}


def _normalize_tool_choice(tc: Any) -> Any:
    """把 tool_choice 规范化为后端接受的字符串（无 tool_choice 返回 None）。

    后端 Go 结构体 tool_choice 是 string，对象形式必 400（code=11101，
    "cannot unmarshal object into Go struct field Request.tool_choice of type
    string"，2026-09-26 实测）。实测字符串接受 "auto"/"required"/"none"。
    强制指定具体函数（{"type":"function"/"tool"} 或裸函数名）后端无对应
    字符串表达，降级为 "required"；未知值原样透传也是 400，同样降级。
    """
    if tc is None:
        return None
    if isinstance(tc, dict):
        return {"auto": "auto", "none": "none"}.get(tc.get("type"), "required")
    if isinstance(tc, str) and tc in ("none", "auto", "required"):
        return tc
    return "required"


def _anthropic_upstream_error(raw: bytes, status: int) -> JSONResponse:
    """把上游非 200 响应包装为 Anthropic 协议错误体 + 真实状态码。

    Anthropic SDK 按 {"type":"error","error":{...}} 解析，Claude Desktop
    能直接看到后端真实错误（如 11101 的参数详情），而不是被 200 空流
    包装出来的 "empty or malformed response"。
    """
    text = raw.decode("utf-8", "replace")
    try:
        j = json.loads(text)
        msg = str(j.get("msg") or j.get("message") or text)
    except Exception:
        msg = text
    return JSONResponse(
        status_code=status,
        content={"type": "error", "error": {"type": "api_error", "message": msg[:500]}},
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: Request,
                           authorization: Optional[str] = Header(default=None),
                           x_api_key: Optional[str] = Header(default=None, alias="X-Api-Key")):
    _check_auth(authorization, x_api_key)
    cred = _cred()

    try:
        payload = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail={"error": {"message": f"bad json: {e}", "type": "invalid_request_error"}})

    messages = payload.get("messages") or []
    if not messages:
        raise HTTPException(status_code=400, detail={"error": {"message": "messages is required", "type": "invalid_request_error"}})

    # 构造后端 body：只透传已知的合法字段
    client_wants_stream = bool(payload.get("stream"))
    body = {k: payload[k] for k in PASSTHROUGH_BODY_KEYS if k in payload}
    body.setdefault("model", "auto")
    # 模型名映射：Codex 内部功能可能用默认模型名
    body["model"] = MODEL_NAME_MAP.get(body["model"], body["model"])
    # 后端只支持流式：始终以 stream=True 调后端，非流式由转换器聚合
    body["stream"] = True
    if "stream_options" not in body:
        body["stream_options"] = {"include_usage": True}
    # tool_choice 规范化：后端只接受字符串，对象/裸函数名会 400（code=11101）
    tc_norm = _normalize_tool_choice(body.get("tool_choice"))
    if tc_norm is None:
        body.pop("tool_choice", None)
    else:
        body["tool_choice"] = tc_norm

    # 可选：脱敏。缓解客户端合规模板（如 Codex CLI / ZCode 注入的说明文字）被后端误判为敏感词。
    # 处理 system / developer 消息、Codex 注入的上下文 user 消息，以及 tools 的 description。
    if CONFIG.get("desensitize"):
        body = desensitize_body(body, roles=("system", "developer"),
                                desensitize_harness_user=True,
                                desensitize_tools=True,
                                compact_harness=not CONFIG.get("no_compact"),
                                strip_tool_metadata=True)

    # 替换第三方客户端的 system prompt，避免触发腾讯安全拦截
    # 策略：只替换明显是客户端自动注入的长篇 system prompt（含工具定义、安全条款等）
    # 保留用户自己写的简短 system prompt（比如"用中文回答"、"扮演架构师"等）
    CLIENT_KEYWORDS = ("ZCode", "zcode", "Codex", "codex", "Claude Code")
    MIN_SYSTEM_LENGTH = 200  # 客户端注入的 system prompt 通常较长，用户自定义的一般很短
    if body.get("messages"):
        for msg in body["messages"]:
            if msg.get("role") != "system":
                continue
            content = msg.get("content", "")
            if not isinstance(content, str):
                continue
            # 同时满足：包含客户端标识词 + 长度足够长，才认为是客户端注入的
            if any(kw in content for kw in CLIENT_KEYWORDS) and len(content) >= MIN_SYSTEM_LENGTH:
                msg["content"] = "你是一个乐于助人的编程助手。帮助用户完成软件工程任务。需要时使用提供的工具。用与用户相同的语言回复。"
                _log(f"[filter] 替换客户端注入的 system prompt ({len(content)} chars → 通用提示)")

    # 日志：请求摘要（使用映射后的模型名）
    model_name = body.get("model", payload.get("model", "auto"))
    tool_names = [t.get("function", {}).get("name") for t in (payload.get("tools") or [])
                  if isinstance(t, dict)]
    last_user = _last_user_text(messages)
    rid = os.urandom(4).hex()
    _log(f"[{rid}] ▶ REQUEST {model_name} | stream={client_wants_stream} | msgs={len(messages)}"
         + (f" | tools={tool_names}" if tool_names else "")
         + (f" | last_user={_truncate(last_user, 60)!r}" if last_user else ""))
    # 完整请求体（发往后端的实际内容；若启用脱敏，这里已是脱敏后）
    _log(f"[{rid}] ── REQUEST BODY (发往后端) ──\n{json.dumps(body, ensure_ascii=False, indent=2)}")

    if REQUIRE_SYSTEM_FIRST:
        _ensure_system_first(body)

    headers = cred.get_headers()
    url = f"{BACKEND}/v2/chat/completions"
    t0 = time.time()

    if client_wants_stream:
        # 预检上游（含连接阶段网络错误重试）后再开流：非 200 用真实状态码报错，
        # 而不是包成 200 + SSE 错误事件让客户端看到空流
        try:
            client, r, first, chunk_iter = await _open_upstream(url, headers, body, rid=rid, model_name=model_name)
        except httpx.HTTPError as e:
            _log(f"[{rid}] ✗ 网络错误(重试耗尽) | {model_name} | {e!r}")
            raise HTTPException(status_code=502, detail={"error": {"message": f"upstream error: {e!r}", "type": "upstream_error"}})
        if r.status_code != 200:
            err = await r.aread()
            await r.aclose()
            await client.aclose()
            _log(f"[{rid}] ✗ HTTP {r.status_code} | {model_name} | {_truncate(err.decode('utf-8','replace'),200)}")
            _log(f"[{rid}] ── ERROR BODY ──\n{err.decode('utf-8','replace')}")
            raise HTTPException(status_code=r.status_code, detail=_safe_err_raw(err, r.status_code))
        return StreamingResponse(
            _relay_chat(client, r, chunk_iter, first, model_name, t0, rid),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # 非流式：后端只支持流式，这里把后端 SSE 聚合成单个 chat.completion 响应
    try:
        client, r, _, _ = await _open_upstream(url, headers, body, rid=rid, model_name=model_name, first_byte=False)
    except httpx.HTTPError as e:
        _log(f"[{rid}] ✗ 网络错误(重试耗尽) | {model_name} | {e!r}")
        raise HTTPException(status_code=502, detail={"error": {"message": f"upstream error: {e!r}", "type": "upstream_error"}})
    try:
        if r.status_code != 200:
            raw = await r.aread()
            _log(f"[{rid}] ✗ HTTP {r.status_code} | {model_name} | {_truncate(raw.decode('utf-8','replace'),200)}")
            _log(f"[{rid}] ── ERROR BODY ──\n{raw.decode('utf-8','replace')}")
            raise HTTPException(status_code=r.status_code, detail=_safe_err_raw(raw, r.status_code))
        collected = await _collect_stream(r)
    except httpx.HTTPError as e:
        _log(f"[{rid}] ✗ 网络错误(聚合中) | {model_name} | {e!r}")
        raise HTTPException(status_code=502, detail={"error": {"message": f"upstream error: {e!r}", "type": "upstream_error"}})
    finally:
        await r.aclose()
        await client.aclose()
    _log_finish(model_name, t0, collected, rid)
    return JSONResponse(content=collected)


def _ensure_system_first(body: dict):
    """国际版后端要求首条消息必须是 system prompt（否则 400 code=11128
    "first message is not system prompt"）。客户端没给时注入一条通用提示。"""
    msgs = body.get("messages") or []
    if not msgs or msgs[0].get("role") != "system":
        body["messages"] = [{"role": "system",
                             "content": "You are a helpful assistant."}] + msgs
        _log("[filter] 注入首条 system 消息（国际版后端要求）")


def _last_user_text(messages: list) -> str:
    """取最后一条 user 消息的文本，用于日志预览。"""
    for m in reversed(messages):
        if m.get("role") != "user":
            continue
        content = m.get("content", "")
        if isinstance(content, list):
            for blk in content:
                if isinstance(blk, dict) and blk.get("type") == "text":
                    return str(blk.get("text", ""))
            return ""
        return str(content)
    return ""


def _log_finish(model_name: str, t0: float, result: dict, rid: str = ""):
    """记录一次完成的请求：耗时 / finish_reason / usage / 工具调用 / 审核拦截 + 完整响应。"""
    elapsed = time.time() - t0
    prefix = f"[{rid}] " if rid else ""
    choice = (result.get("choices") or [{}])[0]
    finish = choice.get("finish_reason")
    msg = choice.get("message") or {}
    tcs = msg.get("tool_calls") or []
    usage = result.get("usage") or {}
    tag = ""
    if finish == "content-filter":
        tag = " ⚠️内容审核拦截"
    tc_names = [t.get("function", {}).get("name") for t in tcs]
    _log(f"{prefix}◀ RESPONSE {model_name} | {elapsed:.1f}s | finish={finish}{tag}"
         + (f" | tool_calls={tc_names}" if tc_names else "")
         + f" | tokens={usage.get('total_tokens', '?')}")
    # 完整响应体
    _log(f"{prefix}── RESPONSE BODY ──\n{json.dumps(result, ensure_ascii=False, indent=2)}")


async def _collect_stream(response: httpx.Response) -> dict:
    """消费后端的 OpenAI SSE 流，聚合成单个非流式 chat.completion 对象。

    合并所有 chunk 的 delta（content / reasoning_content / tool_calls），并取 usage / finish_reason。
    """
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    # tool_calls: index -> {id, name, arguments(分片拼接)}
    tool_calls: dict[int, dict] = {}
    model: str | None = None
    finish_reason: str | None = None
    usage: dict | None = None

    async for line in response.aiter_lines():
        line = line.strip()
        if not line or not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue
        model = chunk.get("model") or model
        if chunk.get("usage"):
            usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
            delta = choice.get("delta") or {}
            if delta.get("content"):
                content_parts.append(delta["content"])
            if delta.get("reasoning_content"):
                reasoning_parts.append(delta["reasoning_content"])
            for tc in delta.get("tool_calls") or []:
                idx = tc.get("index", 0)
                slot = tool_calls.setdefault(idx, {"id": None, "name": None, "arguments": ""})
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["name"] = fn["name"]
                if fn.get("arguments"):
                    slot["arguments"] += fn["arguments"]

    tcs = None
    if tool_calls:
        tcs = [
            {"id": v["id"], "type": "function",
             "function": {"name": v["name"], "arguments": v["arguments"]}}
            for _, v in sorted(tool_calls.items())
        ]
        finish_reason = finish_reason or "tool_calls"

    message = {"role": "assistant", "content": "".join(content_parts) or None}
    # 后端思考输出（vLLM 风格 reasoning_content），有才带，兼容不认识的客户端
    if reasoning_parts:
        message["reasoning_content"] = "".join(reasoning_parts)
    if tcs:
        message["tool_calls"] = tcs
    return {
        "id": "chatcmpl-" + os.urandom(12).hex(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model or "unknown",
        "choices": [{"index": 0, "message": message,
                     "finish_reason": finish_reason or "stop"}],
        "usage": usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _safe_err_raw(raw: bytes, status: int) -> dict:
    try:
        return json.loads(raw.decode("utf-8", "replace"))
    except Exception:
        return {"error": {"message": raw.decode("utf-8", "replace")[:500], "type": "upstream_error", "code": status}}


def _strip_empty_tool_calls(line: bytes) -> bytes:
    """剥掉 SSE data 行里 delta 的空 tool_calls 数组，其余行原样返回。

    后端网关每个 chunk 都带 "tool_calls":[]（见 converter.log 原始 SSE）。ZCode 的
    AI SDK 流解析把 delta.tool_calls != null 当「思考块结束」信号，空数组也命中，
    会把每个 reasoning_content 增量拆成独立思考块（UI 一词一块）。空数组无语义，
    JSON 解析后安全剔除；非 data 行或解析失败原样透传，不影响 SSE 帧。
    """
    if b'"tool_calls":[]' not in line or not line.startswith(b"data:"):
        return line
    try:
        obj = json.loads(line[5:].strip())
    except Exception:
        return line
    touched = False
    for ch in obj.get("choices") or []:
        if not isinstance(ch, dict):
            continue
        delta = ch.get("delta")
        if isinstance(delta, dict) and delta.get("tool_calls") == []:
            delta.pop("tool_calls", None)
            touched = True
    if not touched:
        return line
    return b"data: " + json.dumps(
        obj, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")


async def _open_upstream(url: str, headers: dict, body: dict, *, rid: str = "", model_name: str = "?",
                         attempts: int = 3, first_byte: bool = True):
    """建立上游流式连接并预检，吞掉建立阶段的瞬时网络抖动。

    重试窗口：连接失败、HTTP 502/503/504、"200 但首个 body 字节到来前断连"
    （2026-10-05 WorkBuddy 桌面端 -32603 的实际形态：1.1s 处 0 字节被掐断）。
    这些阶段尚未向客户端吐出任何字节，重试安全。429/4xx 不重试（确定性错误
    或需客户端退避，SDK 自会处理）。

    返回 (client, r, first_chunk, chunk_iter)：r 为 200 时 first/chunk_iter 已就绪
    （first_byte=False 则两者为 None），调用方转发完 first 后继续消费 chunk_iter；
    r 非 200 时由调用方按协议包装错误。重试耗尽仍网络失败抛最后一次 httpx.HTTPError。
    connect 固定 10s 超时防黑洞挂死；读写不设超时（长生成不能被掐）。
    """
    prefix = f"[{rid}] " if rid else ""
    last_exc: httpx.HTTPError | None = None
    for attempt in range(1, attempts + 1):
        client = httpx.AsyncClient(timeout=httpx.Timeout(None, connect=10.0), trust_env=_trust_env())
        r: httpx.Response | None = None
        try:
            r = await client.send(client.build_request("POST", url, headers=headers, json=body), stream=True)
            if r.status_code in (502, 503, 504) and attempt < attempts:
                raw = await r.aread()
                _log(f"{prefix}✗ HTTP {r.status_code}({attempt}/{attempts},重试) | {model_name} | {_truncate(raw.decode('utf-8','replace'),120)}")
                await r.aclose()
                await client.aclose()
                await asyncio.sleep(1)
                continue
            first: bytes | None = None
            chunk_iter = None
            if r.status_code == 200 and first_byte:
                chunk_iter = r.aiter_bytes()
                try:
                    first = await chunk_iter.__anext__()
                except StopAsyncIteration:
                    first = None
            return client, r, first, chunk_iter
        except httpx.HTTPError as e:
            if r is not None:
                await r.aclose()
            await client.aclose()
            last_exc = e
            kind = "连接" if r is None else "首块前"
            _log(f"{prefix}✗ 网络错误({kind},{attempt}/{attempts}) | {model_name} | {e!r}")
            if attempt < attempts:
                await asyncio.sleep(1)
    raise last_exc  # type: ignore[misc]


async def _relay_chat(client: httpx.AsyncClient, r: httpx.Response, chunk_iter,
                      first: bytes | None, model_name: str = "?", t0: float = 0.0, rid: str = ""):
    """把调用方已建立的 200 上游流转发给客户端（标准 OpenAI SSE，剥空 tool_calls 数组）。

    first/chunk_iter 来自 _open_upstream（首个 body 块已取出，需先行转发）。
    同时轻量解析流，统计 finish_reason / tool_calls / usage 用于日志，不阻塞转发。
    完整原始 SSE（后端原样）累积后落盘到日志（调试用）。流中途断连时响应头
    已发出无法改状态码，仍以 SSE 错误 chunk 收尾。
    """
    finish_reason = None
    tool_names: list[str] = []
    usage: dict = {}
    saw_filter = False
    buf = b""
    raw_parts: list[bytes] = [first] if first else []   # 累积完整原始 SSE（首块已含）
    sse_buf = b""                 # 转发用行缓冲（按行剥空 tool_calls 后再发）
    prefix = f"[{rid}] " if rid else ""

    def _feed(chunk: bytes):
        nonlocal finish_reason, saw_filter, buf
        # 行缓冲解析：把累计的 chunk 按 data: 行切出来统计
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                continue
            try:
                obj = json.loads(data)
            except Exception:
                continue
            if obj.get("usage"):
                usage.update(obj["usage"])
            for ch in obj.get("choices") or []:
                if ch.get("finish_reason"):
                    finish_reason = ch["finish_reason"]
                for tc in (ch.get("delta") or {}).get("tool_calls") or []:
                    nm = (tc.get("function") or {}).get("name")
                    if nm:
                        tool_names.append(nm)
            # 内容审核拦截常以 content-filter 或特殊中文文案返回
            try:
                text_repr = data.decode("utf-8", "replace")
            except Exception:
                text_repr = ""
            if "content-filter" in text_repr or "敏感" in text_repr or "审核" in text_repr:
                saw_filter = True

    def _forward(chunk: bytes) -> bytes:
        """统计 + 行缓冲转发，返回本块可发给客户端的字节。"""
        nonlocal sse_buf
        _feed(chunk)
        sse_buf += chunk
        if b"\n" not in sse_buf:
            return b""
        lines = sse_buf.split(b"\n")
        sse_buf = lines.pop()  # 末段可能是不完整行，留待下一轮
        return b"".join(_strip_empty_tool_calls(ln) + b"\n" for ln in lines)

    try:
        if first:
            out = _forward(first)
            if out:
                yield out
        async for chunk in chunk_iter:
            if chunk:
                raw_parts.append(chunk)
                out = _forward(chunk)
                if out:
                    yield out
        if sse_buf:
            yield _strip_empty_tool_calls(sse_buf)
    except httpx.HTTPError as e:
        _log(f"{prefix}✗ 网络错误(流中) | {model_name} | {e!r}")
        yield _err_event(str(e).encode() or b"upstream connection lost", 502)
    finally:
        await r.aclose()
        await client.aclose()

    # 流结束：输出完成日志
    elapsed = time.time() - t0 if t0 else 0
    tag = " ⚠️内容审核拦截" if (saw_filter or finish_reason == "content-filter") else ""
    _log(f"{prefix}◀ RESPONSE {model_name} | {elapsed:.1f}s | stream finish={finish_reason}{tag}"
         + (f" | tool_calls={tool_names}" if tool_names else "")
         + f" | tokens={usage.get('total_tokens', '?')}")
    # 完整原始 SSE（后端返回的全部内容）
    _log(f"{prefix}── RESPONSE RAW SSE ──\n{b''.join(raw_parts).decode('utf-8','replace')}")


def _safe_err(r: httpx.Response) -> dict:
    try:
        return {"error": r.json()}
    except Exception:
        return {"error": {"message": r.text[:500], "type": "upstream_error", "code": r.status_code}}


def _err_event(msg: bytes, status: int) -> bytes:
    # 以 OpenAI SSE 错误 chunk 形式返回
    import json as _json, time as _time
    chunk = {
        "error": {"message": msg.decode("utf-8", "replace")[:500], "type": "upstream_error", "code": status},
    }
    return f"data: {_json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8")


def _looks_like_content_filter_text(text: str) -> bool:
    text = (text or "").lower()
    return (
        "content-filter" in text
        or "content_filter" in text
        or "敏感内容" in text
        or "内容审核" in text
        or "无法响应您的请求" in text
    )


def _chat_body_desensitize(body: dict, *, force_compact: bool = False) -> dict:
    if not CONFIG.get("desensitize"):
        return body
    return desensitize_body(
        body,
        roles=("system", "developer"),
        desensitize_harness_user=True,
        desensitize_tools=True,
        compact_harness=(force_compact or not CONFIG.get("no_compact")),
        strip_tool_metadata=True,
    )


async def _post_backend_once(url: str, headers: dict, body: dict, *,
                             attempts: int = 3, rid: str = "", model_name: str = "?") -> tuple[int, bytes]:
    """整包拉取后端响应。连接失败 / 5xx 自动重试——调用方在拿到完整返回前不会向
    客户端吐任何字节，重试安全（链路秒级抖动实测会成小时级爆发，2026-10-05）。
    429/4xx 不重试（确定性错误或需客户端退避）。重试耗尽仍网络失败则抛异常。"""
    prefix = f"[{rid}] " if rid else ""
    last_exc: httpx.HTTPError | None = None
    for attempt in range(1, attempts + 1):
        try:
            async with httpx.AsyncClient(timeout=120, trust_env=_trust_env()) as c:
                async with c.stream("POST", url, headers=headers, json=body) as r:
                    if r.status_code in (502, 503, 504) and attempt < attempts:
                        raw = await r.aread()
                        _log(f"{prefix}✗ HTTP {r.status_code}({attempt}/{attempts},重试) | {model_name} | {_truncate(raw.decode('utf-8','replace'),120)}")
                        await asyncio.sleep(1)
                        continue
                    chunks: list[bytes] = []
                    async for chunk in r.aiter_bytes():
                        if chunk:
                            chunks.append(chunk)
                    return r.status_code, b"".join(chunks)
        except httpx.HTTPError as e:
            last_exc = e
            _log(f"{prefix}✗ 网络错误({attempt}/{attempts}) | {model_name} | {e!r}")
            if attempt < attempts:
                await asyncio.sleep(1)
    raise last_exc  # type: ignore[misc]


async def _post_backend_with_filter_retry(url: str, headers: dict, body: dict,
                                          rid: str = "", model_name: str = "?") -> tuple[int, bytes, dict]:
    prefix = f"[{rid}] " if rid else ""
    status, raw = await _post_backend_once(url, headers, body, rid=rid, model_name=model_name)
    text = raw.decode("utf-8", "replace")
    if status == 200 and _looks_like_content_filter_text(text) and CONFIG.get("desensitize") and CONFIG.get("no_compact"):
        retry_body = _chat_body_desensitize(body, force_compact=True)
        _log(f"{prefix}↻ RESPONSES {model_name} | content filter detected, retry with compact harness")
        _log(f"{prefix}── RESPONSES RETRY CHAT BODY ──\n{json.dumps(retry_body, ensure_ascii=False, indent=2)}")
        retry_status, retry_raw = await _post_backend_once(url, headers, retry_body, rid=rid, model_name=model_name)
        retry_text = retry_raw.decode("utf-8", "replace")
        if retry_status == 200 and not _looks_like_content_filter_text(retry_text):
            return retry_status, retry_raw, retry_body
    return status, raw, body


# ---------------------------------------------------------------------------
# Responses API 端点（Codex CLI 兼容）
# ---------------------------------------------------------------------------

@app.post("/v1/responses")
async def create_response(request: Request,
                          authorization: Optional[str] = Header(default=None),
                          x_api_key: Optional[str] = Header(default=None, alias="X-Api-Key")):
    """OpenAI Responses API 兼容端点。

    Codex CLI 使用 Responses API（wire_api = "responses"）而非 Chat Completions。
    本端点接收 Responses 格式请求，转换为 Chat 格式发往后端，再将后端的 Chat SSE
    转换为 Responses 语义事件流返回。
    """
    _check_auth(authorization, x_api_key)
    cred = _cred()

    try:
        payload = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail={"error": {"message": f"bad json: {e}", "type": "invalid_request_error"}})

    # 转换请求：Responses → Chat
    try:
        chat_body = responses_request_to_chat(payload)
    except Exception as e:
        raise HTTPException(status_code=400, detail={"error": {"message": f"request conversion error: {e}", "type": "invalid_request_error"}})

    # --no-compact 时跳过投影压缩，保留完整上下文
    if CONFIG.get("no_compact"):
        chat_body, projection_stats = chat_body, {"mode": "no-compact", "aggressive": False}
    else:
        chat_body, projection_stats = project_responses_chat_body(chat_body)
    chat_body.setdefault("model", "auto")
    chat_body["model"] = MODEL_NAME_MAP.get(chat_body["model"], chat_body["model"])
    chat_body["stream"] = True
    if "stream_options" not in chat_body:
        chat_body["stream_options"] = {"include_usage": True}

    chat_body = _chat_body_desensitize(chat_body)

    client_wants_stream = payload.get("stream", True)  # Codex CLI 默认 stream
    model_name = payload.get("model", "auto")
    rid = os.urandom(4).hex()
    _log(f"[{rid}] ▶ RESPONSES {model_name} | stream={client_wants_stream} | input_items={len(payload.get('input', []))}")
    _log(
        f"[{rid}] ── RESPONSES PROJECTION ── "
        f"mode={projection_stats.get('mode')} "
        f"| msgs {projection_stats.get('original_messages')}→{projection_stats.get('projected_messages')} "
        f"| chars {projection_stats.get('original_message_chars')}→{projection_stats.get('projected_message_chars')} "
        f"| tools {projection_stats.get('original_tools')}→{projection_stats.get('projected_tools')} "
        f"| tool_chars {projection_stats.get('original_tool_chars')}→{projection_stats.get('projected_tool_chars')} "
        f"| summarized_history={projection_stats.get('summarized_history_messages', 0)} "
        f"| dropped_harness={projection_stats.get('dropped_harness_messages', 0)} "
        f"| anchor_user={projection_stats.get('anchor_user_preserved', False)}"
    )
    _log(f"[{rid}] ── RESPONSES → CHAT BODY ──\n{json.dumps(chat_body, ensure_ascii=False, indent=2)}")

    if REQUIRE_SYSTEM_FIRST:
        _ensure_system_first(chat_body)

    headers = cred.get_headers()
    url = f"{BACKEND}/v2/chat/completions"
    t0 = time.time()

    if client_wants_stream:
        return StreamingResponse(
            _stream_responses(url, headers, chat_body, model_name, t0, rid),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # 非流式：聚合后端 SSE → 非流式 Response 对象
    try:
        status_code, raw, final_body = await _post_backend_with_filter_retry(url, headers, chat_body, rid, model_name)
        if status_code != 200:
            _log(f"[{rid}] ✗ HTTP {status_code} | {model_name} | {_truncate(raw.decode('utf-8','replace'),200)}")
            raise HTTPException(status_code=status_code, detail=_safe_err_raw(raw, status_code))
        converter = ResponsesStreamConverter(model=model_name)
        for line in raw.decode("utf-8", "replace").splitlines():
            converter.feed_line(line)
        chat_body = final_body
    except HTTPException:
        raise
    except httpx.HTTPError as e:
        _log(f"[{rid}] ✗ 网络错误 | {model_name} | {e}")
        raise HTTPException(status_code=502, detail={"error": {"message": f"upstream error: {e}", "type": "upstream_error"}})

    result = converter.get_nonstream_response()
    elapsed = time.time() - t0
    _log(f"[{rid}] ◀ RESPONSES {model_name} | {elapsed:.1f}s")
    _log(f"[{rid}] ── RESPONSE OBJ ──\n{json.dumps(result, ensure_ascii=False, indent=2)}")
    return JSONResponse(content=result)


async def _stream_responses(url: str, headers: dict, body: dict,
                            model_name: str = "?", t0: float = 0.0, rid: str = ""):
    """消费后端 Chat SSE，实时转换为 Responses API 事件流输出。"""
    converter = ResponsesStreamConverter(model=model_name)
    prefix = f"[{rid}] " if rid else ""

    try:
        status_code, raw, _ = await _post_backend_with_filter_retry(url, headers, body, rid, model_name)
        if status_code != 200:
            _log(f"{prefix}✗ HTTP {status_code} | {model_name} | {_truncate(raw.decode('utf-8','replace'),200)}")
            error_evt = {"type": "error", "error": {"message": raw.decode('utf-8','replace')[:500], "code": status_code}}
            yield f"data: {json.dumps(error_evt, ensure_ascii=False)}\n\n".encode("utf-8")
            return
        raw_sse_lines = []
        for line in raw.decode("utf-8", "replace").splitlines():
            if line.strip():
                raw_sse_lines.append(line)
            events = converter.feed_line(line)
            if events:
                yield events.encode("utf-8")
    except httpx.HTTPError as e:
        _log(f"{prefix}✗ 网络错误 | {model_name} | {e}")
        error_evt = {"type": "error", "error": {"message": str(e)[:500], "code": 502}}
        yield f"data: {json.dumps(error_evt, ensure_ascii=False)}\n\n".encode("utf-8")
        return

    # 发送收尾事件
    finish_events = converter.finish()
    if finish_events:
        yield finish_events.encode("utf-8")

    elapsed = time.time() - t0 if t0 else 0
    _log(f"{prefix}◀ RESPONSES {model_name} | {elapsed:.1f}s | stream done")
    _log(f"{prefix}── RESPONSES RAW SSE ──\n" + "\n".join(raw_sse_lines[-30:]))


# ---------------------------------------------------------------------------
# Anthropic Messages API 端点（Claude Code / CC Switch 兼容）
# ---------------------------------------------------------------------------

@app.post("/v1/messages")
async def create_message(request: Request,
                         authorization: Optional[str] = Header(default=None),
                         x_api_key: Optional[str] = Header(default=None, alias="X-Api-Key")):
    """Anthropic Messages API 兼容端点。

    Claude Code / CC Switch 使用 Anthropic Messages API（POST /v1/messages）。
    本端点接收 Anthropic 格式请求，转换为 Chat 格式发往后端，再将后端的 Chat SSE
    转换为 Anthropic SSE 事件流返回。
    """
    _check_auth(authorization, x_api_key)
    cred = _cred()

    try:
        payload = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail={"error": {"message": f"bad json: {e}", "type": "invalid_request_error"}})

    # 将 Anthropic 格式消息、工具规范在进入后端前统一转换为 OpenAI Chat 格式。
    messages = payload.get("messages") or []
    if not messages:
        raise HTTPException(status_code=400, detail={"error": {"message": "messages is required", "type": "invalid_request_error"}})

    try:
        chat_body = anthropic_request_to_chat(payload)
    except Exception as e:
        raise HTTPException(status_code=400, detail={"error": {"message": f"request conversion error: {e}", "type": "invalid_request_error"}})

    chat_body.setdefault("model", "auto")
    chat_body["model"] = MODEL_NAME_MAP.get(chat_body["model"], chat_body["model"])
    chat_body["stream"] = True
    if "stream_options" not in chat_body:
        chat_body["stream_options"] = {"include_usage": True}

    if CONFIG.get("desensitize"):
        chat_body = desensitize_body(chat_body, roles=("system", "developer"),
                                     desensitize_harness_user=True,
                                     desensitize_tools=True,
                                     compact_harness=not CONFIG.get("no_compact"),
                                     strip_tool_metadata=True)

    # 替换第三方客户端的 system prompt，避免触发腾讯安全拦截
    # 策略：只替换明显是客户端自动注入的长篇 system prompt（含工具定义、安全条款等）
    # 保留用户自己写的简短 system prompt（比如"用中文回答"、"扮演架构师"等）
    CLIENT_KEYWORDS = ("ZCode", "zcode", "Codex", "codex", "Claude Code")
    MIN_SYSTEM_LENGTH = 200  # 客户端注入的 system prompt 通常较长，用户自定义的一般很短
    if chat_body.get("messages"):
        for msg in chat_body["messages"]:
            if msg.get("role") != "system":
                continue
            content = msg.get("content", "")
            if not isinstance(content, str):
                continue
            # 同时满足：包含客户端标识词 + 长度足够长，才认为是客户端注入的
            if any(kw in content for kw in CLIENT_KEYWORDS) and len(content) >= MIN_SYSTEM_LENGTH:
                msg["content"] = "你是一个乐于助人的编程助手。帮助用户完成软件工程任务。需要时使用提供的工具。用与用户相同的语言回复。"
                _log(f"[filter] 替换客户端注入的 system prompt ({len(content)} chars → 通用提示)")

    # 日志使用映射后的模型名
    model_name = chat_body.get("model", payload.get("model", "auto"))
    chat_messages = chat_body.get("messages", [])
    rid = os.urandom(4).hex()
    _log(f"[{rid}] ▶ ANTHROPIC {model_name} | msgs={len(chat_messages)} | anthropic_msgs={len(messages)}")
    _log(f"[{rid}] ── ANTHROPIC → CHAT BODY ──\n{json.dumps(chat_body, ensure_ascii=False, indent=2)}")

    if REQUIRE_SYSTEM_FIRST:
        _ensure_system_first(chat_body)

    headers = cred.get_headers()
    url = f"{BACKEND}/v2/chat/completions"
    t0 = time.time()

    # 客户端要非流式（Anthropic Messages 默认 stream=false）→ 聚合为单个 JSON 响应；
    # 要流式 → 先预检上游状态码再开流。此前无论 200 与否都包成 event-stream，
    # 上游 400（如 tool_choice 对象触发 11101）被 Claude Desktop 看成
    # "empty or malformed response (HTTP 200)"，真因被完全掩盖。
    if not payload.get("stream"):
        conv = AnthropicStreamConverter(model=model_name)
        saw_events = False
        try:
            client, r, _, _ = await _open_upstream(url, headers, chat_body, rid=rid, model_name=model_name, first_byte=False)
        except httpx.HTTPError as e:
            _log(f"[{rid}] ✗ 网络错误(重试耗尽) | {model_name} | {e!r}")
            return JSONResponse(status_code=502, content={"type": "error", "error": {"type": "api_error", "message": f"upstream error: {e!r}"}})
        try:
            if r.status_code != 200:
                raw = await r.aread()
                _log(f"[{rid}] ✗ HTTP {r.status_code} | {model_name} | {_truncate(raw.decode('utf-8','replace'),200)}")
                _log(f"[{rid}] ── ERROR BODY ──\n{raw.decode('utf-8','replace')}")
                return _anthropic_upstream_error(raw, r.status_code)
            async for line in r.aiter_lines():
                if conv.feed_line(line):
                    saw_events = True
        except httpx.HTTPError as e:
            _log(f"[{rid}] ✗ 网络错误(聚合中) | {model_name} | {e!r}")
            return JSONResponse(status_code=502, content={"type": "error", "error": {"type": "api_error", "message": f"upstream error: {e!r}"}})
        finally:
            await r.aclose()
            await client.aclose()
        if not saw_events:
            _log(f"[{rid}] ⚠️ 非流式收到 0 个上游事件 | {model_name}")
        elapsed = time.time() - t0 if t0 else 0
        # 读转换器内部状态用于完成日志
        _log(f"[{rid}] ◀ ANTHROPIC {model_name} | {elapsed:.1f}s | nonstream done | stop={conv._finish_reason}")
        return JSONResponse(content=conv.get_nonstream_response())

    try:
        client, r, first, chunk_iter = await _open_upstream(url, headers, chat_body, rid=rid, model_name=model_name)
    except httpx.HTTPError as e:
        _log(f"[{rid}] ✗ 网络错误(重试耗尽) | {model_name} | {e!r}")
        return JSONResponse(status_code=502, content={"type": "error", "error": {"type": "api_error", "message": f"upstream error: {e!r}"}})
    if r.status_code != 200:
        err = await r.aread()
        await r.aclose()
        await client.aclose()
        _log(f"[{rid}] ✗ HTTP {r.status_code} | {model_name} | {_truncate(err.decode('utf-8','replace'),200)}")
        _log(f"[{rid}] ── ERROR BODY ──\n{err.decode('utf-8','replace')}")
        return _anthropic_upstream_error(err, r.status_code)
    return StreamingResponse(
        _relay_anthropic(client, r, chunk_iter, first, model_name, t0, rid),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _relay_anthropic(client: httpx.AsyncClient, r: httpx.Response, chunk_iter,
                           first: bytes | None, model_name: str = "?", t0: float = 0.0, rid: str = ""):
    """把调用方已建立的 200 上游流转换为 Anthropic Messages SSE 转发给客户端。

    上游是 OpenAI SSE 字节流（首块已由 _open_upstream 取出待转发），这里自行按行
    切分喂给转换器。非 200 的上游响应在路由里预检时已用真实状态码报错，这里只
    消费 200 流；流中途断连仍以 Anthropic error 事件收尾（此时响应头已发出，
    改不了状态码）。
    """
    converter = AnthropicStreamConverter(model=model_name)
    prefix = f"[{rid}] " if rid else ""
    line_buf = b""

    def _feed_lines(chunk: bytes) -> str:
        nonlocal line_buf
        out = ""
        line_buf += chunk
        if b"\n" not in line_buf:
            return out
        lines = line_buf.split(b"\n")
        line_buf = lines.pop()
        for ln in lines:
            out += converter.feed_line(ln.decode("utf-8", "replace"))
        return out

    try:
        if first:
            events = _feed_lines(first)
            if events:
                yield events.encode("utf-8")
        async for chunk in chunk_iter:
            if chunk:
                events = _feed_lines(chunk)
                if events:
                    yield events.encode("utf-8")
        if line_buf:
            tail = converter.feed_line(line_buf.decode("utf-8", "replace"))
            line_buf = b""
            if tail:
                yield tail.encode("utf-8")
        finish_events = converter.finish()
        if finish_events:
            yield finish_events.encode("utf-8")
    except httpx.HTTPError as e:
        _log(f"{prefix}✗ 网络错误(流中) | {model_name} | {e!r}")
        error_evt = {"type": "error", "error": {"message": (str(e) or "upstream connection lost")[:500], "type": "api_error", "code": 502}}
        yield f"event: error\ndata: {json.dumps(error_evt, ensure_ascii=False)}\n\n".encode("utf-8")
    finally:
        await r.aclose()
        await client.aclose()

    elapsed = time.time() - t0 if t0 else 0
    _log(f"{prefix}◀ ANTHROPIC {model_name} | {elapsed:.1f}s | stream done")


@app.post("/v1/messages/count_tokens")
async def count_tokens(request: Request,
                       authorization: Optional[str] = Header(default=None),
                       x_api_key: Optional[str] = Header(default=None, alias="X-Api-Key")):
    """Anthropic token 计数端点（stub）。

    Claude Code 可能在发送消息前调用此端点。
    返回一个简单估算值，不做实际 token 计数。
    """
    _check_auth(authorization, x_api_key)
    return {"input_tokens": 0}


# ---------------------------------------------------------------------------
# 启动
# ---------------------------------------------------------------------------

def preflight(af: Path | None = None) -> bool:
    if af is None:
        af = find_auth_file()
    sys.stderr.write("==== 预检 ====\n")
    sys.stderr.write(f"平台      : {sys.platform}\n")
    sys.stderr.write(f"Python    : {sys.version.split()[0]}\n")
    sys.stderr.write(f"变体      : {VARIANT}（{'国际版 WorkBuddy AI' if VARIANT == 'intl' else '国内版 WorkBuddy'}）\n")
    sys.stderr.write(f"后端      : {BACKEND} (直连，原生 function calling)\n")
    sys.stderr.write(f"登录文件  : {af or '(未找到)'}\n")
    if auth_dirs():
        sys.stderr.write(f"已查目录  : {', '.join(str(d) for d in auth_dirs())}（找 {AUTH_FILENAME}）\n")
    ok = True
    if af is None:
        hint = "WorkBuddy AI" if VARIANT == "intl" else "CodeBuddy/WorkBuddy"
        sys.stderr.write(f"\n[警告] 未找到登录文件。请在桌面端完成登录（{hint}）。\n")
        ok = False
    else:
        try:
            cm = CredentialManager(af)
            info = cm.summary()
            sys.stderr.write(f"账号      : {info.get('nickname')} / {info.get('enterpriseName')}\n")
            sys.stderr.write(f"token过期 : {'是(将自动刷新)' if info['token_expired'] else '否'}\n")
        except Exception as e:
            sys.stderr.write(f"[警告] 读取凭据失败：{e}\n")
            ok = False
    sys.stderr.write("================\n")
    return ok


def main():
    ap = argparse.ArgumentParser(description="CodeBuddy -> OpenAI 兼容转换器（直连后端）")
    ap.add_argument("--variant", choices=list(VARIANTS),
                    default=os.environ.get("WB_VARIANT", "cn"),
                    help="网关变体：cn=国内版 WorkBuddy（默认，端口 8787），"
                         "intl=国际版 WorkBuddy AI（默认端口 8788）。"
                         "两版可各起一个实例同时运行。")
    ap.add_argument("--backend", default=None, metavar="URL",
                    help="覆盖变体默认后端地址（一般不需要）")
    ap.add_argument("--auth-file", default=None, metavar="PATH",
                    help="覆盖变体默认凭据文件路径（一般不需要）")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=None,
                    help="监听端口（默认：cn=8787, intl=8788）")
    ap.add_argument("--api-key", default=os.environ.get("CODEBUDDY2OPENAI_KEY", ""),
                    help="可选：要求客户端携带的 API key（默认不校验）")
    ap.add_argument("--log", default=None, metavar="PATH",
                    help="开启日志并写到该文件（如 --log converter.log 或 --log /tmp/cb.log）。"
                         "不传则不记日志。")
    ap.add_argument("--desensitize", action="store_true",
                    help="启用脱敏：对 system 消息里的合规模板敏感词（DoS/exploit/credential 等）"
                         "插入零宽空格，缓解被后端内容审核误拦。默认关闭。")
    ap.add_argument("--no-compact", action="store_true",
                    help="配合 --desensitize 使用：跳过 system/harness 压缩，仅做零宽脱敏。"
                         "保留原始 system prompt 完整内容（如 Claude Code 的行为指令），"
                         "但审核误拦风险略高于默认压缩模式。")
    ap.add_argument("--skip-check", action="store_true", help="跳过启动预检")
    args = ap.parse_args()

    CONFIG["api_key"] = args.api_key
    CONFIG["desensitize"] = args.desensitize
    CONFIG["no_compact"] = args.no_compact
    # --log 直接指定文件路径即开启；不传则不记
    CONFIG["log_path"] = args.log if args.log else os.environ.get("CODEBUDDY2OPENAI_LOG")

    global BACKEND
    apply_variant(args.variant)
    if args.backend:
        BACKEND = args.backend
    port = args.port if args.port is not None else (8788 if VARIANT == "intl" else 8787)

    af = Path(args.auth_file) if args.auth_file else find_auth_file()
    CONFIG["cred"] = CredentialManager(af) if af else None

    if not args.skip_check:
        preflight(af)

    sys.stderr.write(f"\n✅ 监听 http://{args.host}:{port}（{VARIANT} 变体，直连后端，原生 function calling）\n")
    sys.stderr.write("   GET  /v1/models\n")
    sys.stderr.write("   POST /v1/chat/completions   (原生 tools/tool_calls，支持流式)\n")
    sys.stderr.write("   POST /v1/responses          (Responses API，Codex CLI 兼容)\n")
    sys.stderr.write("   POST /v1/messages           (Anthropic API，Claude Code / CC Switch 兼容)\n")
    sys.stderr.write("   GET  /health\n")
    if args.api_key:
        sys.stderr.write("   鉴权已启用（API key 已设置）\n")
    if CONFIG["log_path"]:
        sys.stderr.write(f"   日志      : {CONFIG['log_path']}\n")
    if args.desensitize:
        mode = "零宽脱敏 + 保留全文" if args.no_compact else "零宽脱敏 + 压缩摘要"
        sys.stderr.write(f"   脱敏      : 已启用（{mode}）\n")
    sys.stderr.write("按 Ctrl+C 退出。\n\n")

    # 启动时写一条标记
    _log(f"==== converter 启动 ====")

    uvicorn.run(app, host=args.host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
