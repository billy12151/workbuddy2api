"""
responses_projection — /v1/responses 的后端投影层。

目标
----
Codex CLI 会把大量运行时提示、完整工具 schema、长历史、以及工具输出一并塞进
/v1/responses 请求里。腾讯后端对这类 agentic payload 很容易触发内容审核，或者
因为上下文过长而表现不稳定。

本模块在保持外部 OpenAI Responses 兼容的前提下，只对发往后端的 Chat body 做
"最小语义闭包"投影：

- 固定短 system 摘要替换 Codex/Claude Code harness
- 保留最新用户意图
- 保留最近一段真实 assistant/tool 链路
- 把更早历史压缩成规则摘要
- 把 tool schema 收敛成结构字段
- 把超长 tool output / tool arguments 压缩成可继续推理的摘要
"""

from __future__ import annotations

import json
import re
from typing import Any


AGENTIC_TOOL_NAMES = {
    "exec_command",
    "write_stdin",
    "update_plan",
    "request_user_input",
    "view_image",
    "get_goal",
    "create_goal",
    "update_goal",
    "apply_patch",
    "tool_search_tool",
}

HARNESS_USER_MARKERS = (
    "# AGENTS.md instructions",
    "<environment_context>",
    "<permissions instructions>",
    "<collaboration_mode>",
    "<skills_instructions>",
    "<system-reminder>",
    "# claudeMd",
)


def _strip_harness_blocks(text: str) -> str:
    """剥离 user 消息里成对的 harness 标签块，返回剩余的真实用户文本。

    与 desensitize 模块的同名逻辑保持一致；此处内联以维持本模块零依赖。
    """
    if not text:
        return ""
    out = text
    for tag in (
        "system-reminder",
        "environment_context",
        "permissions instructions",
        "collaboration_mode",
        "skills_instructions",
        "plugins_instructions",
    ):
        out = re.sub(
            r"\s*" + re.escape(f"<{tag}>") + r".*?" + re.escape(f"</{tag}>") + r"\s*",
            "\n",
            out,
            flags=re.DOTALL,
        )
    stripped = out.strip()
    for prefix in ("# AGENTS.md instructions", "# claudeMd"):
        if stripped.startswith(prefix):
            body = stripped[len(prefix):].lstrip("\n").strip()
            # 用户自己写的指引：保留正文（截断防膨胀），只剥标题行
            if len(body) > 8000:
                body = body[:8000].rstrip() + " …"
            return body
    return stripped

HARNESS_SYSTEM_MARKERS = (
    "You are a coding agent running in the Codex CLI",
    "Within this context, Codex refers to",
    "# AGENTS.md spec",
    "<permissions instructions>",
    "<collaboration_mode>",
    "<skills_instructions>",
    "The following deferred tools are now available via ToolSearch.",
    "### Available skills",
    "## request_user_input availability",
    "You are Claude Code",
)

BASE_SYSTEM_PROMPT = (
    "你是一个乐于助人的编程助手。帮助用户完成软件工程任务。"
    "需要时使用提供的工具。"
    "用与用户相同的语言回复。"
)

HISTORY_PREFIX = "Earlier conversation summary (condensed):"

# 预算参考：Hy3 上下文 256K token。tail 预算 120K 字符 + 历史摘要 24K + system，
# 合计约 150K 字符（≈ 40-60K token），远低于窗口上限，同时让尾部工作集基本完整。
MAX_SYSTEM_GUIDANCE_CHARS = 4000
MAX_USER_CHARS = 20000
MAX_ASSISTANT_CHARS = 20000
MAX_TOOL_OUTPUT_CHARS = 50000
MAX_TOOL_ARGS_CHARS = 4000
MAX_HISTORY_SUMMARY_CHARS = 24000
MAX_HISTORY_ITEMS = 60
MAX_TAIL_MESSAGES = 60
MAX_TAIL_CHARS = 120000

SCHEMA_KEEP_KEYS = {
    "type",
    "properties",
    "required",
    "items",
    "enum",
    "oneOf",
    "anyOf",
    "allOf",
    "additionalProperties",
    "format",
    "minimum",
    "maximum",
    "minItems",
    "maxItems",
    "minLength",
    "maxLength",
    "nullable",
    "description",  # 参数级一句说明，保留给模型；脱敏层会做零宽处理
    "title",
}


def project_responses_chat_body(body: dict) -> tuple[dict, dict]:
    """把 Responses 转出来的 Chat body 投影成更适合腾讯后端的最小上下文。"""
    projected = dict(body)
    messages = list(body.get("messages") or [])
    tools = list(body.get("tools") or [])

    projected_tools, tool_stats = _project_tools(tools)
    if projected_tools:
        projected["tools"] = projected_tools
    elif "tools" in projected:
        projected["tools"] = []

    aggressive = _looks_like_agentic_cli(messages, tools)
    if not aggressive:
        projected["messages"] = _project_messages_conservative(messages)
        return projected, {
            "mode": "conservative",
            "aggressive": False,
            "original_messages": len(messages),
            "projected_messages": len(projected["messages"]),
            "original_message_chars": _messages_size(messages),
            "projected_message_chars": _messages_size(projected["messages"]),
            **tool_stats,
        }

    tool_name_by_call_id = _build_tool_call_name_map(messages)
    preserved_guidance: list[str] = []
    conversation: list[dict] = []
    dropped_harness_messages = 0

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        text = _content_to_text(msg.get("content", ""))

        if role == "system":
            if _looks_like_harness_system(text):
                dropped_harness_messages += 1
                continue
            guidance = _truncate_text(text, MAX_SYSTEM_GUIDANCE_CHARS)
            if guidance:
                preserved_guidance.append(guidance)
            continue

        if role == "user" and _looks_like_harness_user(text):
            # 混合消息（真实提问 + harness 注入）：只剥离标签块、保留提问；
            # 剥离后为空才是纯 harness 消息，才整条丢弃
            kept_text = _strip_harness_blocks(text)
            if not kept_text:
                dropped_harness_messages += 1
                continue
            conversation.append({"role": "user", "content": kept_text})
            continue

        projected_msg = _project_conversation_message(msg)
        if projected_msg is not None:
            conversation.append(projected_msg)

    if not conversation:
        conversation = _project_messages_conservative(messages)

    tail_start = _choose_tail_start(conversation)
    tail_start = _expand_tail_for_tool_context(conversation, tail_start)
    latest_user_idx = _latest_user_index(conversation)

    anchor_user = None
    if latest_user_idx is not None and latest_user_idx < tail_start:
        anchor_user = dict(conversation[latest_user_idx])

    omitted: list[dict] = []
    for idx, msg in enumerate(conversation):
        if idx >= tail_start:
            break
        if latest_user_idx is not None and idx == latest_user_idx and anchor_user is not None:
            continue
        omitted.append(msg)

    final_messages: list[dict] = [{"role": "system", "content": BASE_SYSTEM_PROMPT}]
    guidance_message = _merge_guidance_messages(preserved_guidance)
    if guidance_message:
        final_messages.append({"role": "system", "content": guidance_message})

    history_summary = _build_history_summary(omitted, tool_name_by_call_id)
    if history_summary:
        final_messages.append({"role": "system", "content": history_summary})

    if anchor_user is not None:
        final_messages.append(anchor_user)

    final_messages.extend(conversation[tail_start:])
    projected["messages"] = final_messages

    return projected, {
        "mode": "aggressive",
        "aggressive": True,
        "dropped_harness_messages": dropped_harness_messages,
        "preserved_guidance_messages": len(preserved_guidance),
        "summarized_history_messages": len(omitted),
        "anchor_user_preserved": anchor_user is not None,
        "tail_messages": len(conversation[tail_start:]),
        "original_messages": len(messages),
        "projected_messages": len(final_messages),
        "original_message_chars": _messages_size(messages),
        "projected_message_chars": _messages_size(final_messages),
        **tool_stats,
    }


def _looks_like_agentic_cli(messages: list[dict], tools: list[dict]) -> bool:
    tool_names = {
        _tool_name(tool)
        for tool in tools
        if _tool_name(tool)
    }
    if tool_names & AGENTIC_TOOL_NAMES:
        return True

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        text = _content_to_text(msg.get("content", ""))
        if _looks_like_harness_user(text) or _looks_like_harness_system(text):
            return True
    return False


def _project_messages_conservative(messages: list[dict]) -> list[dict]:
    out: list[dict] = []
    for msg in messages:
        projected = _project_conversation_message(msg, conservative=True)
        if projected is not None:
            out.append(projected)
    return out


def _project_conversation_message(msg: dict, conservative: bool = False) -> dict | None:
    if not isinstance(msg, dict):
        return None

    role = msg.get("role")
    out = dict(msg)

    if role == "system":
        text = _content_to_text(msg.get("content", ""))
        out["content"] = _truncate_text(text, MAX_SYSTEM_GUIDANCE_CHARS)
        return out

    if role == "user":
        text = _content_to_text(msg.get("content", ""))
        out["content"] = _truncate_text(text, MAX_USER_CHARS)
        return out

    if role == "assistant":
        text = _content_to_text(msg.get("content", ""))
        out["content"] = _summarize_free_text(text, MAX_ASSISTANT_CHARS)
        tool_calls = []
        for tool_call in msg.get("tool_calls") or []:
            projected_call = _project_tool_call(tool_call)
            if projected_call is not None:
                tool_calls.append(projected_call)
        if tool_calls:
            out["tool_calls"] = tool_calls
        elif "tool_calls" in out:
            out.pop("tool_calls", None)
        return out

    if role == "tool":
        out["content"] = _summarize_tool_output(_content_to_text(msg.get("content", "")))
        return out

    if conservative:
        text = _content_to_text(msg.get("content", ""))
        out["content"] = _truncate_text(text, MAX_ASSISTANT_CHARS)
        return out

    return None


def _project_tool_call(tool_call: dict) -> dict | None:
    if not isinstance(tool_call, dict):
        return None

    function = tool_call.get("function") or {}
    name = function.get("name", "")
    arguments = function.get("arguments", "")

    return {
        "id": tool_call.get("id"),
        "type": tool_call.get("type", "function"),
        "function": {
            "name": name,
            "arguments": _summarize_tool_arguments(name, arguments),
        },
    }


def _summarize_tool_arguments(name: str, arguments: Any) -> str:
    if not isinstance(arguments, str):
        try:
            return json.dumps(arguments, ensure_ascii=False)
        except Exception:
            return json.dumps({"summary": _truncate_text(str(arguments), 240)}, ensure_ascii=False)

    if len(arguments) <= MAX_TOOL_ARGS_CHARS:
        return arguments

    if name == "apply_patch":
        return json.dumps(
            {"summary": "Large apply_patch payload omitted; a patch was prepared or applied in a previous step."},
            ensure_ascii=False,
        )

    try:
        parsed = json.loads(arguments)
    except Exception:
        return json.dumps({"summary": _truncate_text(arguments, 320)}, ensure_ascii=False)

    return json.dumps(_shrink_json_value(parsed), ensure_ascii=False)


def _shrink_json_value(value: Any, depth: int = 0, key: str = "") -> Any:
    if depth >= 4:
        return "<omitted>"

    if isinstance(value, dict):
        out = {}
        items = list(value.items())
        for idx, (item_key, item_value) in enumerate(items):
            if idx >= 12:
                out["_omitted_keys"] = len(items) - idx
                break
            out[item_key] = _shrink_json_value(item_value, depth + 1, item_key)
        return out

    if isinstance(value, list):
        trimmed = [_shrink_json_value(item, depth + 1, key) for item in value[:6]]
        if len(value) > 6:
            trimmed.append(f"<omitted {len(value) - 6} items>")
        return trimmed

    if isinstance(value, str):
        limit = 240 if key in {"cmd", "chars", "patch", "content", "text", "question"} else 120
        return _truncate_text(value, limit)

    return value


def _project_tools(tools: list[dict]) -> tuple[list[dict], dict]:
    projected = []
    original_chars = _tools_size(tools)

    for tool in tools:
        if not isinstance(tool, dict):
            continue

        if tool.get("type") != "function":
            continue

        function = tool.get("function") or tool
        name = function.get("name")
        if not name:
            continue

        projected_function: dict[str, Any] = {"name": name}
        # 保留一句功能指引，模型才知道每个工具是干什么的；后续脱敏层会再做零宽处理
        desc = function.get("description")
        if isinstance(desc, str) and desc.strip():
            projected_function["description"] = desc.strip().split("\n", 1)[0][:240]
        if "parameters" in function:
            projected_function["parameters"] = _project_schema(function.get("parameters"))
        if "strict" in function:
            projected_function["strict"] = function.get("strict")

        projected.append({"type": "function", "function": projected_function})

    return projected, {
        "original_tools": len(tools),
        "projected_tools": len(projected),
        "original_tool_chars": original_chars,
        "projected_tool_chars": _tools_size(projected),
    }


def _project_schema(schema: Any, depth: int = 0) -> Any:
    if depth >= 6:
        return {"type": "object"}

    if isinstance(schema, dict):
        out: dict[str, Any] = {}
        for key, value in schema.items():
            if key not in SCHEMA_KEEP_KEYS:
                continue
            if key == "properties" and isinstance(value, dict):
                out["properties"] = {
                    prop: _project_schema(prop_schema, depth + 1)
                    for prop, prop_schema in value.items()
                }
            elif key == "items":
                out["items"] = _project_schema(value, depth + 1)
            elif key in {"oneOf", "anyOf", "allOf"} and isinstance(value, list):
                out[key] = [_project_schema(item, depth + 1) for item in value[:6]]
            elif key == "additionalProperties" and isinstance(value, dict):
                out[key] = _project_schema(value, depth + 1)
            else:
                out[key] = value
        return out or {"type": "object"}

    if isinstance(schema, list):
        return [_project_schema(item, depth + 1) for item in schema[:6]]

    return schema


def _choose_tail_start(messages: list[dict]) -> int:
    if not messages:
        return 0

    start = len(messages) - 1
    total_chars = 0
    kept = 0

    for idx in range(len(messages) - 1, -1, -1):
        cost = _message_cost(messages[idx])
        if kept > 0 and (kept >= MAX_TAIL_MESSAGES or total_chars + cost > MAX_TAIL_CHARS):
            break
        start = idx
        total_chars += cost
        kept += 1
    return start


def _expand_tail_for_tool_context(messages: list[dict], start: int) -> int:
    if start <= 0 or not messages:
        return start

    needed_call_ids = {
        msg.get("tool_call_id")
        for msg in messages[start:]
        if isinstance(msg, dict) and msg.get("role") == "tool" and msg.get("tool_call_id")
    }
    if not needed_call_ids:
        return start

    expanded = start
    for idx in range(start - 1, -1, -1):
        msg = messages[idx]
        if msg.get("role") != "assistant":
            continue
        call_ids = {
            tool_call.get("id")
            for tool_call in msg.get("tool_calls") or []
            if isinstance(tool_call, dict)
        }
        if call_ids & needed_call_ids:
            expanded = idx
            needed_call_ids -= call_ids
            if not needed_call_ids:
                break
    return expanded


def _latest_user_index(messages: list[dict]) -> int | None:
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].get("role") == "user":
            return idx
    return None


def _build_history_summary(messages: list[dict], tool_name_by_call_id: dict[str, str]) -> str:
    lines: list[str] = []
    total_chars = 0
    summarized = 0

    # 从最近的被省略历史往回收：tail 之前的那几轮对话优先进入摘要，
    # 而不是最早的 60 条（否则模型对"前几轮说了什么"毫无记忆）。
    for msg in reversed(messages):
        line = _history_line(msg, tool_name_by_call_id)
        if not line:
            continue
        if summarized >= MAX_HISTORY_ITEMS or total_chars + len(line) > MAX_HISTORY_SUMMARY_CHARS:
            break
        lines.append(f"- {line}")
        total_chars += len(line)
        summarized += 1
    lines.reverse()

    remaining = len(messages) - summarized
    if remaining > 0:
        lines.append(f"- {remaining} earlier messages or tool results were further condensed.")

    if not lines:
        return ""
    return HISTORY_PREFIX + "\n" + "\n".join(lines)


def _history_line(msg: dict, tool_name_by_call_id: dict[str, str]) -> str:
    role = msg.get("role")
    text = _content_to_text(msg.get("content", ""))

    if role == "user":
        return f"User asked: {_truncate_text(text, 220)}"

    if role == "assistant":
        tool_names = [
            (tool_call.get("function") or {}).get("name")
            for tool_call in msg.get("tool_calls") or []
            if isinstance(tool_call, dict)
        ]
        tool_names = [name for name in tool_names if name]
        if text and tool_names:
            return f"Assistant replied: {_truncate_text(text, 160)} Then called tools: {', '.join(tool_names[:4])}."
        if tool_names:
            return f"Assistant called tools: {', '.join(tool_names[:4])}."
        if text:
            return f"Assistant replied: {_truncate_text(text, 180)}"
        return ""

    if role == "tool":
        tool_name = tool_name_by_call_id.get(msg.get("tool_call_id", ""), "tool")
        summary = _tool_output_inline_summary(text)
        return f"Tool {tool_name} returned: {summary}"

    if role == "system":
        return f"System guidance: {_truncate_text(text, 180)}"

    return ""


def _build_tool_call_name_map(messages: list[dict]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        for tool_call in msg.get("tool_calls") or []:
            if not isinstance(tool_call, dict):
                continue
            call_id = tool_call.get("id")
            name = (tool_call.get("function") or {}).get("name")
            if call_id and name:
                mapping[call_id] = name
    return mapping


def _merge_guidance_messages(messages: list[str]) -> str:
    merged: list[str] = []
    total = 0
    for message in messages[:2]:
        text = message.strip()
        if not text:
            continue
        if total + len(text) > MAX_SYSTEM_GUIDANCE_CHARS:
            text = _truncate_text(text, MAX_SYSTEM_GUIDANCE_CHARS - total)
        merged.append(text)
        total += len(text)
        if total >= MAX_SYSTEM_GUIDANCE_CHARS:
            break
    if not merged:
        return ""
    if len(merged) == 1:
        return merged[0]
    return "Additional instructions:\n" + "\n\n".join(merged)


def _summarize_tool_output(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    # 尾部工作集的文件/命令输出：阈值内必须原样保留，不能按行做头尾摘要，
    # 否则模型读到的是残缺文件（曾导致 Codex 读文件只剩头 10 行尾 6 行）。
    if len(text) <= MAX_TOOL_OUTPUT_CHARS:
        return text

    # 超长输出：剔除运行时噪声行后按字符保留头尾（约 2/3 头 + 1/3 尾），
    # 中段标注省略量，模型仍能看清结构并用 offset 续读。
    lines = text.splitlines()
    exit_line = next((line.strip() for line in lines if "Process exited with code" in line), "")
    noise_prefixes = (
        "Chunk ID:",
        "Wall time:",
        "Original token count:",
        "Process exited with code",
    )
    body = "\n".join(line for line in lines if not line.startswith(noise_prefixes))
    head_keep = MAX_TOOL_OUTPUT_CHARS * 2 // 3
    tail_keep = MAX_TOOL_OUTPUT_CHARS // 3
    head = body[:head_keep].rstrip()
    tail = body[-tail_keep:].lstrip() if tail_keep else ""
    omitted = len(body) - len(head) - len(tail)

    parts: list[str] = []
    if exit_line:
        parts.append(exit_line)
    parts.append(head)
    if tail and omitted > 0:
        parts.append(f"... [{omitted} chars omitted] ...")
        parts.append(tail)
    return "\n".join(parts)


def _tool_output_inline_summary(text: str) -> str:
    summarized = _summarize_tool_output(text)
    summarized = summarized.replace("\n", " | ")
    return _truncate_text(summarized, 220)


def _summarize_free_text(text: str, limit: int) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    if len(text) <= limit:
        return text

    head = text[: limit // 2].rstrip()
    tail = text[-(limit // 3):].lstrip()
    omitted = len(text) - len(head) - len(tail)
    return f"{head}\n... [{omitted} chars omitted] ...\n{tail}"


def _truncate_text(text: str, limit: int) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[: max(limit - 24, 0)].rstrip() + f" ... [truncated {len(text) - max(limit - 24, 0)} chars]"


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if "text" in block:
                    parts.append(str(block.get("text", "")))
                elif "output" in block:
                    parts.append(str(block.get("output", "")))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content)


def _looks_like_harness_user(text: str) -> bool:
    return any(marker in text for marker in HARNESS_USER_MARKERS)


def _looks_like_harness_system(text: str) -> bool:
    return any(marker in text for marker in HARNESS_SYSTEM_MARKERS)


def _message_cost(msg: dict) -> int:
    cost = len(_content_to_text(msg.get("content", "")))
    for tool_call in msg.get("tool_calls") or []:
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function") or {}
        cost += len(function.get("name", ""))
        cost += len(function.get("arguments", ""))
    return cost


def _messages_size(messages: list[dict]) -> int:
    total = 0
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        total += _message_cost(msg)
        total += len(msg.get("role", ""))
    return total


def _tool_name(tool: dict) -> str:
    if not isinstance(tool, dict):
        return ""
    function = tool.get("function") or tool
    return str(function.get("name", "") or "")


def _tools_size(tools: list[dict]) -> int:
    try:
        return len(json.dumps(tools, ensure_ascii=False))
    except Exception:
        return 0
