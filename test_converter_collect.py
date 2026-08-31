#!/usr/bin/env python3
"""
test_converter_collect.py — 验证 /v1/chat/completions 非流式聚合保留 reasoning_content。

直接运行：python3 test_converter_collect.py
"""

import asyncio
import json
import sys
sys.path.insert(0, ".")

from converter import _collect_stream, _strip_empty_tool_calls


class _FakeResponse:
    """带 aiter_lines() 的假 httpx.Response，按行回放 SSE。"""

    def __init__(self, lines: list[str]):
        self._lines = lines

    async def aiter_lines(self):
        for line in self._lines:
            yield line


def _chunk(delta: dict, finish=None, usage=None) -> str:
    obj: dict = {"id": "chatcmpl-1", "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    if usage:
        obj["usage"] = usage
    return "data: " + json.dumps(obj, ensure_ascii=False)


def test_collect_stream_reasoning():
    lines = [
        _chunk({"role": "assistant"}),
        _chunk({"reasoning_content": "思考A"}),
        _chunk({"reasoning_content": "思考B"}),
        _chunk({"content": "正文"}),
        _chunk({}, finish="stop", usage={"prompt_tokens": 3, "completion_tokens": 6, "total_tokens": 9}),
        "data: [DONE]",
    ]
    resp = asyncio.run(_collect_stream(_FakeResponse(lines)))

    msg = resp["choices"][0]["message"]
    assert msg["content"] == "正文"
    assert msg["reasoning_content"] == "思考A思考B"
    assert resp["choices"][0]["finish_reason"] == "stop"
    assert resp["usage"]["total_tokens"] == 9
    print("✅ test_collect_stream_reasoning")


def test_collect_stream_without_reasoning():
    """无思考输出时不带 reasoning_content 字段，老客户端零感知。"""
    lines = [
        _chunk({"role": "assistant"}),
        _chunk({"content": "hi"}),
        _chunk({}, finish="stop"),
        "data: [DONE]",
    ]
    resp = asyncio.run(_collect_stream(_FakeResponse(lines)))
    msg = resp["choices"][0]["message"]
    assert msg["content"] == "hi"
    assert "reasoning_content" not in msg
    print("✅ test_collect_stream_without_reasoning")


def test_strip_empty_tool_calls():
    """ZCode 的 AI SDK 流解析把 delta.tool_calls != null 当「思考块结束」，空数组
    会把 reasoning 拆成一词一块；透传层必须剥掉空数组、保留真实 tool_calls。"""
    # reasoning chunk：剥掉空 tool_calls，reasoning_content 保留
    line = ('data: {"id":"g1","choices":[{"index":0,"delta":{"role":"assistant",'
            '"content":"","reasoning_content":"Let","tool_calls":[]}}]}').encode()
    out = _strip_empty_tool_calls(line)
    obj = json.loads(out[5:])
    assert "tool_calls" not in obj["choices"][0]["delta"]
    assert obj["choices"][0]["delta"]["reasoning_content"] == "Let"
    print("✅ test_strip_empty_tool_calls (reasoning chunk)")


def test_strip_keeps_real_tool_calls():
    """非空 tool_calls（真实工具调用）与 usage/finish_reason 原样保留。"""
    line = ('data: {"id":"g1","choices":[{"index":0,"delta":{"tool_calls":[{"id":"c1",'
            '"type":"function","function":{"name":"Read","arguments":"{}"},"index":0}]},'
            '"finish_reason":"tool_calls"}],"usage":{"total_tokens":9}}').encode()
    out = _strip_empty_tool_calls(line)
    obj = json.loads(out[5:])
    tc = obj["choices"][0]["delta"]["tool_calls"]
    assert tc[0]["function"]["name"] == "Read"
    assert obj["choices"][0]["finish_reason"] == "tool_calls"
    assert obj["usage"]["total_tokens"] == 9
    print("✅ test_strip_keeps_real_tool_calls")


def test_strip_passthrough():
    """无空数组的行 / [DONE] / 非 data 行 / 坏 JSON / tool_calls 为 null，全部原样透传。"""
    cases = [
        b'data: {"choices":[{"delta":{"content":"hi","tool_calls":null}}]}',
        b"data: [DONE]",
        b"",
        b'data: {"choices": broken',
        b'event: ping',
    ]
    for line in cases:
        assert _strip_empty_tool_calls(line) == line
    print("✅ test_strip_passthrough")


if __name__ == "__main__":
    test_collect_stream_reasoning()
    test_collect_stream_without_reasoning()
    test_strip_empty_tool_calls()
    test_strip_keeps_real_tool_calls()
    test_strip_passthrough()
    print(f"\n🎉 All {5} tests passed!")
