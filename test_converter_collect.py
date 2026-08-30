#!/usr/bin/env python3
"""
test_converter_collect.py — 验证 /v1/chat/completions 非流式聚合保留 reasoning_content。

直接运行：python3 test_converter_collect.py
"""

import asyncio
import json
import sys
sys.path.insert(0, ".")

from converter import _collect_stream


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


if __name__ == "__main__":
    test_collect_stream_reasoning()
    test_collect_stream_without_reasoning()
    print(f"\n🎉 All {2} tests passed!")
