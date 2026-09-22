#!/usr/bin/env python3
"""decrypt_fields.py — 解密 WorkBuddy 5.6.x auth 文件里的 $wbEncrypted 字段。

envelope = base64(JSON{suite,keyId,nonce,authTag,ciphertext})，AES-256-GCM。
AAD（sym-v1, field framing）：
  "WB-AAD\\0" || 0x01 || lp("WBEV1") || lp("sym-v1") || u32(suite) || lp(keyId)
  || 0x02(field) || 0x00(无 sequence) || 0x00(final 未定义)

用法:
  python3 decrypt_fields.py          # 自检：解密 auth 文件并打印字段名/前缀
  作为模块: unwrap(value, key, key_id) -> str
"""

import base64
import hashlib
import json
import os
import struct


def _lp(b: bytes) -> bytes:
    return struct.pack(">I", len(b)) + b


def build_field_aad(key_id: str, suite: int) -> bytes:
    return b"".join([
        b"WB-AAD\x00",
        b"\x01",
        _lp(b"WBEV1"),
        _lp(b"sym-v1"),
        struct.pack(">I", suite),
        _lp(key_id.encode()),
        b"\x02",      # FRAMING_CODE field
        b"\x00",      # 无 sequence
        b"\x00",      # final 未定义
    ])


def unwrap(value, key: bytes, key_id: str):
    """$wbEncrypted 包装 -> 明文字符串；非包装值原样返回。"""
    if not isinstance(value, dict) or value.get("$wbEncrypted") != 1:
        return value
    envelope = json.loads(base64.b64decode(value["envelope"]))
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    aad = build_field_aad(envelope["keyId"], envelope["suite"])
    nonce = base64.b64decode(envelope["nonce"])
    tag = base64.b64decode(envelope["authTag"])
    ct = base64.b64decode(envelope["ciphertext"])
    plain = AESGCM(key).decrypt(nonce, ct + tag, aad)
    return plain.decode("utf-8")


def load_key(path=None):
    path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                os.pardir, "secrets.atrest.json")
    blob = json.load(open(path))
    sym = blob["symmetricKey"]
    key_id = sym["keyId"]
    if isinstance(sym.get("keyBase64"), str):
        key = base64.b64decode(sym["keyBase64"])
    else:
        key = bytes(sym["key"]["data"])
    assert len(key) == 32 and hashlib.sha256(key).hexdigest()[:16] == key_id, "key/keyId mismatch"
    return key, key_id


def main():
    key, key_id = load_key()
    auth_path = os.path.expanduser(
        "~/Library/Application Support/CodeBuddyExtension/Data/Public/auth/workbuddy-desktop.info")
    doc = json.load(open(auth_path))

    def walk(node, path=""):
        if isinstance(node, dict):
            if node.get("$wbEncrypted") == 1:
                try:
                    plain = unwrap(node, key, key_id)
                    print(f"OK   {path} -> {plain[:24]}...(len={len(plain)})")
                except Exception as e:
                    print(f"FAIL {path}: {e}")
                return
            for k, v in node.items():
                walk(v, f"{path}.{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")

    walk(doc)


if __name__ == "__main__":
    main()
