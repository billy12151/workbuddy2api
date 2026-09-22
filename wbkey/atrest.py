#!/usr/bin/env python3
"""atrest.py — WorkBuddy 5.6.x auth 文件 $wbEncrypted 字段解密。

信封: {"$wbEncrypted":1,"envelope":"<base64(JSON{suite,keyId,nonce,authTag,ciphertext})>"}
算法: AES-256-GCM，key 由 CredentialManager 传入（来自 wbkey/grab_key.py 抓取的
主进程推送，存 secrets.atrest.json）。AAD 仅由 keyId+suite+framing 决定：
  "WB-AAD\\0" || 0x01 || lp("WBEV1") || lp("sym-v1") || u32(suite) || lp(keyId)
  || 0x02(field) || 0x00 || 0x00
"""

import base64
import hashlib
import json
import os
import struct

KEY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        os.pardir, "secrets.atrest.json")


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
        b"\x02",
        b"\x00",
        b"\x00",
    ])


def load_keys(path=KEY_FILE) -> dict:
    """返回 {keyId: key_bytes}。"""
    with open(path, "r", encoding="utf-8") as f:
        blob = json.load(f)
    keys = {}
    for name in ("symmetricKey", "userKey"):
        sym = blob.get(name) or {}
        key_id = sym.get("keyId")
        key_b64 = sym.get("keyBase64")
        if key_id and key_b64:
            key = base64.b64decode(key_b64)
            if len(key) == 32 and hashlib.sha256(key).hexdigest()[:16] == key_id:
                keys[key_id] = key
    if not keys:
        raise RuntimeError("secrets.atrest.json 中没有可用钥匙，重新运行 wbkey/grab_key.py")
    return keys


def unwrap(value, keys: dict):
    """$wbEncrypted 包装 -> 明文；非包装值原样返回；钥匙缺失抛 RuntimeError。"""
    if not isinstance(value, dict) or value.get("$wbEncrypted") != 1:
        return value
    envelope = json.loads(base64.b64decode(value["envelope"]))
    key_id = envelope.get("keyId", "")
    key = keys.get(key_id)
    if key is None:
        raise RuntimeError(
            f"钥匙 keyId={key_id} 不在本地钥匙文件中（WorkBuddy 可能升级了），"
            "重新运行 wbkey/grab_key.py 抓取")
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    aad = build_field_aad(envelope["keyId"], envelope["suite"])
    nonce = base64.b64decode(envelope["nonce"])
    tag = base64.b64decode(envelope["authTag"])
    ct = base64.b64decode(envelope["ciphertext"])
    return AESGCM(key).decrypt(nonce, ct + tag, aad).decode("utf-8")


def unwrap_doc(doc: dict, keys: dict) -> dict:
    """深度遍历 doc，原地解密所有 $wbEncrypted 字段。"""
    if isinstance(doc, list):
        for i, v in enumerate(doc):
            if isinstance(v, (dict, list)):
                unwrap_doc(v, keys)
        return doc
    if not isinstance(doc, dict):
        return doc
    for k, v in list(doc.items()):
        if isinstance(v, dict):
            if v.get("$wbEncrypted") == 1:
                doc[k] = unwrap(v, keys)
            else:
                unwrap_doc(v, keys)
        elif isinstance(v, list):
            unwrap_doc(v, keys)
    return doc
