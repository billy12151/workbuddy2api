#!/usr/bin/env python3
"""check_key.py — 检查本地钥匙是否能解当前 auth 文件。

退出码 0：不需要动作（auth 明文格式，或钥匙文件覆盖所有信封 keyId）。
退出码 1：需要重抓（钥匙文件缺失/不匹配），由 start.sh 自动走 grab 流程。
"""

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from atrest import load_keys  # noqa: E402


def find_auth_file():
    base = os.path.expanduser(
        "~/Library/Application Support/CodeBuddyExtension/Data/Public/auth")
    for f in sorted(glob.glob(os.path.join(base, "*.info"))):
        return f
    return None


def collect_envelope_key_ids(node, out):
    if isinstance(node, dict):
        if node.get("$wbEncrypted") == 1:
            try:
                envelope = json.loads(__import__("base64").b64decode(node["envelope"]))
                out.add(envelope.get("keyId", ""))
            except Exception:
                out.add("?unparsable")
            return
        for v in node.values():
            collect_envelope_key_ids(v, out)
    elif isinstance(node, list):
        for v in node:
            collect_envelope_key_ids(v, out)


def main():
    auth_path = find_auth_file()
    if not auth_path:
        print("no-auth-file")
        return 0
    with open(auth_path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    key_ids = set()
    collect_envelope_key_ids(doc, key_ids)
    if not key_ids:
        print("plaintext-auth")
        return 0
    key_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            os.pardir, "secrets.atrest.json")
    if not os.path.exists(key_file):
        print("key-file-missing")
        return 1
    keys = load_keys()
    missing = key_ids - set(keys)
    if missing:
        print(f"key-mismatch auth={sorted(key_ids)} have={sorted(keys)}")
        return 1
    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
