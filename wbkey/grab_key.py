#!/usr/bin/env python3
"""grab_key.py — 接住 WorkBuddy 主进程推给 sidecar 的 at-rest 钥匙。

原理（WorkBuddy 5.6.x）：app 每次创建会话会 spawn 一个 CLI sidecar，并在它 env 里
放 CODEBUDDY_SIDECAR_CREDENTIAL_BOOTSTRAP_SOCKET=<unix socket 路径>，然后主动连上
该路径推送 {"version":1,sessionId,token,pid,bootstrap:{policy,symmetricKey,...}}
（ENOTCONN/ECONNREFUSED 每 125ms 重试共 60s），等 {"ok":true,"policy":...} 的 ack。

本工具轮询新进程的 env，抢先 bind 同一路径，app 的重试循环就会把钥匙推送过来。
代价：那一个 sidecar 拿不到钥匙，对应会话启动失败（一次性代价）。

用法: python3 grab_key.py   （抓到即退出；钥匙写 secrets.atrest.json，chmod 600）
"""

import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(BASE, os.pardir, "secrets.atrest.json")
LOG_PATH = os.path.join(BASE, "grab_key.log")
ENV_VAR = "CODEBUDDY_SIDECAR_CREDENTIAL_BOOTSTRAP_SOCKET="
POLL_SEC = 0.025
ACCEPT_TIMEOUT = 75          # app 推送重试窗口 60s，留余量
NEXT_UNIQ_PATTERN = re.compile(r" [A-Z_][A-Z0-9_]*=")


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_PATH, "a") as f:
            f.write(line + "\n")
    except OSError:
        pass


def list_pids():
    out = subprocess.run(["ps", "-axo", "pid="], capture_output=True, text=True).stdout
    return {int(tok) for tok in out.split() if tok.isdigit()}


def read_env_socket_path(pid):
    """返回该进程 env 里的 bootstrap socket 路径，没有则 None。"""
    try:
        r = subprocess.run(["ps", "-Eww", "-o", "command=", "-p", str(pid)],
                           capture_output=True, text=True, timeout=3)
    except (subprocess.TimeoutExpired, OSError):
        return None
    if r.returncode != 0:
        return None
    s = r.stdout
    i = s.find(ENV_VAR)
    if i < 0:
        return None
    rest = s[i + len(ENV_VAR):]
    m = NEXT_UNIQ_PATTERN.search(rest)
    return rest[:m.start()] if m else rest.strip()


def key_to_bytes(value):
    """key 可能是 [int,...] / {"type":"Buffer","data":[...]} / hex / base64。"""
    if isinstance(value, list):
        return bytes(int(x) for x in value)
    if isinstance(value, dict) and isinstance(value.get("data"), list):
        return bytes(int(x) for x in value["data"])
    if isinstance(value, str):
        if len(value) == 64:
            try:
                return bytes.fromhex(value)
            except ValueError:
                pass
        import base64
        try:
            raw = base64.b64decode(value, validate=True)
            if len(raw) == 32:
                return raw
        except Exception:
            pass
    return None


def capture(path):
    """在该路径上抢先监听并接收推送。成功返回 bootstrap dict。"""
    # 已有活监听说明子进程先绑了，放弃这一轮
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    probe.settimeout(0.15)
    try:
        probe.connect(path)
        probe.close()
        return None
    except OSError:
        probe.close()
    try:
        os.unlink(path)          # 探测无人应答，清掉陈留文件
    except OSError:
        pass
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(path)
    srv.listen(1)
    srv.settimeout(ACCEPT_TIMEOUT)
    log(f"listening on {path}")
    try:
        conn, _ = srv.accept()
        conn.settimeout(10)
        buf = b""
        while b"\n" not in buf:
            chunk = conn.recv(65536)
            if not chunk:
                break
            buf += chunk
        if not buf.strip():
            log("connection closed without payload")
            return None
        raw_line = buf.split(b"\n", 1)[0].decode()
        # 无论后续校验成败，原始报文先落盘（调试 + 防丢）
        with open(os.path.join(BASE, "last_push.json"), "w") as f:
            f.write(raw_line)
        payload = json.loads(raw_line)
        bootstrap = payload.get("bootstrap") or {}
        policy = bootstrap.get("policy")
        sym = bootstrap.get("symmetricKey") or {}
        key_id = sym.get("keyId") or ""
        # 线格式: symmetricKey:{keyId, keyBase64}；兼容旧的嵌套/数组形态
        key = key_to_bytes(sym.get("keyBase64"))
        if key is None:
            key = key_to_bytes(sym.get("key"))
        if key is None or len(key) != 32:
            log(f"payload has no usable symmetricKey: policy={policy} "
                f"sym_keys={sorted(sym.keys()) if isinstance(sym, dict) else type(sym).__name__}")
            return None
        derived = hashlib.sha256(key).hexdigest()[:16]
        if derived != key_id:
            log(f"keyId mismatch: derived={derived} declared={key_id}")
            return None
        log(f"got key keyId={key_id} policy={policy} "
            f"userKey={'yes' if bootstrap.get('userKey') else 'no'}")
        try:
            conn.sendall((json.dumps({"ok": True, "policy": policy}) + "\n").encode())
        except OSError:
            pass
        conn.close()
        return bootstrap
    finally:
        srv.close()
        try:
            os.unlink(path)
        except OSError:
            pass


def main():
    os.makedirs(BASE, exist_ok=True)
    if os.path.exists(OUT_PATH):
        log(f"key file already exists: {OUT_PATH}，退出")
        return
    seen = list_pids()
    log(f"watching for new sidecar processes (baseline {len(seen)} pids)...")
    while True:
        time.sleep(POLL_SEC)
        pids = list_pids()
        fresh = pids - seen
        seen = pids
        for pid in fresh:
            path = read_env_socket_path(pid)
            if not path:
                continue
            log(f"pid {pid} carries bootstrap socket: {path}")
            try:
                bootstrap = capture(path)
            except Exception as e:
                log(f"capture error on {path}: {e!r}")
                continue
            if bootstrap:
                tmp = OUT_PATH + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(bootstrap, f, ensure_ascii=False, indent=1)
                os.chmod(tmp, 0o600)
                os.replace(tmp, OUT_PATH)
                log(f"saved -> {OUT_PATH}")
                return


if __name__ == "__main__":
    main()
