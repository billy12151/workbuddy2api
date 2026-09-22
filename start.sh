#!/bin/bash
# workbuddy2api 启动脚本
# 用法: bash start.sh

cd "$(dirname "$0")"

# 杀掉旧进程
pkill -f "converter.py" 2>/dev/null
sleep 1

# 启动
source .venv/bin/activate
# 显式关闭代理鉴权：converter 的 --api-key 默认会取环境变量 CODEBUDDY2OPENAI_KEY
# （~/.zshrc 有 export），导致"从终端启动就要求 key=any-value、从其他环境启动不校验"，
# ZCode/Codex 一侧 key 不匹配就 401。本机回环监听，统一不校验，行为恒定。

# ---------------------------------------------------------------------------
# 钥匙自检（WorkBuddy 5.6.2+ auth 字段加密）：缺失/不匹配时自动重抓。
# 原理：WorkBuddy 冷启动会话时会把解密钥匙推给 sidecar 的 unix socket，
# grab_key.py 抢绑该 socket 接住推送。冷启动只在 prewarm 池空时发生，
# 所以先清池再触发一个任务会话。抓到前先启动服务也没关系——converter
# 支持钥匙热重载，后台抓到后下一个请求自动生效。
# ---------------------------------------------------------------------------
KEY_CHECK=$(.venv/bin/python3 wbkey/check_key.py 2>/dev/null)
if [ "$KEY_CHECK" != "ok" ] && [ "$KEY_CHECK" != "plaintext-auth" ] && [ "$KEY_CHECK" != "no-auth-file" ]; then
    echo "🔑 钥匙缺失或不匹配（$KEY_CHECK），自动重抓..."
    rm -f secrets.atrest.json
    pkill -f "grab_key.py" 2>/dev/null
    pkill -f "codebuddy --prewarm" 2>/dev/null
    .venv/bin/python3 wbkey/grab_key.py > /dev/null 2>&1 &
    GRAB_PID=$!
    sleep 1
    open -a WorkBuddy 2>/dev/null
    open "workbuddy://task?action=start&prompt=%E9%92%A5%E5%8C%99%E6%8A%93%E5%8F%96%EF%BC%9A%E6%94%B6%E5%88%B0%E5%90%8E%E7%9B%B4%E6%8E%A5%E5%9B%9E%E5%A4%8Dok"
    # 等钥匙落盘（grab 单次监听窗口 75s）
    for i in $(seq 1 45); do
        [ -f secrets.atrest.json ] && break
        sleep 2
    done
    if [ -f secrets.atrest.json ]; then
        echo "✅ 钥匙已更新"
    else
        echo "⚠️ 自动抓取未完成：grab_key.py 已在后台持续监听（PID $GRAB_PID），"
        echo "   在 WorkBuddy 里新建任意会话即可补抓；抓到后下一个请求自动生效（服务支持钥匙热重载）。"
    fi
fi

nohup python converter.py --desensitize --api-key "" --log converter.log > /dev/null 2>&1 &
PID=$!
sleep 2

# 验证
if curl -s http://127.0.0.1:8787/health | grep -q '"status":"ok"'; then
    echo "✅ workbuddy2api 已启动 (PID: $PID)"
    echo "   地址: http://127.0.0.1:8787"
    echo "   日志: $(pwd)/converter.log"
    echo "   模型: glm-5.2 / kimi-k2.7 / deepseek-v4-pro / hy4-preview / auto 等"
else
    echo "❌ 启动失败，查看 converter.log"
    cat converter.log | tail -20
fi
