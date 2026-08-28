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
