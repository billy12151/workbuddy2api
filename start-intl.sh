#!/bin/bash
# workbuddy2api 国际版网关启动脚本（WorkBuddy AI -> OpenAI 兼容 API）
# 用法: bash start-intl.sh
# 与 start.sh（国内版，端口 8787）互不影响，可同时运行。

cd "$(dirname "$0")"

# 只杀国际版实例，不动国内版网关
pkill -f "converter.py.*--variant intl" 2>/dev/null
sleep 1

source .venv/bin/activate
# 显式关闭代理鉴权，与 start.sh 同理（本机回环监听，统一不校验）

# ---------------------------------------------------------------------------
# 钥匙自检：国际版 auth 文件（workbuddy-desktop-ai.info）同样是 $wbEncrypted
# 字段加密；国内外两版共用同一把静态保护器钥匙（keyId 9127dea1b44020a7），
# 所以 secrets.atrest.json 对两版通用，任一 app 抓到的钥匙都可用。
# ---------------------------------------------------------------------------
KEY_CHECK=$(.venv/bin/python3 wbkey/check_key.py workbuddy-desktop-ai.info 2>/dev/null)
if [ "$KEY_CHECK" != "ok" ] && [ "$KEY_CHECK" != "plaintext-auth" ] && [ "$KEY_CHECK" != "no-auth-file" ]; then
    echo "🔑 钥匙缺失或不匹配（$KEY_CHECK），自动重抓..."
    rm -f secrets.atrest.json
    pkill -f "grab_key.py" 2>/dev/null
    .venv/bin/python3 wbkey/grab_key.py > /dev/null 2>&1 &
    GRAB_PID=$!
    sleep 1
    open -a "WorkBuddy AI" 2>/dev/null
    # 等钥匙落盘（grab 单次监听窗口 75s）
    for i in $(seq 1 45); do
        [ -f secrets.atrest.json ] && break
        sleep 2
    done
    if [ -f secrets.atrest.json ]; then
        echo "✅ 钥匙已更新"
    else
        echo "⚠️ 自动抓取未完成：grab_key.py 已在后台持续监听（PID $GRAB_PID），"
        echo "   在 WorkBuddy AI 里新建任意会话即可补抓；抓到后下一个请求自动生效（服务支持钥匙热重载）。"
    fi
fi

nohup python converter.py --variant intl --desensitize --api-key "" --log converter-intl.log > /dev/null 2>&1 &
PID=$!
sleep 2

# 验证（uvicorn 启动需要几秒，重试）
OK=""
for i in $(seq 1 10); do
    if curl -s -m 2 http://127.0.0.1:8788/health | grep -q '"status":"ok"'; then
        OK=1; break
    fi
    sleep 1
done
if [ -n "$OK" ]; then
    echo "✅ workbuddy2api 国际版网关已启动 (PID: $PID)"
    echo "   地址: http://127.0.0.1:8788"
    echo "   日志: $(pwd)/converter-intl.log"
    echo "   模型: gpt-6-luna / gemini-3.8-flash / grok-4.7 / kimi-k3 / glm-5.3 / auto 等"
    echo "   注意: 国际版后端要求首条消息为 system，网关已自动注入"
else
    echo "❌ 启动失败，查看 converter-intl.log"
    tail -20 converter-intl.log 2>/dev/null
fi
