#!/bin/bash
# Looped Transformer — 实时训练监控面板 · 一键启动脚本
# ============================================================

set -e

cd "$(dirname "$0")"

echo "=========================================="
echo "  Looped Transformer 监控面板"
echo "=========================================="

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "❌ 未找到 python3，请先安装 Python 3.9+"
    exit 1
fi

# 检查依赖（已安装则跳过）
echo "📦 检查依赖..."
python3 -c "import fastapi, uvicorn" 2>/dev/null || {
    echo "   正在安装 fastapi uvicorn ..."
    uv pip install fastapi uvicorn -q 2>/dev/null || \
    python3 -m pip install fastapi uvicorn -q 2>/dev/null || {
        echo "❌ 依赖安装失败，请手动执行: uv pip install fastapi uvicorn"
        exit 1
    }
}

echo "✓ 依赖就绪"
echo ""
echo "🚀 启动服务: http://localhost:8000"
echo "   按 Ctrl+C 停止"
echo "=========================================="
echo ""

cd web_monitor
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
