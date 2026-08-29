#!/usr/bin/env bash
#
# run_all.sh - RoadVision Pothole Detection Launch Script
#
# Cổng ứng dụng:
#   - Web App & FastAPI Backend: http://localhost:8000
#
# Usage:
#   chmod +x run_all.sh
#   ./run_all.sh
#

set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_PID=""

cleanup() {
    echo ""
    echo ">>> Đang dừng RoadVision..."
    [ -n "$BACKEND_PID" ] && kill "$BACKEND_PID" 2>/dev/null && echo "  [OK] Đã dừng server (PID $BACKEND_PID)"
    echo ">>> Đã dừng xong."
    exit 0
}

trap cleanup SIGINT SIGTERM

echo "============================================"
echo "  RoadVision — Pothole Detection AI System"
echo "============================================"
echo ""

# Kích hoạt virtual environment nếu có
if [ -d "$ROOT_DIR/.venv" ]; then
    echo "[*] Kích hoạt virtual environment..."
    source "$ROOT_DIR/.venv/bin/activate"
elif [ -d "$ROOT_DIR/venv" ]; then
    echo "[*] Kích hoạt virtual environment..."
    source "$ROOT_DIR/venv/bin/activate"
fi

# Start FastAPI Web Server (port 8000)
echo "[*] Khởi động Web App & FastAPI backend (port 8000)..."
cd "$ROOT_DIR"
python main.py &
BACKEND_PID=$!

echo ""
echo "============================================"
echo "  RoadVision đã sẵn sàng!"
echo "  Truy cập: http://localhost:8000"
echo "  Nhấn Ctrl+C để dừng."
echo "============================================"

wait
