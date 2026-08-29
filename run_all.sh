#!/usr/bin/env bash
#
# run_all.sh - Pothole Detection System Launch Script
# Khởi động tất cả services với một lệnh duy nhất.
#
# Các port:
#   - FastAPI Backend:  http://localhost:8000
#   - Streamlit Dashboard: http://localhost:8501
#
# Usage:
#   chmod +x run_all.sh
#   ./run_all.sh
#
# Stop:
#   Ctrl+C để dừng tất cả services

set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_PID=""
STREAMLIT_PID=""

cleanup() {
    echo ""
    echo ">>> Đang dừng tất cả services..."
    [ -n "$BACKEND_PID" ] && kill "$BACKEND_PID" 2>/dev/null && echo "  [OK] FastAPI backend (PID $BACKEND_PID)"
    [ -n "$STREAMLIT_PID" ] && kill "$STREAMLIT_PID" 2>/dev/null && echo "  [OK] Streamlit dashboard (PID $STREAMLIT_PID)"
    echo ">>> Đã dừng xong."
    exit 0
}

trap cleanup SIGINT SIGTERM

echo "============================================"
echo "  Pothole Detection System"
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

# 1. Start FastAPI backend (port 8000)
echo "[*] Khởi động FastAPI backend (port 8000)..."
cd "$ROOT_DIR"
python main.py &
BACKEND_PID=$!
echo "  -> PID: $BACKEND_PID | http://localhost:8000"

# Đợi backend khởi động
sleep 10

# 2. Start Streamlit dashboard (port 8501)
echo "[*] Khởi động Streamlit dashboard (port 8501)..."
cd "$ROOT_DIR"
streamlit run app_streamlit.py --server.port 8501 --server.headless true &
STREAMLIT_PID=$!
echo "  -> PID: $STREAMLIT_PID | http://localhost:8501"

echo ""
echo "============================================"
echo "  Tất cả services đã khởi động!"
echo "  FastAPI:  http://localhost:8000"
echo "  Streamlit: http://localhost:8501"
echo "  Nhấn Ctrl+C để dừng tất cả."
echo "============================================"

# Giữ script chạy cho đến khi nhận được Ctrl+C
wait
