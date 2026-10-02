#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# run_all.sh - RoadGuard Launch Script (Web App + FastAPI Backend)
#
# Cổng ứng dụng:
#   - Web App & FastAPI Backend: http://localhost:8000
#
# Usage:
#   chmod +x run_all.sh
#   ./run_all.sh
#
# LƯU Ý: phải dùng .venv (Python 3.12). Venv cũ chạy Python 3.14 không có
# torch/pygame nên không chạy được pipeline FCW.
# ─────────────────────────────────────────────────────────────────────

set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_PID=""

cleanup() {
    echo ""
    echo ">>> Đang dừng RoadGuard..."
    [ -n "$BACKEND_PID" ] && kill "$BACKEND_PID" 2>/dev/null && echo "  [OK] Đã dừng server (PID $BACKEND_PID)"
    echo ">>> Đã dừng xong."
    exit 0
}

trap cleanup SIGINT SIGTERM

echo "============================================"
echo "  RoadGuard — Forward Collision Warning"
echo "============================================"
echo ""

# Kích hoạt virtual environment (bắt buộc .venv cho Python 3.12 + torch)
if [ -d "$ROOT_DIR/.venv" ]; then
    echo "[*] Kích hoạt .venv (Python 3.12 + PyTorch)..."
    source "$ROOT_DIR/.venv/bin/activate"
elif [ -d "$ROOT_DIR/venv" ]; then
    echo "[!] Cảnh báo: chỉ thấy 'venv' (Python 3.14) — thiếu torch/pygame."
    echo "    Tạo .venv bằng: uv venv --python 3.12 $ROOT_DIR/.venv"
    echo "                   uv pip install --python $ROOT_DIR/.venv/bin/python \\"
    echo "                       -r $ROOT_DIR/requirements-fcw.txt \\"
    echo "                       --index-url https://download.pytorch.org/whl/cpu"
    exit 1
fi

# Start FastAPI Web Server (port 8000)
echo "[*] Khởi động Web App & FastAPI backend (port 8000)..."
cd "$ROOT_DIR"
python main.py &
BACKEND_PID=$!

echo ""
echo "============================================"
echo "  RoadGuard đã sẵn sàng!"
echo "  Truy cập: http://localhost:8000"
echo "  Nhấn Ctrl+C để dừng."
echo "============================================"

wait
