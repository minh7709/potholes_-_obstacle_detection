#!/usr/bin/env bash
# Camera điện thoại (IP Webcam) -> xử lý trên máy tính -> cửa sổ kết quả.
#
#   ./tools/live.sh --url http://PHONE_IP:8080/video
#   ./tools/live.sh --url http://PHONE_IP:8080/video --width 960
#   ./tools/live.sh --url dashcam.mp4                  dùng video file
#
# Trên điện thoại: mở IP Webcam và bấm Start Server trước.
# Thay PHONE_IP bằng IP thực của điện thoại trên mạng LAN.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"

if [ ! -x "$PY" ]; then
    # Fallback: try system python or Scripts on Windows
    if [ -x "$ROOT/.venv/Scripts/python.exe" ]; then
        PY="$ROOT/.venv/Scripts/python.exe"
    else
        echo "Không tìm thấy Python trong .venv — tạo môi trường ảo trước đã." >&2
        exit 1
    fi
fi

cd "$ROOT"

echo "Chạy. Bấm q để thoát."
exec "$PY" "$ROOT/tools/live.py" "$@"