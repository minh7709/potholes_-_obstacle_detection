#!/usr/bin/env bash
# Camera điện thoại (IP Webcam) -> xử lý trên máy tính -> cửa sổ kết quả.
#
#   ./tools/live.sh                      dùng URL mặc định
#   ./tools/live.sh http://IP:8080/video dùng URL khác
#
# Trên điện thoại: mở IP Webcam và bấm Start Server trước.

set -euo pipefail

URL="${1:-http://5.53.241.254:8080/video}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"

if [ ! -x "$PY" ]; then
    echo "Không tìm thấy $PY — tạo môi trường ảo trước đã." >&2
    exit 1
fi

cd "$ROOT"

# Thử chạm vào camera trước, để lỗi hiện ra nhanh thay vì đợi nạp model xong mới biết.
if ! timeout 20 "$PY" - "$URL" <<'EOF'
import sys, cv2
c = cv2.VideoCapture(sys.argv[1])
if not c.isOpened():
    raise SystemExit(1)
ok, f = c.read()
print(f"Camera OK: {f.shape[1]}x{f.shape[0]}", flush=True)
c.release()
EOF
then
    echo "" >&2
    echo "Không nối được camera." >&2
    echo "  URL      : $URL" >&2
    echo "  IP Webcam: đã bấm Start Server chưa? IP trong URL có đúng IP điện thoại không?" >&2
    exit 1
fi

echo "Chạy. Bấm q để thoát."
exec "$PY" "$ROOT/tools/live.py" "$URL"