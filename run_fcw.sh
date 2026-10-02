#!/usr/bin/env bash
#
# run_fcw.sh - Launch the Forward Collision Warning system
#
# Usage:
#   ./run_fcw.sh                          # webcam (source 0)
#   ./run_fcw.sh video.mp4                # video file
#   ./run_fcw.sh image.jpg                # single image (distance only)
#   ./run_fcw.sh 0 --no-display           # headless, no GUI window
#   ./run_fcw.sh video.mp4 --save-video   # also write an annotated video
#
# Any extra arguments are forwarded to main_fcw.py.

set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "[!] Virtual environment not found at $VENV_DIR"
    echo "    Create it with:"
    echo "      uv venv --python 3.12 $VENV_DIR"
    echo "      uv pip install --python $VENV_DIR/bin/python -r $ROOT_DIR/requirements-fcw.txt \\"
    echo "          --index-url https://download.pytorch.org/whl/cpu"
    exit 1
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

cd "$ROOT_DIR"

# Default source is the webcam when the caller supplies none.
if [ $# -eq 0 ]; then
    set -- --source 0
fi

echo "============================================"
echo "  RoadGuard — Forward Collision Warning"
echo "============================================"
echo "  Command: python main_fcw.py $*"
echo "============================================"

exec python main_fcw.py "$@"
