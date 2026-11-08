#!/usr/bin/env bash
# Build one of the two patched code bases used in the paper.
#
#   bash ultralytics_patches/setup_env.sh v12   # Environment A: YOLOv12 models
#   bash ultralytics_patches/setup_env.sh v26   # Environment B: YOLO26 models
#
# Both code bases install a package called "ultralytics", so use a separate
# virtual environment (or conda env) for each one.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WHICH="${1:-}"
case "$WHICH" in
  v12) REPO=https://github.com/sunsmarterjie/yolov12.git
       COMMIT=2db2baea0f62e896ac85d4c412925a03136ddda7 ;;   # YOLOv12 main, ultralytics 8.3.63
  v26) REPO=https://github.com/ultralytics/ultralytics.git
       COMMIT=12d61b806db2860e4209b5a86ab634e5a4954d88 ;;   # ultralytics main, 8.4.117
  *)   echo "usage: $0 v12|v26 [target_dir]"; exit 1 ;;
esac
DEST="${2:-$HERE/../third_party/ultralytics-$WHICH}"

if [ ! -d "$DEST/.git" ]; then
  git clone "$REPO" "$DEST"
fi
git -C "$DEST" checkout --quiet "$COMMIT"

# Overwrite the changed files with the versions used in the paper.
cp -rv "$HERE/$WHICH/ultralytics/." "$DEST/ultralytics/"

pip install -e "$DEST"
echo
echo "Done: patched $WHICH code base at $DEST (commit ${COMMIT:0:9})."
