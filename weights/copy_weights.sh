#!/usr/bin/env bash
# Collect the six published checkpoints from the original run folders, give them
# clear names, and write SHA-256 checksums. Run on the training machine:
#
#   bash weights/copy_weights.sh /home/constantin/Doctorat/YoloLib/YoloModels/YoloV12/BestLuggageArchs
#
# The .pt files are NOT meant to be committed to git (GitHub rejects files > 100 MB);
# upload them as assets of a GitHub Release (or to Zenodo) and keep only
# SHA256SUMS.txt in the repository.
set -euo pipefail

SRC="${1:?usage: $0 /path/to/BestLuggageArchs}"
OUT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# published name                      source file (relative to BestLuggageArchs)
cp -v "$SRC/best12/yolov12s_default/weights/best.pt"       "$OUT/yolov12s_default.pt"
cp -v "$SRC/best12/ls_shift_sqrt/weights/best.pt"          "$OUT/yolov12s_lsshift_sqrt.pt"
cp -v "$SRC/best12/yolov12x_lsshift_sqrt2/weights/best.pt" "$OUT/yolov12x_lsshift.pt"
cp -v "$SRC/best26/y26_base_rep/weights/best.pt"           "$OUT/yolo26s_default.pt"
cp -v "$SRC/best26/y26_p234rich_b1/weights/best.pt"        "$OUT/yolo26s_p234rich_beta1.pt"
cp -v "$SRC/best26/y26x_p234rich-6/weights/best.pt"        "$OUT/yolo26x_p234rich.pt"

# Note: in the YOLOv12 runs, best.pt holds the FINAL-epoch checkpoint (the training code
# used for those runs wrote the last epoch to best.pt). These are the files evaluated in
# the paper.

cd "$OUT" && sha256sum ./*.pt | tee SHA256SUMS.txt
