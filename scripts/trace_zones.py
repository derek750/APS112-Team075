#!/usr/bin/env python3
"""
Trace heatmap-reference-layout.png: each zone has a distinct fill color. Sample
median BGR in a small patch per zone, classify every pixel by nearest color in
LAB (with a distance cutoff to ignore the grid background), clip to the main
room, then export 8 PNG masks at 1024×412.

Overlay pixels are forced to pure red (R channel in BGR) + alpha; hue from the
reference is only used for segmentation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

REF = Path(__file__).resolve().parent.parent / "heatmap-reference-layout.png"
OUT_DIR = Path(__file__).resolve().parent.parent / "zones"
ORDER = ["A", "B", "C", "D", "E", "F", "G", "H"]

# Sample centers inside each colored region (1024-wide reference; height ~411).
REF_SAMPLE_XY = {
    "A": (340, 247),
    "B": (501, 235),
    "C": (550, 100),
    "D": (430, 315),
    "E": (755, 310),
    "F": (700, 200),
    "G": (940, 270),
    "H": (800, 180),
}

# Max LAB Euclidean distance to accept a pixel as a zone (excludes pale grid).
LAB_DIST_MAX = 40.0

# Interior of black walls on floor-plan-new (same as SVG clip).
CLIP_X0, CLIP_Y0, CLIP_X1, CLIP_Y1 = 270, 48, 996, 382

TARGET_W, TARGET_H = 1024, 412
SAMPLE_R = 5


def main() -> int:
    if not REF.is_file():
        print("missing", REF, file=sys.stderr)
        return 1

    bgr = cv2.imread(str(REF))
    if bgr is None:
        return 1
    h, w = bgr.shape[:2]

    medians: list[np.ndarray] = []
    for letter in ORDER:
        x, y = REF_SAMPLE_XY[letter]
        x = int(np.clip(x, 0, w - 1))
        y = int(np.clip(y, 0, h - 1))
        y0, y1 = max(0, y - SAMPLE_R), min(h, y + SAMPLE_R + 1)
        x0, x1 = max(0, x - SAMPLE_R), min(w, x + SAMPLE_R + 1)
        patch = bgr[y0:y1, x0:x1]
        medians.append(np.median(patch.reshape(-1, 3), axis=0))

    refs_bgr = np.array(medians, dtype=np.uint8).reshape(8, 1, 1, 3)
    refs_lab = cv2.cvtColor(refs_bgr.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(8, 3).astype(
        np.float32
    )

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    diff = lab[:, :, np.newaxis, :] - refs_lab[np.newaxis, np.newaxis, :, :]
    dist = np.sqrt((diff**2).sum(axis=3))
    owner = dist.argmin(axis=2).astype(np.int32)
    min_d = dist.min(axis=2)
    valid = min_d < LAB_DIST_MAX

    yy, xx = np.mgrid[0:h, 0:w]
    inside = (xx >= CLIP_X0) & (xx < CLIP_X1) & (yy >= CLIP_Y0) & (yy < CLIP_Y1)
    owner = np.where(valid & inside, owner, -1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    labels: dict[str, list[float]] = {}

    owner_big = cv2.resize(
        owner.astype(np.float32),
        (TARGET_W, TARGET_H),
        interpolation=cv2.INTER_NEAREST,
    )
    owner_big = np.round(owner_big).astype(np.int32)

    dilate_k = np.ones((5, 5), np.uint8)
    blur_ksize = (41, 41)

    for i, letter in enumerate(ORDER):
        m = (owner_big == i).astype(np.uint8) * 255
        m = cv2.dilate(m, dilate_k, iterations=2)
        m = cv2.GaussianBlur(m, blur_ksize, 0)
        bgra = np.zeros((TARGET_H, TARGET_W, 4), dtype=np.uint8)
        bgra[:, :, 2] = 255  # BGR red
        bgra[:, :, 3] = m
        out_path = OUT_DIR / f"mask-{letter}.png"
        cv2.imwrite(str(out_path), bgra)
        M = cv2.moments(m)
        if M["m00"] > 0:
            labels[letter] = [
                round(M["m10"] / M["m00"], 1),
                round(M["m01"] / M["m00"], 1),
            ]
        print("wrote", out_path, "nonzero", int((m > 0).sum()))

    labels_path = OUT_DIR.parent / "zone-labels.json"
    with open(labels_path, "w") as f:
        json.dump(labels, f, indent=2)
    print("wrote", labels_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
