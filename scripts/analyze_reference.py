#!/usr/bin/env python3
"""Inspect reference overlay colors for tracing."""
from PIL import Image
import numpy as np

path = "/Users/derek/Developer/MoS/heatmap-reference-layout.png"
im = Image.open(path).convert("RGBA")
a = np.array(im)
rgb = a[:, :, :3].astype(np.float32)
h, w = rgb.shape[:2]
print("size", w, h)

# saturation
mx = rgb.max(axis=2)
mn = rgb.min(axis=2)
sat = np.where(mx > 1, (mx - mn) / mx, 0)

# not background: grid is light gray
r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
bg = (r > 210) & (g > 210) & (b > 210) & (sat < 0.12)
overlay = ~bg & (a[:, :, 3] > 200)
print("overlay pixels", overlay.sum())

# sample colors from overlay
ov = rgb[overlay]
print("RGB min/max", ov.min(0), ov.max(0))
print("mean RGB", ov.mean(0))
