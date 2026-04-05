#!/usr/bin/env python3
from PIL import Image
import numpy as np

path = "/Users/derek/Developer/MoS/heatmap-reference-layout.png"
im = Image.open(path).convert("RGBA")
a = np.array(im)
alpha = a[:, :, 3]
print("alpha unique sample", np.unique(alpha)[:20], "...", "max", alpha.max())
rgb = a[:, :, :3].astype(np.float32)

# Foosball room etc - try mask: pixels that look like tinted overlay (not pure white/gray floor)
r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
# semi-transparent colored: deviation from white
dev = np.sqrt((r - 248) ** 2 + (g - 248) ** 2 + (b - 248) ** 2)
mask = dev > 25
print("dev mask", mask.sum())

# K-means k=8 on masked pixels positions+color
from numpy.random import default_rng
rng = default_rng(0)
ys, xs = np.where(mask)
pixels = np.column_stack([xs / 1024, ys / 411, r[mask] / 255, g[mask] / 255, b[mask] / 255])
# subsample if huge
n = pixels.shape[0]
idx = rng.choice(n, size=min(8000, n), replace=False)
X = pixels[idx]
k = 8
# random init centroids
centroids = X[rng.choice(X.shape[0], k, replace=False)]
for _ in range(25):
    d = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
    labels = d.argmin(axis=1)
    new_c = np.array([X[labels == j].mean(0) if (labels == j).any() else centroids[j] for j in range(k)])
    centroids = new_c

d = np.linalg.norm(pixels[:, None, :] - centroids[None, :, :], axis=2)
labels = d.argmin(axis=1)
print("cluster sizes", np.bincount(labels, minlength=k))

# Map clusters to approximate letter order by centroid x then y
order = np.lexsort((centroids[:, 1], centroids[:, 0]))
print("cluster order by position", order)
print("centroids xy (norm)", centroids[:, 0] * 1024, centroids[:, 1] * 411)
