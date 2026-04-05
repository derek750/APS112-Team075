import cv2
import numpy as np

path = "/Users/derek/Developer/MoS/heatmap-reference-layout.png"
bgr = cv2.imread(path)
hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
# red wraps in OpenCV H 0-179
lower1 = np.array([0, 40, 40])
upper1 = np.array([12, 255, 255])
lower2 = np.array([165, 40, 40])
upper2 = np.array([179, 255, 255])
m1 = cv2.inRange(hsv, lower1, upper1)
m2 = cv2.inRange(hsv, lower2, upper2)
red = cv2.bitwise_or(m1, m2)
n, labels, stats, centroids = cv2.connectedComponentsWithStats(red, connectivity=8)
print("red components", n - 1)
for i in range(1, n):
    print(i, "area", stats[i, cv2.CC_STAT_AREA], "bbox", stats[i, cv2.CC_STAT_LEFT:cv2.CC_STAT_LEFT+4], "c", centroids[i])
