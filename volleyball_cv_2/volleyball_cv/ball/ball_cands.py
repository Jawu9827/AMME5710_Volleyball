import cv2, numpy as np
def candidates(f, bg):
    fg = (cv2.absdiff(f, bg).max(2) > 35)
    hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
    yel = cv2.inRange(hsv, (14, 110, 110), (36, 255, 255)) > 0
    m = (fg & yel).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, _, st, cen = cv2.connectedComponentsWithStats(m)
    out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if 15 <= a <= 700 and 4 <= max(w, h) <= 32 and 0.45 <= w / h <= 2.2 and a / (w * h) > 0.4:
            out.append((float(cen[i][0]), float(cen[i][1]), float(2 * np.sqrt(a / np.pi)), int(a)))
    return out
