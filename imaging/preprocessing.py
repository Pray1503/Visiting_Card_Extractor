import cv2
import numpy as np

from core.config import CFG


def deskew(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    lines = cv2.HoughLinesP(
        cv2.Canny(gray, 50, 150),
        1,
        np.pi / 180,
        threshold=80,
        minLineLength=60,
        maxLineGap=20,
    )
    if lines is None:
        return img
    angles = []
    for l in lines:
        a = np.degrees(np.arctan2(l[0][3] - l[0][1], l[0][2] - l[0][0]))
        if abs(a) < 45:
            angles.append(a)
    if not angles:
        return img
    angle = float(np.median(angles))
    if abs(angle) < 0.5:
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos_a, sin_a = abs(M[0, 0]), abs(M[0, 1])
    nw = int(h * sin_a + w * cos_a)
    nh = int(h * cos_a + w * sin_a)
    M[0, 2] += (nw - w) / 2
    M[1, 2] += (nh - h) / 2
    return cv2.warpAffine(
        img, M, (nw, nh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def preprocess(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    roi = gray[
        int(img.shape[0] * 0.3) : int(img.shape[0] * 0.7),
        int(img.shape[1] * 0.3) : int(img.shape[1] * 0.7),
    ]
    if roi.size > 0 and float(np.mean(roi)) < 120:
        gray = cv2.bitwise_not(gray)
    gray = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
    sharpened = cv2.addWeighted(gray, 1.55, cv2.GaussianBlur(gray, (0, 0), 3), -0.55, 0)
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)


def segment_cards(img: np.ndarray) -> list[np.ndarray]:
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    dilated = cv2.dilate(edges, kernel, iterations=3)
    cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pad = CFG["SEG_PAD_PX"]
    rois = []
    for c in cnts:
        area = cv2.contourArea(c)
        if not (
            h * w * CFG["SEG_MIN_AREA_FRAC"] <= area <= h * w * CFG["SEG_MAX_AREA_FRAC"]
        ):
            continue
        x, y, cw, ch = cv2.boundingRect(c)
        aspect = cw / max(ch, 1)
        if CFG["SEG_CARD_ASPECT_MIN"] <= aspect <= CFG["SEG_CARD_ASPECT_MAX"]:
            rois.append((x, y, cw, ch))
    if not rois:
        return [img]
    rois.sort(key=lambda r: (r[1] // 100, r[0]))
    return [
        img[
            max(0, y - pad) : min(h, y + ch + pad),
            max(0, x - pad) : min(w, x + cw + pad),
        ]
        for x, y, cw, ch in rois
    ]
