import cv2
import numpy as np
import re
from typing import Any, Dict, List, Optional, Tuple

from core.config import CFG
from core.models import OCRToken

_PADDLE_CACHE: Dict[str, Any] = {}


def _get_paddle(lang: str = "en"):
    if lang not in _PADDLE_CACHE:
        from paddleocr import PaddleOCR

        _PADDLE_CACHE[lang] = PaddleOCR(lang=lang)
    return _PADDLE_CACHE[lang]


def _parse_paddle_result(results: Any) -> List[Tuple[str, float, List]]:
    parsed = []

    if not results:
        return parsed

    try:
        result = results[0]
        texts = result.get("rec_texts", [])
        scores = result.get("rec_scores", [])
        polys = result.get("rec_polys", [])

        for text, conf, poly in zip(texts, scores, polys):
            text = str(text).strip()
            conf = float(conf)
            if not text:
                continue
            if conf < CFG["OCR_CONF_THRESH"]:
                continue
            pts = [[float(x), float(y)] for x, y in poly]
            parsed.append((text, conf, pts))
    except Exception:
        pass
    return parsed


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).lower()


def _bbox_from_pts(pts: List) -> tuple[float, float, float, float]:
    xs = [float(p[0]) for p in pts]
    ys = [float(p[1]) for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def _rect_iou(a: dict, b: dict) -> float:
    ax0, ay0, ax1, ay1 = _bbox_from_pts(a["bbox"])
    bx0, by0, bx1, by1 = _bbox_from_pts(b["bbox"])
    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)
    inter_w = max(0.0, inter_x1 - inter_x0)
    inter_h = max(0.0, inter_y1 - inter_y0)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union_area = area_a + area_b - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def _is_duplicate_token(a: dict, b: dict) -> bool:
    if _normalize_text(a["text"]) != _normalize_text(b["text"]):
        return False
    if _rect_iou(a, b) >= 0.40:
        return True
    return False


def _iou_dedup(pool: dict) -> dict:
    tokens = sorted(pool.values(), key=lambda t: t["conf"], reverse=True)
    kept: list[dict] = []
    for token in tokens:
        if any(_is_duplicate_token(token, existing) for existing in kept):
            continue
        kept.append(token)
    return {idx: tok for idx, tok in enumerate(kept)}


def _run_ocr_pass(img_bgr: np.ndarray, lang: str, cell_px: float, pool: dict):
    if img_bgr.ndim == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)

    try:
        ocr = _get_paddle(lang)
        result = ocr.predict(img_bgr)
        parsed = _parse_paddle_result(result)
        added = 0
        duplicate_found = 0
        for text, conf, pts in parsed:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            tok = {
                "text": text,
                "conf": conf,
                "bbox": pts,
                "cx": sum(xs) / len(xs),
                "cy": sum(ys) / len(ys),
                "th": max(ys) - min(ys),
                "tw": max(xs) - min(xs),
            }
            if CFG["DEBUG"]:
                if any(
                    existing["text"] == text and existing["bbox"] == pts
                    for existing in pool.values()
                ):
                    duplicate_found += 1
            pool[len(pool)] = tok
            added += 1
        if CFG["DEBUG"]:
            print(
                f"OCR_PASS [{lang}] added={added} duplicates_in_pass={duplicate_found} pool_size={len(pool)}"
            )
    except Exception:
        pass


def detect_scripts(text: str) -> Tuple[str, Optional[str]]:
    counts: Dict[str, int] = {"latin": 0, "cjk": 0, "arabic": 0, "devanagari": 0}
    for ch in text:
        if not ch.strip():
            continue
        cp = ord(ch)
        if 0x4E00 <= cp <= 0x9FFF:
            counts["cjk"] += 1
        elif 0x0600 <= cp <= 0x06FF:
            counts["arabic"] += 1
        elif 0x0900 <= cp <= 0x097F:
            counts["devanagari"] += 1
        elif ch.isalpha():
            counts["latin"] += 1
    ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    total = max(sum(counts.values()), 1)
    secondary = next(
        (sc for sc, cnt in ranked[1:] if cnt / total >= CFG["SECONDARY_SCRIPT_THRESH"]),
        None,
    )
    return ranked[0][0], secondary


def ocr_card(
    proc: np.ndarray, lang_override: Optional[str] = None
) -> Tuple[List[dict], str, Optional[str]]:
    h = proc.shape[0]
    cell_px = max(h * CFG["IOU_DEDUP_CELL_FACTOR"], CFG["IOU_DEDUP_CELL_MIN_PX"])
    pool: dict = {}
    base = lang_override or "en"
    if CFG["DEBUG"]:
        print("OCR_CARD: starting pass1")
    _run_ocr_pass(proc, base, cell_px, pool)
    if CFG["DEBUG"]:
        print(f"OCR_CARD: after pass1 pool_size={len(pool)}")
    _run_ocr_pass(
        cv2.convertScaleAbs(
            proc, alpha=CFG["OCR_BRIGHTNESS_ALPHA"], beta=CFG["OCR_BRIGHTNESS_BETA"]
        ),
        base,
        cell_px,
        pool,
    )
    if CFG["DEBUG"]:
        print(f"OCR_CARD: after pass2 pool_size={len(pool)}")
    pool = _iou_dedup(pool)
    tokens = list(pool.values())
    if CFG["DEBUG"]:
        print(f"OCR_CARD: after dedup pool_size={len(tokens)}")
    primary, secondary = detect_scripts(" ".join(t["text"] for t in tokens))
    return tokens, primary, secondary
