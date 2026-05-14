#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    OCR DIAGNOSTIC TOOL v1.0                                 ║
║                                                                              ║
║  Run this BEFORE the main extractor to see exactly what EasyOCR returns.   ║
║  Share the full terminal output so the root cause can be identified.        ║
╚══════════════════════════════════════════════════════════════════════════════╝

PURPOSE:
    The main extractor v4.0 still produced wrong output.
    This script dumps the RAW EasyOCR tokens with their:
        - exact text string
        - confidence score
        - bounding box coordinates (top-left Y = vertical position)
        - which preprocessing mode was chosen

    This tells us EXACTLY what EasyOCR sees before any of our logic runs,
    so we can pinpoint whether the bug is in:
        (a) preprocessing  — image given to EasyOCR is bad
        (b) OCR itself     — EasyOCR misreads the image
        (c) reconstruction — our FIX-1/FIX-3 logic missed a pattern
        (d) extraction     — field logic still has a gap

RUN:
    python ocr_diagnostic.py
    → select your image file in the dialog
    → copy-paste the full output
"""

import os
import re
import sys
import cv2
import numpy as np
import easyocr
import torch
from tkinter import Tk, filedialog
from pathlib import Path

# ── Minimal preprocessing copy (identical to main script) ─────────────────
DARK_BG_THRESHOLD = 128
RESIZE_SCALE = 0.85
CONTRAST_ALPHA = 1.4
CONTRAST_BETA = 0
STD_DEV_HIGH = 60
ADAPTIVE_BLOCK_SIZE = 15
ADAPTIVE_C = 8
OCR_WIDTH_THRESHOLD = 0.7
OCR_MIN_CONFIDENCE = 0.45


def preprocess_image(img):
    img = cv2.resize(
        img, None, fx=RESIZE_SCALE, fy=RESIZE_SCALE, interpolation=cv2.INTER_AREA
    )
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # FIX-8: Sample centre 40% only — ignores white borders around the card
    h, w = gray.shape
    y1, y2 = int(h * 0.30), int(h * 0.70)
    x1, x2 = int(w * 0.30), int(w * 0.70)
    centre = gray[y1:y2, x1:x2]
    centre_mean = float(np.mean(centre))
    dark_pixel_ratio = float(np.sum(centre < DARK_BG_THRESHOLD)) / centre.size
    dark = (centre_mean < DARK_BG_THRESHOLD) and (dark_pixel_ratio > 0.50)

    if dark:
        gray = cv2.bitwise_not(gray)
        alpha = 2.0  # FIX-9: stronger boost for inverted dark cards
    else:
        alpha = CONTRAST_ALPHA

    std_dev = float(np.std(gray))
    if std_dev > STD_DEV_HIGH:
        mode = f"GLOBAL CONTRAST BOOST α={alpha} (σ={std_dev:.1f})"
        out = cv2.convertScaleAbs(gray, alpha=alpha, beta=CONTRAST_BETA)
    else:
        mode = f"ADAPTIVE THRESHOLD (σ={std_dev:.1f})"
        out = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=ADAPTIVE_BLOCK_SIZE,
            C=ADAPTIVE_C,
        )
    return out, dark, mode


# ── Spaced-letter detector (identical to main script) ──────────────────────
def _is_spaced_letter_token(text):
    segments = text.strip().split()
    if len(segments) < 2:
        return False
    return all(re.fullmatch(r"[A-Za-z]\.?", seg) for seg in segments)


# ── Bbox helpers ───────────────────────────────────────────────────────────
def mid_y(bbox):
    return (bbox[0][1] + bbox[2][1]) / 2.0


def top_y(bbox):
    return bbox[0][1]


def left_x(bbox):
    return bbox[0][0]


def right_x(bbox):
    return bbox[1][0]


def height_px(bbox):
    return bbox[2][1] - bbox[0][1]


def width_px(bbox):
    return bbox[1][0] - bbox[0][0]


# ══════════════════════════════════════════════════════════════════════════════
# MAIN DIAGNOSTIC
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 72)
    print("  OCR DIAGNOSTIC TOOL v1.0")
    print("=" * 72)

    # ── File selection ────────────────────────────────────────────────────
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select ONE Visiting Card Image",
        filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif")],
    )
    root.destroy()

    if not file_path:
        print("No file selected. Exiting.")
        return

    print(f"\n  File : {os.path.basename(file_path)}")
    print(f"  Path : {file_path}")

    # ── Load image ────────────────────────────────────────────────────────
    img = cv2.imread(file_path)
    if img is None:
        print(f"\n  ERROR: OpenCV could not read the file.")
        print(f"  Make sure it is a valid JPG/PNG and not corrupt.")
        return

    h_orig, w_orig = img.shape[:2]
    print(f"  Size : {w_orig} × {h_orig} px")

    # ── Preprocessing ─────────────────────────────────────────────────────
    processed, is_dark, mode = preprocess_image(img)
    h_proc, w_proc = processed.shape[:2]

    print(f"\n{'─' * 72}")
    print(f"  PREPROCESSING")
    print(f"{'─' * 72}")
    print(
        f"  Dark background detected : {'YES — image was INVERTED' if is_dark else 'No'}"
    )
    print(f"  Mode chosen              : {mode}")
    print(f"  Processed size           : {w_proc} × {h_proc} px")

    # Save preprocessed image so you can visually inspect it
    debug_path = os.path.join(os.path.dirname(file_path), "DEBUG_preprocessed.png")
    cv2.imwrite(debug_path, processed)
    print(f"  Preprocessed image saved : {debug_path}")
    print(f"  ↑ Open this file to see exactly what EasyOCR receives.")

    # ── OPTION 4: Show preprocessed image in a popup window ──────────────
    # Resize for display if the image is very large (keeps window on screen)
    disp_max = 1000
    disp_h, disp_w = processed.shape[:2]
    if disp_w > disp_max or disp_h > disp_max:
        scale_d = disp_max / max(disp_w, disp_h)
        disp_img = cv2.resize(
            processed, None, fx=scale_d, fy=scale_d, interpolation=cv2.INTER_AREA
        )
    else:
        disp_img = processed.copy()

    print(f"\n  ┌─────────────────────────────────────────────────────────┐")
    print(f"  │  POPUP WINDOW 1: What EasyOCR sees after preprocessing  │")
    print(f"  │  White = background  |  Black = text (what OCR reads)   │")
    print(f"  │  Press ANY KEY to close and continue...                 │")
    print(f"  └─────────────────────────────────────────────────────────┘")
    cv2.imshow(
        "WINDOW 1: Preprocessed image (what EasyOCR sees) — press any key", disp_img
    )
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # ── EasyOCR ───────────────────────────────────────────────────────────
    use_gpu = torch.cuda.is_available()
    print(f"\n{'─' * 72}")
    print(f"  LOADING EASYOCR  (GPU: {'YES' if use_gpu else 'NO'})")
    print(f"{'─' * 72}")
    reader = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)
    print("  EasyOCR loaded.")

    # Run OCR on preprocessed image
    raw = reader.readtext(processed, width_ths=OCR_WIDTH_THRESHOLD)

    print(f"\n{'─' * 72}")
    print(f"  RAW OCR OUTPUT  ({len(raw)} tokens total)")
    print(f"{'─' * 72}")
    print(
        f"  {'#':<4} {'CONF':>6}  {'TOP-Y':>6}  {'LEFT-X':>7}  "
        f"{'W×H':>9}  {'SPACED?':<8}  TEXT"
    )
    print(f"  {'─'*4} {'─'*6}  {'─'*6}  {'─'*7}  {'─'*9}  {'─'*8}  {'─'*40}")

    above_threshold = []
    below_threshold = []

    for i, (bbox, text, conf) in enumerate(raw):
        ty = top_y(bbox)
        lx = left_x(bbox)
        w = width_px(bbox)
        h = height_px(bbox)
        sp = "YES ←" if _is_spaced_letter_token(text) else ""
        mark = "  " if conf >= OCR_MIN_CONFIDENCE else "✗ "  # ✗ = below threshold

        print(
            f"  {mark}{i+1:<3} {conf:>6.3f}  {ty:>6.1f}  {lx:>7.1f}  "
            f"{w:>4.0f}×{h:<4.0f}  {sp:<8}  {text!r}"
        )

        if conf >= OCR_MIN_CONFIDENCE:
            above_threshold.append((bbox, text, conf))
        else:
            below_threshold.append((bbox, text, conf))

    print(
        f"\n  Above confidence threshold ({OCR_MIN_CONFIDENCE}): {len(above_threshold)}"
    )
    print(
        f"  Below confidence threshold ({OCR_MIN_CONFIDENCE}): "
        f"{len(below_threshold)}  ← these are DROPPED before extraction"
    )

    # ── Line grouping simulation ──────────────────────────────────────────
    SAME_LINE_Y_TOL = 18
    print(f"\n{'─' * 72}")
    print(f"  LINE GROUPING SIMULATION  (Y-tolerance = {SAME_LINE_Y_TOL}px)")
    print(f"{'─' * 72}")
    print(f"  Shows how tokens are grouped into lines for email/company assembly.")
    print()

    sorted_r = sorted(above_threshold, key=lambda r: (mid_y(r[0]), left_x(r[0])))

    lines = []
    if sorted_r:
        current = [sorted_r[0]]
        for item in sorted_r[1:]:
            if abs(mid_y(item[0]) - mid_y(current[0][0])) <= SAME_LINE_Y_TOL:
                current.append(item)
            else:
                lines.append(current)
                current = [item]
        lines.append(current)

    for li, line in enumerate(lines):
        texts = [t for _, t, _ in line]
        confs = [c for _, _, c in line]
        joined = re.sub(r"\s+", "", "".join(texts))
        avg_conf = sum(confs) / len(confs)
        print(f"  Line {li+1:02d}  (avg_conf={avg_conf:.2f})")
        print(f"    Tokens   : {texts}")
        print(f"    Joined   : {joined!r}")
        print()

    # ── Spaced-letter detection ───────────────────────────────────────────
    spaced = [(t, c) for _, t, c in above_threshold if _is_spaced_letter_token(t)]
    print(f"{'─' * 72}")
    print(f"  SPACED-LETTER TOKENS DETECTED: {len(spaced)}")
    print(f"{'─' * 72}")
    if spaced:
        for t, c in spaced:
            print(f"  conf={c:.3f}  {t!r}")
    else:
        print("  None found.")
        print("  ← If the name is wrong, the name is NOT a spaced-letter token.")
        print("    It arrived as a normal token but was still mis-extracted.")
        print("    Check the raw token list above for the actual name string.")

    # ── Email line scan ───────────────────────────────────────────────────
    EMAIL_RE = re.compile(
        r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", re.IGNORECASE
    )
    print(f"\n{'─' * 72}")
    print(f"  EMAIL SEARCH RESULTS")
    print(f"{'─' * 72}")
    found_email = False
    for li, line in enumerate(lines):
        texts = [t for _, t, _ in line]
        joined = re.sub(r"\s+", "", "".join(texts))
        m = EMAIL_RE.search(joined)
        if m:
            print(f"  ✓ Line {li+1}: email FOUND in joined string → {m.group()!r}")
            found_email = True
        # Also check individual tokens
        for _, t, c in line:
            compact = re.sub(r"\s+", "", t)
            m2 = EMAIL_RE.search(compact)
            if m2:
                print(f"  ✓ Line {li+1}: email FOUND in single token → {m2.group()!r}")
                found_email = True
    if not found_email:
        print("  ✗ No email found in ANY line or token.")
        print("  ← Email line as OCR sees it:")
        for li, line in enumerate(lines):
            texts = [t for _, t, _ in line]
            joined_display = " | ".join(texts)
            if any(c in t for t in texts for c in ["@", ".", "urbanedge", "aditya"]):
                print(f"    Line {li+1}: {joined_display}")

    # ── Name candidate scan ───────────────────────────────────────────────
    print(f"\n{'─' * 72}")
    print(f"  NAME CANDIDATE SCAN")
    print(f"{'─' * 72}")
    card_h = (
        max(int(bbox[2][1]) for bbox, _, _ in above_threshold) if above_threshold else 1
    )
    name_zone_y = card_h * 0.65
    print(f"  Card height : {card_h}px")
    print(f"  Name zone   : top {name_zone_y:.0f}px (65% of card)")
    print()

    JOB_KW = {
        "manager",
        "developer",
        "engineer",
        "director",
        "partner",
        "consultant",
        "founder",
        "ceo",
        "cto",
        "coo",
        "cfo",
        "analyst",
        "lead",
        "specialist",
        "head",
        "officer",
        "executive",
        "president",
        "associate",
        "senior",
        "junior",
        "principal",
        "architect",
        "vp",
        "vice",
        "intern",
        "trainee",
        "advisor",
        "coordinator",
        "supervisor",
        "assistant",
        "deputy",
        "md",
        "proprietor",
        "owner",
        "chairman",
        "trustee",
        "secretary",
        "treasurer",
    }
    CO_KW = {
        "ltd",
        "limited",
        "inc",
        "llp",
        "pvt",
        "private",
        "solutions",
        "consulting",
        "studio",
        "technologies",
        "tech",
        "group",
        "company",
        "labs",
        "services",
        "systems",
        "enterprises",
        "associates",
        "builders",
        "construction",
        "industries",
        "global",
        "international",
        "ventures",
        "holdings",
        "capital",
        "networks",
        "media",
        "digital",
        "infra",
        "infrastructure",
    }
    ADDR_KW = {
        "road",
        "street",
        "complex",
        "floor",
        "level",
        "nagar",
        "city",
        "park",
        "tower",
        "building",
        "block",
        "sector",
        "phase",
        "plot",
        "avenue",
        "lane",
        "colony",
        "society",
        "residency",
        "plaza",
        "centre",
        "center",
        "mall",
    }

    for bbox, text, conf in sorted_r:
        words = text.strip().split()
        ty = top_y(bbox)

        # Check each filter
        len_ok = 1 <= len(words) <= 6
        char_ok = bool(re.fullmatch(r"[A-Za-z\s\.\-\']+", text.strip()))
        no_jobtit = not any(w.lower() in JOB_KW for w in words)
        no_co = not any(w.lower() in CO_KW for w in words)
        no_addr = not any(kw in text.lower() for kw in ADDR_KW)
        no_email = not re.search(r"\.(com|in|org|net|co\.in)", text.lower())
        in_zone = ty <= name_zone_y

        passes = len_ok and char_ok and no_jobtit and no_co and no_addr and no_email
        flags = []
        if not len_ok:
            flags.append(f"WORD_COUNT={len(words)}")
        if not char_ok:
            flags.append("BAD_CHARS")
        if not no_jobtit:
            flags.append("JOB_KEYWORD")
        if not no_co:
            flags.append("COMPANY_KEYWORD")
        if not no_addr:
            flags.append("ADDRESS_KEYWORD")
        if not no_email:
            flags.append("EMAIL_PATTERN")

        status = "✓ CANDIDATE" if passes else f"✗ REJECTED ({', '.join(flags)})"
        zone = "IN ZONE" if in_zone else "out of zone"
        caps = (
            "CAPS"
            if text.isupper()
            else ("Title" if all(w[0].isupper() for w in words if w) else "mixed")
        )

        print(f"  conf={conf:.2f}  y={ty:>6.1f}  [{zone}]  [{caps}]  {status}")
        print(f"    {text!r}")
        print()

    # ── OPTION 4: Second popup — OCR boxes drawn on original image ───────
    # GREEN box = above confidence threshold (kept)
    # RED box   = below confidence threshold (dropped)
    annotated = img.copy()
    scale_back_x = w_orig / w_proc
    scale_back_y = h_orig / h_proc

    for bbox, text, conf in raw:
        pts = np.array(
            [[int(pt[0] * scale_back_x), int(pt[1] * scale_back_y)] for pt in bbox],
            dtype=np.int32,
        )
        color = (0, 200, 0) if conf >= OCR_MIN_CONFIDENCE else (0, 0, 220)
        cv2.polylines(annotated, [pts], isClosed=True, color=color, thickness=2)

        label = f"{text[:28]}{'...' if len(text) > 28 else ''} [{conf:.2f}]"
        label_pos = (pts[0][0], max(pts[0][1] - 6, 12))
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(
            annotated,
            (label_pos[0], label_pos[1] - lh - 2),
            (label_pos[0] + lw, label_pos[1] + 2),
            (30, 30, 30),
            -1,
        )
        cv2.putText(
            annotated,
            label,
            label_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    annotated_path = os.path.join(os.path.dirname(file_path), "DEBUG_annotated.png")
    cv2.imwrite(annotated_path, annotated)
    print(f"\n  Annotated image saved    : {annotated_path}")

    ann_h, ann_w = annotated.shape[:2]
    if ann_w > disp_max or ann_h > disp_max:
        scale_d2 = disp_max / max(ann_w, ann_h)
        disp_ann = cv2.resize(
            annotated, None, fx=scale_d2, fy=scale_d2, interpolation=cv2.INTER_AREA
        )
    else:
        disp_ann = annotated.copy()

    print(f"\n  ┌─────────────────────────────────────────────────────────┐")
    print(f"  │  POPUP WINDOW 2: OCR detections on original image       │")
    print(f"  │  GREEN box = kept (conf >= {OCR_MIN_CONFIDENCE})                     │")
    print(f"  │  RED box   = dropped (conf < {OCR_MIN_CONFIDENCE})                   │")
    print(f"  │  Press ANY KEY to close and finish...                   │")
    print(f"  └─────────────────────────────────────────────────────────┘")
    cv2.imshow(
        "WINDOW 2: OCR detections (GREEN=kept RED=dropped) — press any key", disp_ann
    )
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"{'═' * 72}")
    print(f"  DIAGNOSTIC COMPLETE")
    print(f"{'═' * 72}")
    print(f"  1. Share the ENTIRE terminal output (copy from terminal)")
    print(f"  2. Take a screenshot of WINDOW 1 (preprocessed image)")
    print(f"  3. Take a screenshot of WINDOW 2 (annotated OCR boxes)")
    print(f"  OR share the saved files:")
    print(f"     DEBUG_preprocessed.png  — what EasyOCR sees")
    print(f"     DEBUG_annotated.png     — boxes drawn on original")
    print(f"  These will identify the exact bug.")
    print(f"{'═' * 72}")


if __name__ == "__main__":
    main()
