# #!/usr/bin/env python3
# """
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║        VISITING CARD OCR ENGINE  v8.0  —  PADDLEOCR PRECISION EDITION      ║
# ║                                                                              ║
# ║  Architecture (4 layers):                                                   ║
# ║  ┌─────────────────────────────────────────────────────────────────────┐   ║
# ║  │ L1 — CV PIPELINE   Deskew → CLAHE → IOU Dedup → Fragment Merge     │   ║
# ║  │ L2 — OCR           Triple-pass PaddleOCR + Tesseract fallback       │   ║
# ║  │ L3 — EXTRACTION    Regex + phonenumbers + spaCy NER (supplemental)  │   ║
# ║  │ L4 — LLM (opt.)    HuggingFace Llama-3.1-8B post-processing        │   ║
# ║  └─────────────────────────────────────────────────────────────────────┘   ║
# ║                                                                              ║
# ║  UPGRADE v8.0 — PaddleOCR replaces EasyOCR:                                 ║
# ║  • ~3-5× faster on CPU (DB detector + CRNN recogniser)                      ║
# ║  • Better accuracy on dense/small text typical of business cards            ║
# ║  • Better multilingual support (Indian scripts via lang="en" + "ml"/"hi")   ║
# ║  • Lower memory footprint; handles rotated/curved text                      ║
# ║  • Triple-pass: normal + brightened + contrast-boosted images               ║
# ║                                                                              ║
# ║ INSTALL:                                                                     ║
# ║   pip install paddlepaddle paddleocr                          # CPU          ║
# ║   pip install paddlepaddle-gpu paddleocr                      # GPU          ║
# ║   pip install opencv-python-headless numpy pillow openpyxl pandas           ║
# ║   pip install phonenumbers spacy tldextract                                  ║
# ║   python -m spacy download en_core_web_sm                                   ║
# ║   (Optional LLM)  pip install langchain langchain-huggingface                ║
# ║   (Optional PDF)  pip install pdf2image                                      ║
# ║   (Optional OCR)  pip install pytesseract + Tesseract binary                ║
# ║                                                                              ║
# ║ USAGE:                                                                       ║
# ║   python card_engine_v8.py                        # GUI picker              ║
# ║   python card_engine_v8.py img.jpg                # single image            ║
# ║   python card_engine_v8.py ./cards/               # batch folder            ║
# ║   python card_engine_v8.py img.jpg --debug        # debug images            ║
# ║   python card_engine_v8.py img.jpg --llm HF_TOKEN # LLM post-processing    ║
# ║   python card_engine_v8.py img.jpg --lang hi      # Hindi card              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
# """

# # ══════════════════════════════════════════════════════════════════════════════
# #  IMPORTS
# # ══════════════════════════════════════════════════════════════════════════════
# import os, re, sys, json, math, csv, warnings, argparse, unicodedata, traceback
# from pathlib import Path
# from datetime import datetime

# import cv2
# import numpy as np
# from PIL import Image

# warnings.filterwarnings("ignore")

# # ── Optional / graceful imports ───────────────────────────────────────────────

# # PaddleOCR — primary engine
# try:
#     from paddleocr import PaddleOCR

#     PADDLE_OK = True
# except ImportError:
#     PADDLE_OK = False
#     print("[WARN] pip install paddlepaddle paddleocr")

# try:
#     import pytesseract

#     TESS_OK = True
# except ImportError:
#     TESS_OK = False
#     print("[WARN] pip install pytesseract")

# try:
#     import torch

#     GPU_OK = torch.cuda.is_available()
# except ImportError:
#     GPU_OK = False

# try:
#     import phonenumbers
#     from phonenumbers import PhoneNumberMatcher, PhoneNumberFormat, NumberParseException

#     PH_OK = True
# except ImportError:
#     PH_OK = False
#     print("[WARN] pip install phonenumbers")

# try:
#     import spacy

#     SPACY_OK = True
# except ImportError:
#     SPACY_OK = False
#     print("[WARN] pip install spacy && python -m spacy download en_core_web_sm")

# try:
#     import tldextract

#     TLD_OK = True
# except ImportError:
#     TLD_OK = False
#     print("[WARN] pip install tldextract")

# try:
#     import pandas as pd
#     from openpyxl import Workbook
#     from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
#     from openpyxl.utils import get_column_letter

#     EXCEL_OK = True
# except ImportError:
#     EXCEL_OK = False
#     print("[WARN] pip install openpyxl pandas")

# try:
#     from pdf2image import convert_from_path

#     PDF_OK = True
# except ImportError:
#     PDF_OK = False

# # ── LLM imports (lazy — only loaded when --llm is passed) ────────────────────
# LLM_OK = False
# try:
#     from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
#     from langchain_core.prompts import ChatPromptTemplate
#     from langchain_core.output_parsers import JsonOutputParser

#     LLM_OK = True
# except ImportError:
#     pass


# # ══════════════════════════════════════════════════════════════════════════════
# #  CONFIGURATION
# # ══════════════════════════════════════════════════════════════════════════════
# CFG = {
#     # Image sizing
#     "RESIZE_MAX_SIDE": 2200,
#     "RESIZE_MIN_SIDE": 900,
#     # Dark background detection (centre 40%×40% crop)
#     "DARK_BG_THRESHOLD": 128,
#     "DARK_PIXEL_RATIO": 0.50,
#     # CLAHE
#     "CLAHE_CLIP": 2.5,
#     "CLAHE_TILE": (8, 8),
#     "UNSHARP_AMOUNT": 0.55,
#     "STD_DEV_ADAPTIVE": 42,
#     # Deskew
#     "DESKEW_MIN_ANGLE": 0.5,
#     "DESKEW_HOUGH_THRESH": 80,
#     "DESKEW_MIN_LINE": 60,
#     "DESKEW_MAX_GAP": 20,
#     # OCR
#     "OCR_CONF_THRESH": 0.28,
#     "OCR_WIDTH_THS": 0.75,
#     "OCR_DECODER": "greedy",
#     # PaddleOCR specific
#     "PADDLE_DET_DB_THRESH": 0.3,  # detection binarisation threshold
#     "PADDLE_DET_DB_BOX_THRESH": 0.5,
#     "PADDLE_REC_ALGORITHM": "CRNN",
#     "PADDLE_LANG": "en",  # overridden by --lang CLI flag
#     # IOU spatial dedup
#     "IOU_THRESH": 0.30,
#     # Horizontal fragment merge
#     "HMF_GAP_FACTOR": 1.8,
#     # Line grouping
#     "LINE_Y_BASE_TOL": 28,
#     # Phone
#     "PHONE_DEFAULT_REGION": "IN",
#     "PHONE_MIN_DIGITS": 7,
#     "PHONE_MAX_DIGITS": 15,
#     # Name
#     "NAME_MIN_ALPHA_RATIO": 0.60,
#     # LLM
#     "LLM_REPO_ID": "meta-llama/Llama-3.1-8B-Instruct",
#     "LLM_TEMPERATURE": 0.1,
#     "LLM_TIMEOUT": 120,
#     # PDF
#     "PDF_DPI": 200,
#     "POPPLER_PATH": os.environ.get("POPPLER_PATH", None),
# }


# # ══════════════════════════════════════════════════════════════════════════════
# #  REGEX  (compiled once)
# # ══════════════════════════════════════════════════════════════════════════════
# _TLD_PATTERN = (
#     r"(?:com|in|org|net|co\.in|co\.uk|io|biz|info|edu|gov|"
#     r"ae|us|au|nz|sg|hk|my|ph|za|de|fr|jp|cn|ca|br|mx|"
#     r"co\.ae|co\.sg|co\.za|co\.au|co\.nz|co\.jp|co\.ca|"
#     r"tech|app|dev|ai)"
# )
# _EMAIL_RE = re.compile(
#     r"[A-Za-z0-9._%+\-]+\s*@\s*[A-Za-z0-9.\-]+\s*\.\s*" + _TLD_PATTERN, re.IGNORECASE
# )
# _EMAIL_AT_RE = re.compile(
#     r"[A-Za-z0-9._%+\-]+\s*(?:@|\(a\)|\[at\]|\bat\b)\s*[A-Za-z0-9.\-]+\s*\.\s*"
#     + _TLD_PATTERN,
#     re.IGNORECASE,
# )
# _WEB_RE = re.compile(
#     r"(?:https?://|www\.)[A-Za-z0-9.\-/_%?=&#]+"
#     r"|[A-Za-z0-9\-]+\.(?:com|in|org|net|io|co\.in|co\.uk|biz)[/\w\-]*",
#     re.IGNORECASE,
# )
# _PHONE_RE = re.compile(
#     r"(?:\+\s*\d{1,4}[\s\-\.]*)?(?:\([\d\s]+\)[\s\-\.]*)?[\d][\d\s\-\.]{4,18}[\d]"
# )
# _LINKEDIN_RE = re.compile(r"linkedin\.com/in/[A-Za-z0-9_\-]+", re.IGNORECASE)
# _TWITTER_RE = re.compile(
#     r"(?:twitter\.com/|x\.com/|@)[A-Za-z0-9_]{2,50}", re.IGNORECASE
# )

# _TITLE_KW = re.compile(
#     r"\b(?:manager|director|engineer|developer|architect|founder|partner|"
#     r"consultant|analyst|officer|executive|president|associate|senior|"
#     r"junior|principal|head|lead|specialist|advisor|coordinator|"
#     r"supervisor|assistant|proprietor|owner|chairman|trustee|secretary|"
#     r"treasurer|intern|trainee|ceo|cto|coo|cfo|cmo|md|vp|"
#     r"vice\s*president|doctor|dr\.?|prof\.?|professor|lawyer|advocate|"
#     r"solicitor|barrister|accountant|auditor|designer|strategist|"
#     r"planner|researcher|scientist|technician|operator|representative|"
#     r"agent|broker|dealer|trader|contractor|builder)\b",
#     re.IGNORECASE,
# )

# _COMPANY_KW = re.compile(
#     r"\b(?:ltd|limited|inc|llp|llc|pvt|private|plc|corp|corporation|"
#     r"solutions|consulting|studio|technologies|tech|group|company|co\.|"
#     r"labs|services|systems|enterprises|associates|builders|construction|"
#     r"industries|global|international|ventures|holdings|capital|networks|"
#     r"media|digital|infra|infrastructure|realty|realtors|properties|"
#     r"developers|architects|interiors|design|agency|firm|bureau|office|"
#     r"foundation|trust|institute|academy|school|college|hospital|clinic|"
#     r"healthcare|pharma|logistics|transport|exports|imports|trading|"
#     r"manufacturing|fabrication|engineering)\b",
#     re.IGNORECASE,
# )

# _ADDR_KW = re.compile(
#     r"\b(?:floor|fl\.|level|suite|no\.|plot|block|sector|phase|unit|"
#     r"road|rd\.|street|st\.|avenue|ave\.|lane|ln\.|boulevard|blvd\.|"
#     r"nagar|vihar|marg|gali|chowk|bazaar|market|colony|society|"
#     r"residency|plaza|centre|center|mall|complex|tower|building|"
#     r"district|area|zone|suburb|village|town|city|state|province|"
#     r"county|region|po\s*box|p\.o\.|zip|pin|cross)\b",
#     re.IGNORECASE,
# )

# _POSTAL_RE = re.compile(r"\b\d{4,9}\b")


# # ══════════════════════════════════════════════════════════════════════════════
# #  SINGLETON: MODELS  (loaded once, reused across all images)
# # ══════════════════════════════════════════════════════════════════════════════
# _PADDLE_READER = None
# _SPACY_NLP = None


# def get_paddle_reader(lang: str = None):
#     """
#     Returns a cached PaddleOCR instance.
#     PaddleOCR auto-downloads models on first use (~100 MB one-time download).
#     use_angle_cls=True handles upside-down / rotated text on cards.
#     show_log=False suppresses verbose Paddle output.
#     """
#     global _PADDLE_READER
#     lang = lang or CFG["PADDLE_LANG"]
#     if _PADDLE_READER is None and PADDLE_OK:
#         print(f"  [INFO] Loading PaddleOCR model (lang={lang})…")
#         _PADDLE_READER = PaddleOCR(
#             use_angle_cls=True,
#             lang=lang,
#             show_log=False,
#             det_db_thresh=CFG["PADDLE_DET_DB_THRESH"],
#             det_db_box_thresh=CFG["PADDLE_DET_DB_BOX_THRESH"],
#             rec_algorithm=CFG["PADDLE_REC_ALGORITHM"],
#         )
#     return _PADDLE_READER


# def get_spacy_nlp():
#     global _SPACY_NLP
#     if _SPACY_NLP is None and SPACY_OK:
#         try:
#             _SPACY_NLP = spacy.load("en_core_web_sm")
#         except OSError:
#             print(
#                 "  [WARN] spaCy model missing — run: python -m spacy download en_core_web_sm"
#             )
#     return _SPACY_NLP


# # ══════════════════════════════════════════════════════════════════════════════
# #  BBOX HELPERS  (unified — works for both PaddleOCR and Tesseract bbox formats)
# # ══════════════════════════════════════════════════════════════════════════════
# def _top_y(b):
#     return float(b[0][1])


# def _mid_y(b):
#     return (float(b[0][1]) + float(b[2][1])) / 2.0


# def _left_x(b):
#     return float(b[0][0])


# def _right_x(b):
#     return float(b[1][0])


# def _height(b):
#     return float(b[2][1]) - float(b[0][1])


# def _width(b):
#     return float(b[1][0]) - float(b[0][0])


# def _grid_key(b):
#     return f"{int(_left_x(b)//14)},{int(_top_y(b)//14)}"


# # ══════════════════════════════════════════════════════════════════════════════
# #  LAYER 1 — CV PIPELINE
# # ══════════════════════════════════════════════════════════════════════════════


# def load_image(path: str) -> np.ndarray:
#     img = cv2.imread(path)
#     if img is None:
#         pil = Image.open(path).convert("RGB")
#         img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
#     return img


# def resize_for_ocr(img: np.ndarray) -> np.ndarray:
#     h, w = img.shape[:2]
#     max_s, min_s = max(h, w), min(h, w)
#     if max_s < CFG["RESIZE_MIN_SIDE"]:
#         scale = CFG["RESIZE_MIN_SIDE"] / min_s
#     elif max_s > CFG["RESIZE_MAX_SIDE"]:
#         scale = CFG["RESIZE_MAX_SIDE"] / max_s
#     else:
#         scale = 1.0
#     if abs(scale - 1.0) < 0.02:
#         return img
#     return cv2.resize(
#         img,
#         None,
#         fx=scale,
#         fy=scale,
#         interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA,
#     )


# def deskew(img: np.ndarray) -> tuple:
#     """
#     Hough-line deskew. Fixes tilted cards that cause OCR to fragment tokens.
#     Returns (corrected_img, angle_degrees).
#     """
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (5, 5), 0)
#     ot, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     edges = cv2.Canny(blur, ot * 0.5, ot)
#     lines = cv2.HoughLinesP(
#         edges,
#         1,
#         math.pi / 180,
#         threshold=CFG["DESKEW_HOUGH_THRESH"],
#         minLineLength=CFG["DESKEW_MIN_LINE"],
#         maxLineGap=CFG["DESKEW_MAX_GAP"],
#     )
#     if lines is None or len(lines) < 3:
#         return img, 0.0
#     angles = []
#     for line in lines:
#         x1, y1, x2, y2 = line[0]
#         a = math.degrees(math.atan2(y2 - y1, x2 - x1))
#         if abs(a) < 45:
#             angles.append(a)
#     if not angles:
#         return img, 0.0
#     angle = float(np.median(angles))
#     if abs(angle) < CFG["DESKEW_MIN_ANGLE"]:
#         return img, 0.0
#     h, w = img.shape[:2]
#     cx, cy = w // 2, h // 2
#     M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
#     cos_a, sin_a = abs(M[0, 0]), abs(M[0, 1])
#     nw = int(h * sin_a + w * cos_a)
#     nh = int(h * cos_a + w * sin_a)
#     M[0, 2] += (nw - w) / 2
#     M[1, 2] += (nh - h) / 2
#     return (
#         cv2.warpAffine(
#             img, M, (nw, nh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
#         ),
#         angle,
#     )


# def _is_dark(gray: np.ndarray) -> bool:
#     h, w = gray.shape
#     c = gray[int(h * 0.30) : int(h * 0.70), int(w * 0.30) : int(w * 0.70)]
#     return (
#         float(np.mean(c)) < CFG["DARK_BG_THRESHOLD"]
#         and float(np.sum(c < CFG["DARK_BG_THRESHOLD"])) / c.size
#         > CFG["DARK_PIXEL_RATIO"]
#     )


# def preprocess(img: np.ndarray) -> tuple:
#     """
#     Returns (processed_gray, is_dark, mode_str).
#     Pipeline: greyscale → invert if dark → CLAHE → unsharp mask → adaptive/grey route.

#     NOTE: PaddleOCR accepts both BGR colour and grayscale numpy arrays.
#     We return a grayscale array here (same as v7) — Paddle handles it fine.
#     """
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     is_dark = _is_dark(gray)
#     if is_dark:
#         gray = cv2.bitwise_not(gray)
#     clahe = cv2.createCLAHE(clipLimit=CFG["CLAHE_CLIP"], tileGridSize=CFG["CLAHE_TILE"])
#     gray = clahe.apply(gray)
#     blur = cv2.GaussianBlur(gray, (0, 0), 3)
#     gray = cv2.addWeighted(
#         gray, 1 + CFG["UNSHARP_AMOUNT"], blur, -CFG["UNSHARP_AMOUNT"], 0
#     )
#     std = float(np.std(gray))
#     if std < CFG["STD_DEV_ADAPTIVE"]:
#         mode = f"ADAPTIVE(σ={std:.1f})"
#         gray = cv2.adaptiveThreshold(
#             gray,
#             255,
#             cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#             cv2.THRESH_BINARY,
#             blockSize=21,
#             C=10,
#         )
#     else:
#         mode = f"CLAHE+USM(σ={std:.1f},dark={is_dark})"
#     return gray, is_dark, mode


# # ══════════════════════════════════════════════════════════════════════════════
# #  LAYER 2 — OCR  (PaddleOCR primary + Tesseract fallback)
# # ══════════════════════════════════════════════════════════════════════════════


# def _paddle_result_to_tokens(result) -> list:
#     """
#     Convert PaddleOCR result format to our standard (bbox, text, conf) tuples.

#     PaddleOCR returns:  [ [ [bbox, (text, conf)], ... ] ]
#     where bbox = [[x0,y0],[x1,y0],[x1,y1],[x0,y1]]  (4 corner points)
#     """
#     tokens = []
#     if result is None:
#         return tokens
#     # Paddle returns a list of pages; we always pass single images
#     for page in result:
#         if page is None:
#             continue
#         for item in page:
#             if item is None or len(item) < 2:
#                 continue
#             bbox = item[0]  # [[x0,y0],[x1,y0],[x1,y1],[x0,y1]]
#             text_conf = item[1]  # (text, confidence)
#             if text_conf is None:
#                 continue
#             text = str(text_conf[0]).strip()
#             conf = float(text_conf[1]) if text_conf[1] is not None else 0.0
#             if text:
#                 tokens.append((bbox, text, conf))
#     return tokens


# def _ocr_paddle(proc: np.ndarray) -> list:
#     """
#     Triple-pass PaddleOCR:
#       Pass 1 — preprocessed image (normal)
#       Pass 2 — brightened  (+25 brightness, ×1.35 contrast)
#       Pass 3 — contrast-boosted (higher contrast, slight sharpening)

#     IOU grid-key dedup keeps the highest-confidence reading per cell.
#     """
#     reader = get_paddle_reader()
#     if reader is None:
#         return []

#     pool = {}

#     def _run_pass(arr):
#         # PaddleOCR accepts numpy array directly (BGR or grayscale)
#         try:
#             result = reader.ocr(arr, cls=True)
#             for bbox, text, conf in _paddle_result_to_tokens(result):
#                 k = _grid_key(bbox)
#                 if k not in pool or conf > pool[k][2]:
#                     pool[k] = (bbox, text, conf)
#         except Exception as e:
#             print(f"  [WARN] PaddleOCR pass error: {e}")

#     # Pass 1: standard preprocessed image
#     _run_pass(proc)

#     # Pass 2: brightened — helps faint light-coloured text on pale backgrounds
#     bright = cv2.convertScaleAbs(proc, alpha=1.35, beta=25)
#     _run_pass(bright)

#     # Pass 3: contrast-boosted — helps dense small text
#     contrast = cv2.convertScaleAbs(proc, alpha=1.6, beta=-20)
#     _run_pass(contrast)

#     return list(pool.values())


# def _ocr_tesseract(proc: np.ndarray) -> list:
#     """Tesseract fallback — supplements PaddleOCR, same as v7."""
#     if not TESS_OK:
#         return []
#     pil = Image.fromarray(proc)
#     pool = {}
#     for psm in [6, 11, 3]:
#         try:
#             data = pytesseract.image_to_data(
#                 pil, config=f"--psm {psm} --oem 3", output_type=pytesseract.Output.DICT
#             )
#         except Exception:
#             continue
#         for i in range(len(data["text"])):
#             text = data["text"][i].strip()
#             if not text:
#                 continue
#             rc = data["conf"][i]
#             if isinstance(rc, str):
#                 rc = float(rc) if rc.strip() not in ("-1", "") else 0.0
#             conf = max(0.0, float(rc) / 100.0)
#             x, y, w, h = (
#                 data["left"][i],
#                 data["top"][i],
#                 data["width"][i],
#                 data["height"][i],
#             )
#             if w < 4 or h < 4:
#                 continue
#             bbox = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
#             k = _grid_key(bbox)
#             if k not in pool or conf > pool[k][2]:
#                 pool[k] = (bbox, text, conf)
#     return list(pool.values())


# def _merge_engines(paddle: list, tess: list) -> list:
#     """Merge PaddleOCR and Tesseract results; highest confidence wins per grid cell."""
#     merged = {}
#     for bbox, text, conf in paddle + tess:
#         k = _grid_key(bbox)
#         if k not in merged or conf > merged[k][2]:
#             merged[k] = (bbox, text, conf)
#     return list(merged.values())


# def _iou(b1, b2) -> float:
#     l1, t1 = _left_x(b1), _top_y(b1)
#     r1, bo1 = l1 + _width(b1), t1 + _height(b1)
#     l2, t2 = _left_x(b2), _top_y(b2)
#     r2, bo2 = l2 + _width(b2), t2 + _height(b2)
#     ix = max(0.0, min(r1, r2) - max(l1, l2))
#     iy = max(0.0, min(bo1, bo2) - max(t1, t2))
#     inter = ix * iy
#     if not inter:
#         return 0.0
#     union = _width(b1) * _height(b1) + _width(b2) * _height(b2) - inter
#     return inter / union if union else 0.0


# def iou_dedup(tokens: list) -> list:
#     """
#     IOU-based spatial dedup: removes ghost tokens from triple-pass overlaps.
#     Higher-confidence token wins when IOU > threshold.
#     """
#     tokens = sorted(tokens, key=lambda x: -x[2])
#     kept = []
#     for bbox, text, conf in tokens:
#         if not any(_iou(bbox, kb) > CFG["IOU_THRESH"] for kb, _, _ in kept):
#             kept.append((bbox, text, conf))
#     return kept


# def merge_fragments(tokens: list) -> list:
#     """
#     Horizontal fragment merger: joins tokens with small inter-token gaps.
#     Fixes: 'FOUNDE'+'R' → 'FOUNDER',  'A'+'Ditya' → 'ADitya'
#     PaddleOCR fragments less than EasyOCR but still benefits from this pass.
#     """
#     if len(tokens) < 2:
#         return tokens
#     heights = [_height(b) for b, _, _ in tokens]
#     med_h = float(np.median(heights)) if heights else 20
#     y_tol = max(18, int(med_h * 0.55))
#     sorted_t = sorted(tokens, key=lambda r: (_mid_y(r[0]), _left_x(r[0])))
#     raw_lines, cur = [], [sorted_t[0]]
#     for item in sorted_t[1:]:
#         if abs(_mid_y(item[0]) - _mid_y(cur[-1][0])) <= y_tol:
#             cur.append(item)
#         else:
#             raw_lines.append(cur)
#             cur = [item]
#     raw_lines.append(cur)
#     out = []
#     for rl in raw_lines:
#         sl = sorted(rl, key=lambda r: _left_x(r[0]))
#         ab, at, ac = sl[0]
#         for cb, ct, cc in sl[1:]:
#             gap = _left_x(cb) - (_left_x(ab) + _width(ab))
#             char_w = _width(ab) / max(len(at.strip()), 1)
#             if gap < char_w * CFG["HMF_GAP_FACTOR"]:
#                 sep = "" if gap < char_w * 0.3 else " "
#                 x0 = min(_left_x(ab), _left_x(cb))
#                 y0 = min(ab[0][1], cb[0][1])
#                 x1 = max(_left_x(ab) + _width(ab), _left_x(cb) + _width(cb))
#                 y1 = max(ab[2][1], cb[2][1])
#                 ab = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
#                 at = at + sep + ct
#                 ac = (ac + cc) / 2.0
#             else:
#                 out.append((ab, at, ac))
#                 ab, at, ac = cb, ct, cc
#         out.append((ab, at, ac))
#     return out


# # ══════════════════════════════════════════════════════════════════════════════
# #  TOKEN CLEANING
# # ══════════════════════════════════════════════════════════════════════════════


# def collapse_spaced(text: str) -> str:
#     """'A D I T Y A' → 'ADITYA'"""
#     segs = text.strip().split()
#     if len(segs) >= 2 and all(re.fullmatch(r"[A-Za-z]\.?", s) for s in segs):
#         return "".join(segs)
#     return text


# _DIGIT_MAP = str.maketrans(
#     {
#         "O": "0",
#         "o": "0",
#         "I": "1",
#         "l": "1",
#         "S": "5",
#         "s": "5",
#         "B": "8",
#         "G": "6",
#         "g": "9",
#         "Z": "2",
#         "z": "2",
#     }
# )


# def fix_digits(text: str) -> str:
#     return text.translate(_DIGIT_MAP)


# def clean_text(text: str) -> str:
#     return re.sub(r" {2,}", " ", unicodedata.normalize("NFKC", text)).strip()


# def smart_title(text: str) -> str:
#     """'ADitya VaRMa' → 'Aditya Varma'"""
#     return " ".join(w.title() for w in text.split())


# def fix_doubled_end(text: str) -> str:
#     """'FOUNDERR' → 'FOUNDER'  (OCR duplication artefact)"""

#     def _fix(w):
#         if len(w) > 4 and w[-1] == w[-2] and w[-1].lower() not in "aeiou":
#             return w[:-1]
#         return w

#     return " ".join(_fix(w) for w in text.split())


# # ══════════════════════════════════════════════════════════════════════════════
# #  LINE GROUPING
# # ══════════════════════════════════════════════════════════════════════════════


# def group_lines(tokens: list) -> list:
#     if not tokens:
#         return []
#     heights = [_height(b) for b, _, _ in tokens]
#     med_h = float(np.median(heights)) if heights else 20
#     y_tol = max(CFG["LINE_Y_BASE_TOL"], int(med_h * 0.65))
#     sorted_t = sorted(tokens, key=lambda r: (_mid_y(r[0]), _left_x(r[0])))
#     lines, cur = [], [sorted_t[0]]
#     for item in sorted_t[1:]:
#         if abs(_mid_y(item[0]) - _mid_y(cur[0][0])) <= y_tol:
#             cur.append(item)
#         else:
#             lines.append(sorted(cur, key=lambda r: _left_x(r[0])))
#             cur = [item]
#     lines.append(sorted(cur, key=lambda r: _left_x(r[0])))
#     return lines


# def line_text(line: list, sep: str = " ") -> str:
#     return sep.join(
#         p
#         for p in [collapse_spaced(clean_text(t)) for _, t, _ in line if t.strip()]
#         if p
#     )


# def line_conf(line: list) -> float:
#     return sum(c for _, _, c in line) / len(line) if line else 0.0


# # ══════════════════════════════════════════════════════════════════════════════
# #  LAYER 3 — FIELD EXTRACTION  (Regex primary + phonenumbers + spaCy NER)
# # ══════════════════════════════════════════════════════════════════════════════

# # ── 3a EMAIL ──────────────────────────────────────────────────────────────────


# def _find_email(text: str) -> str:
#     compact = re.sub(r"\s+", "", text)
#     m = _EMAIL_RE.search(compact)
#     if m:
#         return re.sub(r"\s+", "", m.group()).lower()
#     for sub in [r"\(a\)", r"\[at\]", r"\bat\b"]:
#         t2 = re.sub(r"\s+", "", re.sub(sub, "@", text, flags=re.IGNORECASE))
#         m = _EMAIL_RE.search(t2)
#         if m:
#             return re.sub(r"\s+", "", m.group()).lower()
#     m = _EMAIL_RE.search(compact.replace(",", "."))
#     if m:
#         return re.sub(r"\s+", "", m.group()).lower()
#     m = _EMAIL_AT_RE.search(compact)
#     if m:
#         raw = re.sub(r"\s+", "", m.group()).lower()
#         return re.sub(r"\(a\)|\[at\]|(?<=\w)at(?=\w)", "@", raw)
#     return ""


# def extract_email(lines: list) -> str:
#     hits = []
#     for line in lines:
#         r = _find_email(line_text(line))
#         if r:
#             hits.append((line_conf(line), r))
#         for _, t, c in line:
#             r = _find_email(t)
#             if r:
#                 hits.append((c, r))
#     if not hits:
#         blob = " ".join(line_text(ln) for ln in lines)
#         r = _find_email(blob)
#         if r:
#             hits.append((0.5, r))
#     return max(hits, key=lambda x: x[0])[1] if hits else ""


# # ── 3b PHONES  (phonenumbers.PhoneNumberMatcher) ─────────────────────────────


# def extract_phones(lines: list) -> list:
#     """
#     Uses phonenumbers.PhoneNumberMatcher for validated E.164-formatted numbers.
#     Falls back to regex + strict digit count when library is unavailable.
#     """
#     full_text = "\n".join(fix_digits(line_text(ln)) for ln in lines)
#     found = []

#     if PH_OK:
#         for match in PhoneNumberMatcher(full_text, CFG["PHONE_DEFAULT_REGION"]):
#             num = match.number
#             if phonenumbers.is_valid_number(num):
#                 formatted = phonenumbers.format_number(num, PhoneNumberFormat.E164)
#                 if formatted not in found:
#                     found.append(formatted)
#     else:
#         for line in lines:
#             txt = fix_digits(line_text(line))
#             for m in _PHONE_RE.finditer(txt):
#                 raw = m.group()
#                 digits = re.sub(r"\D", "", raw)
#                 if (
#                     len(digits) < CFG["PHONE_MIN_DIGITS"]
#                     or len(digits) > CFG["PHONE_MAX_DIGITS"]
#                 ):
#                     continue
#                 has_fmt = any(c in raw for c in "+()") or bool(
#                     re.search(r"\d[\-\.]\d", raw)
#                 )
#                 if not has_fmt and len(digits) < 10:
#                     continue
#                 norm = re.sub(r"(?<=\d) (?=\d)", "", raw).strip()
#                 if norm and norm not in found:
#                     found.append(norm)
#     return found


# # ── 3c WEBSITE  (tldextract validation) ──────────────────────────────────────


# def extract_website(lines: list) -> str:
#     for line in lines:
#         txt = line_text(line)
#         m = _WEB_RE.search(txt)
#         if m:
#             url = m.group().strip().rstrip(",.")
#             if TLD_OK:
#                 ext = tldextract.extract(url)
#                 if not (ext.domain and ext.suffix):
#                     continue
#             return ("http://" + url if not url.startswith("http") else url).lower()
#     return ""


# # ── 3d NAME  (font-size scoring + spaCy NER supplement) ──────────────────────


# def extract_name(lines: list, raw_tokens: list) -> str:
#     """
#     Primary: font-size scoring (largest text = name).
#     Supplement: spaCy PERSON entities fill the gap when scoring fails.
#     Applies smart_title() to fix 'ADitya VaRMa' → 'Aditya Varma'.
#     """
#     name = _name_by_font_score(lines, raw_tokens)
#     if not name:
#         name = _name_by_spacy(lines)
#     return name


# def _name_by_font_score(lines, raw_tokens) -> str:
#     if not raw_tokens:
#         return ""
#     card_h = max((_top_y(b) + _height(b)) for b, _, _ in raw_tokens) or 1

#     def _ok(bbox, text, conf):
#         if conf < CFG["OCR_CONF_THRESH"]:
#             return False
#         t = text.strip()
#         if len(t) < 2 or re.search(r"\d", t) or "@" in t:
#             return False
#         if _COMPANY_KW.search(t) or _TITLE_KW.search(t) or _ADDR_KW.search(t):
#             return False
#         return (
#             sum(c.isalpha() for c in t) / max(len(t), 1) >= CFG["NAME_MIN_ALPHA_RATIO"]
#         )

#     cands = [(b, t, c) for b, t, c in raw_tokens if _ok(b, t, c)]
#     if not cands:
#         return ""
#     cand_lines = group_lines(cands)

#     def _score(ln):
#         avg_h = float(np.mean([_height(b) for b, _, _ in ln]))
#         avg_c = line_conf(ln)
#         avg_y = float(np.mean([_top_y(b) for b, _, _ in ln]))
#         return avg_h * avg_c * (1 + 0.30 * (1.0 - avg_y / card_h))

#     best = max(cand_lines, key=_score)
#     parts = []
#     for _, t, _ in sorted(best, key=lambda r: _left_x(r[0])):
#         p = collapse_spaced(clean_text(t))
#         parts.append(p)
#     name = " ".join(parts)
#     name = re.sub(r"^[^A-Za-z]+|[^A-Za-z.'\- ]+$", "", name).strip()
#     return smart_title(name)


# def _name_by_spacy(lines) -> str:
#     nlp = get_spacy_nlp()
#     if not nlp:
#         return ""
#     blob = "\n".join(line_text(ln) for ln in lines)
#     doc = nlp(blob)
#     for ent in doc.ents:
#         if ent.label_ == "PERSON":
#             t = ent.text.strip()
#             if (
#                 len(t) >= 3
#                 and not re.search(r"\d", t)
#                 and not _COMPANY_KW.search(t)
#                 and not _TITLE_KW.search(t)
#             ):
#                 return smart_title(t)
#     return ""


# # ── 3e COMPANY  (keyword scoring + spaCy ORG supplement) ─────────────────────


# def extract_company(lines: list) -> str:
#     company = _company_by_keywords(lines)
#     if not company:
#         company = _company_by_spacy(lines)
#     return company


# def _company_by_keywords(lines) -> str:
#     scored = []
#     for line in lines:
#         t = line_text(line).strip()
#         if not t:
#             continue
#         score = 0
#         if _COMPANY_KW.search(t):
#             score += 10
#         if t.isupper() and len(t.split()) >= 2:
#             score += 3
#         if len(t.split()) >= 2:
#             score += 1
#         if "@" in t or re.search(r"\d{4,}", t):
#             score -= 10
#         if _ADDR_KW.search(t):
#             score -= 5
#         if _TITLE_KW.search(t):
#             score -= 4
#         if score > 0:
#             scored.append((score, line_conf(line), t))
#     if not scored:
#         return ""
#     scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
#     return scored[0][2]


# def _company_by_spacy(lines) -> str:
#     nlp = get_spacy_nlp()
#     if not nlp:
#         return ""
#     blob = "\n".join(line_text(ln) for ln in lines)
#     doc = nlp(blob)
#     for ent in doc.ents:
#         if ent.label_ == "ORG" and len(ent.text.strip()) > 3:
#             return ent.text.strip()
#     return ""


# # ── 3f JOB TITLE ─────────────────────────────────────────────────────────────


# def extract_job_title(lines: list) -> str:
#     for line in lines:
#         t = line_text(line).strip()
#         if _TITLE_KW.search(t) and not _COMPANY_KW.search(t):
#             return fix_doubled_end(t)
#     return ""


# # ── 3g ADDRESS  (structural + phone-aware) ────────────────────────────────────


# def extract_address(lines: list, known_phones: list = None) -> str:
#     known_phones = known_phones or []
#     parts, seen = [], set()
#     for line in lines:
#         t = line_text(line).strip()
#         if not t:
#             continue
#         is_phone_line = any(ph in t for ph in known_phones if len(ph) > 8)
#         if is_phone_line:
#             for ph in known_phones:
#                 t = t.replace(ph, "").strip(" ,")
#             t = re.sub(r"\+?\d[\d\s\-\.\(\)]{7,}", "", t).strip(" ,")
#             if not t:
#                 continue
#         is_addr = bool(_ADDR_KW.search(t))
#         if _POSTAL_RE.search(t) and not re.search(r"[+\-]\s*\d{6,}", t):
#             is_addr = True
#         if re.search(r"[A-Z][a-z]+,\s*[A-Z][a-z]+", t):
#             is_addr = True
#         if is_addr:
#             clean = t
#             for ph in known_phones:
#                 clean = clean.replace(ph, "")
#             clean = re.sub(r"\+?\d[\d\s\-\.\(\)]{7,}", "", clean)
#             clean = re.sub(r"\s{2,}", " ", clean).strip(" ,")
#             if clean and clean.lower() not in seen:
#                 seen.add(clean.lower())
#                 parts.append(clean)
#     return ", ".join(parts)


# # ── 3h SOCIAL ─────────────────────────────────────────────────────────────────


# def extract_social(lines: list) -> dict:
#     r = {"linkedin": "", "twitter": ""}
#     for line in lines:
#         t = line_text(line)
#         if not r["linkedin"]:
#             m = _LINKEDIN_RE.search(t)
#             if m:
#                 r["linkedin"] = m.group().strip()
#         if not r["twitter"]:
#             m = _TWITTER_RE.search(t)
#             if m:
#                 r["twitter"] = m.group().strip()
#     return r


# # ── Quality label ──────────────────────────────────────────────────────────────


# def quality_label(r: dict) -> str:
#     """GREEN ≥5 fields, YELLOW ≥3, RED <3"""
#     fields = [
#         r.get("name", ""),
#         r.get("company", ""),
#         r.get("email", ""),
#         r["phones"][0] if r.get("phones") else "",
#         r.get("address", ""),
#     ]
#     filled = sum(bool(v.strip()) for v in fields)
#     if filled >= 5:
#         return "GREEN"
#     if filled >= 3:
#         return "YELLOW"
#     return "RED"


# # ══════════════════════════════════════════════════════════════════════════════
# #  LAYER 4 — LLM POST-PROCESSING  (optional)
# # ══════════════════════════════════════════════════════════════════════════════


# def llm_correct_and_extract(raw_lines: list, hf_token: str) -> dict | None:
#     """
#     Sends OCR text through Llama-3.1-8B for:
#       1. OCR error correction (0↔O, l↔1, space removal)
#       2. Structured JSON extraction

#     Returns a flat dict matching our schema, or None on failure.
#     LLM fields override regex fields only when the LLM field is non-empty.
#     """
#     if not LLM_OK:
#         print("  [WARN] LangChain/HuggingFace not installed — skipping LLM step")
#         return None

#     raw_text = "\n".join(raw_lines)

#     try:
#         llm = HuggingFaceEndpoint(
#             repo_id=CFG["LLM_REPO_ID"],
#             temperature=CFG["LLM_TEMPERATURE"],
#             huggingfacehub_api_token=hf_token,
#             timeout=CFG["LLM_TIMEOUT"],
#         )
#         model = ChatHuggingFace(llm=llm)

#         corr_prompt = ChatPromptTemplate.from_messages(
#             [
#                 (
#                     "system",
#                     "You are an OCR post-processor. Fix character confusions (0/O, 1/l/I, S/5). "
#                     "Join broken emails (a @ b . com → a@b.com). Separate fields jumbled on one line. "
#                     "Return ONLY the corrected text, nothing else.",
#                 ),
#                 ("human", "Correct:\n{raw_text}"),
#             ]
#         )
#         corrected = (corr_prompt | model).invoke({"raw_text": raw_text}).content

#         ext_prompt = ChatPromptTemplate.from_messages(
#             [
#                 (
#                     "system",
#                     "Extract visiting card fields. Return ONLY strict JSON with keys: "
#                     "name, job_title, company, email, phone_list (array), website, address, linkedin, twitter. "
#                     "Empty string for missing fields. No markdown, no explanation.",
#                 ),
#                 ("human", "{text}"),
#             ]
#         )
#         ai = (ext_prompt | model | JsonOutputParser()).invoke({"text": corrected})
#         return {
#             "name": ai.get("name", ""),
#             "job_title": ai.get("job_title", ""),
#             "company": ai.get("company", ""),
#             "email": ai.get("email", ""),
#             "phones": (
#                 ai.get("phone_list", [])
#                 if isinstance(ai.get("phone_list"), list)
#                 else []
#             ),
#             "website": ai.get("website", ""),
#             "address": ai.get("address", ""),
#             "linkedin": ai.get("linkedin", ""),
#             "twitter": ai.get("twitter", ""),
#             "_llm_corrected_text": corrected,
#         }
#     except Exception as e:
#         err = str(e)
#         if "401" in err or "Unauthorized" in err:
#             print("  [ERROR] Invalid HuggingFace token")
#         elif "429" in err or "Rate limit" in err:
#             print(
#                 "  [ERROR] HuggingFace rate limit — create a new free account for fresh quota"
#             )
#         else:
#             print(f"  [WARN] LLM step failed: {e}")
#         return None


# def _merge_llm(base: dict, llm: dict) -> dict:
#     """LLM values override base only when LLM value is non-empty."""
#     for key in [
#         "name",
#         "job_title",
#         "company",
#         "email",
#         "website",
#         "address",
#         "linkedin",
#         "twitter",
#     ]:
#         if llm.get(key, "").strip():
#             base[key] = llm[key]
#     if llm.get("phones"):
#         base["phones"] = llm["phones"]
#     return base


# # ══════════════════════════════════════════════════════════════════════════════
# #  MAIN PIPELINE
# # ══════════════════════════════════════════════════════════════════════════════


# def process_card(image_path: str, debug: bool = False, hf_token: str = None) -> dict:
#     """
#     Full pipeline for one card image.
#     Returns a dict with all extracted fields + quality label.
#     """
#     result = {
#         "file": os.path.basename(image_path),
#         "name": "",
#         "job_title": "",
#         "company": "",
#         "email": "",
#         "phones": [],
#         "website": "",
#         "address": "",
#         "linkedin": "",
#         "twitter": "",
#         "quality": "RED",
#         "_debug": {},
#     }

#     try:
#         # L1: CV pipeline
#         img = resize_for_ocr(load_image(image_path))
#         img, skew = deskew(img)
#         proc, is_dark, mode = preprocess(img)

#         if debug:
#             dbg_dir = os.path.dirname(os.path.abspath(image_path))
#             cv2.imwrite(os.path.join(dbg_dir, "DEBUG_v8_preprocessed.png"), proc)
#             result["_debug"].update(
#                 {"skew": round(skew, 2), "dark": is_dark, "mode": mode}
#             )

#         # L2: PaddleOCR + Tesseract → IOU dedup → fragment merge → confidence filter
#         paddle_tokens = _ocr_paddle(proc)
#         tess_tokens = _ocr_tesseract(proc)
#         tokens = _merge_engines(paddle_tokens, tess_tokens)
#         tokens = iou_dedup(tokens)
#         tokens = merge_fragments(tokens)
#         kept = [
#             (b, t, c) for b, t, c in tokens if c >= CFG["OCR_CONF_THRESH"] and t.strip()
#         ]

#         if debug:
#             result["_debug"]["total"] = len(tokens)
#             result["_debug"]["kept"] = len(kept)
#             result["_debug"]["paddle"] = len(paddle_tokens)
#             result["_debug"]["tess"] = len(tess_tokens)
#             result["_debug"]["raw"] = [
#                 {"text": t, "conf": round(c, 3), "y": round(_top_y(b), 1)}
#                 for b, t, c in sorted(tokens, key=lambda r: _top_y(r[0]))
#             ]
#             _annotate(img, tokens, image_path)

#         lines = group_lines(kept)

#         if debug:
#             result["_debug"]["lines"] = [
#                 {"n": i + 1, "text": line_text(ln), "conf": round(line_conf(ln), 3)}
#                 for i, ln in enumerate(lines)
#             ]

#         # L3: Field extraction
#         result["email"] = extract_email(lines)
#         result["phones"] = extract_phones(lines)
#         result["website"] = extract_website(lines)
#         result["name"] = extract_name(lines, kept)
#         result["company"] = extract_company(lines)
#         result["job_title"] = extract_job_title(lines)
#         result["address"] = extract_address(lines, result["phones"])
#         soc = extract_social(lines)
#         result["linkedin"] = soc["linkedin"]
#         result["twitter"] = soc["twitter"]

#         # L4: LLM override (optional)
#         if hf_token:
#             raw_lines = [line_text(ln) for ln in lines]
#             llm_data = llm_correct_and_extract(raw_lines, hf_token)
#             if llm_data:
#                 result = _merge_llm(result, llm_data)

#         result["quality"] = quality_label(result)

#     except Exception as exc:
#         print(f"  [ERROR] {image_path}: {exc}")
#         if debug:
#             traceback.print_exc()

#     return result


# def _annotate(img: np.ndarray, tokens: list, src: str):
#     ann = img.copy()
#     for bbox, text, conf in tokens:
#         pts = np.array([[int(p[0]), int(p[1])] for p in bbox], np.int32)
#         color = (0, 200, 0) if conf >= CFG["OCR_CONF_THRESH"] else (0, 0, 200)
#         cv2.polylines(ann, [pts], True, color, 2)
#         label = f"{text[:24]}{'…' if len(text) > 24 else ''} [{conf:.2f}]"
#         pos = (max(pts[0][0], 0), max(pts[0][1] - 5, 12))
#         (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
#         cv2.rectangle(
#             ann, (pos[0], pos[1] - lh - 2), (pos[0] + lw, pos[1] + 2), (20, 20, 20), -1
#         )
#         cv2.putText(
#             ann, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA
#         )
#     out = os.path.join(os.path.dirname(os.path.abspath(src)), "DEBUG_v8_annotated.png")
#     cv2.imwrite(out, ann)
#     print(f"  [DEBUG] Annotated → {out}")


# # ══════════════════════════════════════════════════════════════════════════════
# #  PDF SUPPORT
# # ══════════════════════════════════════════════════════════════════════════════


# def _find_poppler():
#     if CFG["POPPLER_PATH"]:
#         return CFG["POPPLER_PATH"]
#     for c in [
#         r"C:\poppler\Library\bin",
#         r"C:\Program Files\poppler\Library\bin",
#         r"C:\poppler\bin",
#     ]:
#         if os.path.isdir(c):
#             return c
#     return None


# def pdf_to_images(pdf_path: str) -> list:
#     if not PDF_OK:
#         print("  [WARN] pip install pdf2image")
#         return []
#     kw = {"dpi": CFG["PDF_DPI"]}
#     pp = _find_poppler()
#     if pp:
#         kw["poppler_path"] = pp
#     pages = convert_from_path(pdf_path, **kw)
#     return [cv2.cvtColor(np.array(p), cv2.COLOR_RGB2BGR) for p in pages]


# # ══════════════════════════════════════════════════════════════════════════════
# #  OUTPUT
# # ══════════════════════════════════════════════════════════════════════════════


# def print_result(r: dict):
#     W = 65
#     print()
#     print("═" * W)
#     print(f"  FILE      : {r['file']}")
#     print("─" * W)
#     print(f"  NAME      : {r['name']      or '—'}")
#     print(f"  JOB TITLE : {r['job_title'] or '—'}")
#     print(f"  COMPANY   : {r['company']   or '—'}")
#     print(f"  EMAIL     : {r['email']     or '—'}")
#     for i, ph in enumerate(r["phones"], 1):
#         print(f"  PHONE #{i:<3} : {ph}")
#     if not r["phones"]:
#         print(f"  PHONE     : —")
#     print(f"  WEBSITE   : {r['website']   or '—'}")
#     print(f"  ADDRESS   : {r['address']   or '—'}")
#     print(f"  LINKEDIN  : {r['linkedin']  or '—'}")
#     print(f"  TWITTER/X : {r['twitter']   or '—'}")
#     ql = r.get("quality", "—")
#     symbol = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}.get(ql, "")
#     print(f"  QUALITY   : {symbol} {ql}")
#     print("═" * W)
#     if r.get("_debug"):
#         d = r["_debug"]
#         print(
#             f"\n  [DEBUG] skew={d.get('skew')}°  dark={d.get('dark')}  mode={d.get('mode')}"
#         )
#         print(
#             f"  [DEBUG] paddle={d.get('paddle')}  tess={d.get('tess')}  "
#             f"total={d.get('total')}  kept={d.get('kept')}"
#         )
#         if d.get("raw"):
#             print("\n  TOKENS (after dedup+merge):")
#             for tk in d["raw"]:
#                 mark = "  " if tk["conf"] >= CFG["OCR_CONF_THRESH"] else "✗ "
#                 print(
#                     f"  {mark}y={tk['y']:>6.1f}  conf={tk['conf']:.3f}  {tk['text']!r}"
#                 )
#         if d.get("lines"):
#             print("\n  LINES:")
#             for ln in d["lines"]:
#                 print(f"  Line {ln['n']:02d} (conf={ln['conf']:.2f}) : {ln['text']!r}")
#         print()


# def _ts():
#     return datetime.now().strftime("%Y%m%d_%H%M%S")


# def save_json(results: list, out_dir: str):
#     path = os.path.join(out_dir, f"cards_{_ts()}.json")
#     clean = [{k: v for k, v in r.items() if k != "_debug"} for r in results]
#     with open(path, "w", encoding="utf-8") as f:
#         json.dump(clean, f, indent=2, ensure_ascii=False)
#     print(f"  [JSON] → {path}")


# def save_csv(results: list, out_dir: str):
#     path = os.path.join(out_dir, f"cards_{_ts()}.csv")
#     fields = [
#         "file",
#         "name",
#         "job_title",
#         "company",
#         "email",
#         "phone_1",
#         "phone_2",
#         "website",
#         "address",
#         "linkedin",
#         "twitter",
#         "quality",
#     ]
#     with open(path, "w", newline="", encoding="utf-8") as f:
#         w = csv.DictWriter(f, fieldnames=fields)
#         w.writeheader()
#         for r in results:
#             ph = r.get("phones", [])
#             w.writerow(
#                 {
#                     "file": r.get("file", ""),
#                     "name": r.get("name", ""),
#                     "job_title": r.get("job_title", ""),
#                     "company": r.get("company", ""),
#                     "email": r.get("email", ""),
#                     "phone_1": ph[0] if ph else "",
#                     "phone_2": ph[1] if len(ph) > 1 else "",
#                     "website": r.get("website", ""),
#                     "address": r.get("address", ""),
#                     "linkedin": r.get("linkedin", ""),
#                     "twitter": r.get("twitter", ""),
#                     "quality": r.get("quality", "RED"),
#                 }
#             )
#     print(f"  [CSV]  → {path}")


# def save_excel(results: list, out_dir: str):
#     if not EXCEL_OK:
#         print("  [WARN] pip install openpyxl pandas")
#         return
#     path = os.path.join(out_dir, f"cards_{_ts()}.xlsx")
#     wb = Workbook()
#     ws = wb.active
#     ws.title = "Visiting Cards"

#     NAV = "1F4E79"
#     WHT = "FFFFFF"
#     ODD = "EBF3FB"
#     EVN = "FFFFFF"
#     BDR = "BDD7EE"
#     GFILL = PatternFill("solid", fgColor="C6EFCE")
#     YFILL = PatternFill("solid", fgColor="FFEB9C")
#     RFILL = PatternFill("solid", fgColor="FFC7CE")
#     QFILL = {"GREEN": GFILL, "YELLOW": YFILL, "RED": RFILL}

#     thin = Side(style="thin", color=BDR)
#     bdr = Border(left=thin, right=thin, top=thin, bottom=thin)

#     COLS = [
#         ("File", "file", 28),
#         ("Name", "name", 22),
#         ("Job Title", "job_title", 30),
#         ("Company", "company", 28),
#         ("Email", "email", 32),
#         ("Phone 1", "phone_1", 22),
#         ("Phone 2", "phone_2", 22),
#         ("Website", "website", 30),
#         ("Address", "address", 42),
#         ("LinkedIn", "linkedin", 35),
#         ("Twitter/X", "twitter", 22),
#         ("Quality", "quality", 12),
#     ]

#     hf = PatternFill("solid", fgColor=NAV)
#     hfont = Font(name="Arial", bold=True, color=WHT, size=11)
#     halign = Alignment(horizontal="center", vertical="center")

#     for ci, (label, _, _) in enumerate(COLS, 1):
#         c = ws.cell(row=1, column=ci, value=label)
#         c.fill = hf
#         c.font = hfont
#         c.alignment = halign
#         c.border = bdr
#     ws.row_dimensions[1].height = 22

#     dfont = Font(name="Arial", size=10)
#     dalign = Alignment(vertical="center", wrap_text=False)

#     for ri, r in enumerate(results, 2):
#         ph = r.get("phones", [])
#         q = r.get("quality", "RED")
#         rfill = PatternFill("solid", fgColor=ODD if ri % 2 == 0 else EVN)
#         vals = {
#             "file": r.get("file", ""),
#             "name": r.get("name", ""),
#             "job_title": r.get("job_title", ""),
#             "company": r.get("company", ""),
#             "email": r.get("email", ""),
#             "phone_1": ph[0] if ph else "",
#             "phone_2": ph[1] if len(ph) > 1 else "",
#             "website": r.get("website", ""),
#             "address": r.get("address", ""),
#             "linkedin": r.get("linkedin", ""),
#             "twitter": r.get("twitter", ""),
#             "quality": q,
#         }
#         for ci, (_, key, _) in enumerate(COLS, 1):
#             c = ws.cell(row=ri, column=ci, value=vals[key])
#             c.fill = QFILL.get(q, rfill) if key == "quality" else rfill
#             c.font = dfont
#             c.alignment = dalign
#             c.border = bdr
#         ws.row_dimensions[ri].height = 18

#     for ci, (_, _, width) in enumerate(COLS, 1):
#         ws.column_dimensions[get_column_letter(ci)].width = width

#     ws.freeze_panes = "A2"
#     ws.auto_filter.ref = ws.dimensions
#     ws.page_setup.orientation = "landscape"
#     ws.page_setup.fitToPage = True
#     ws.page_setup.fitToWidth = 1
#     ws.page_setup.fitToHeight = 0
#     ws.sheet_view.showGridLines = False

#     wb.save(path)
#     print(f"  [XLSX] → {path}")


# # ══════════════════════════════════════════════════════════════════════════════
# #  ENTRY POINT
# # ══════════════════════════════════════════════════════════════════════════════
# _IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".heic"}


# def collect_paths(target: str) -> list:
#     p = Path(target)
#     if p.is_file():
#         return [str(p)]
#     if p.is_dir():
#         return sorted(
#             str(f)
#             for f in p.iterdir()
#             if f.suffix.lower() in _IMG_EXTS or f.suffix.lower() == ".pdf"
#         )
#     return []


# def gui_pick() -> list:
#     try:
#         from tkinter import Tk, filedialog

#         root = Tk()
#         root.withdraw()
#         paths = filedialog.askopenfilenames(
#             title="Select Card Image(s) or PDF",
#             filetypes=[
#                 ("Images & PDFs", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.webp *.pdf")
#             ],
#         )
#         root.destroy()
#         return list(paths)
#     except Exception:
#         print("  [WARN] Tkinter not available. Pass image path as CLI argument.")
#         return []


# def main():
#     ap = argparse.ArgumentParser(
#         description="Visiting Card OCR Engine v8.0 — PaddleOCR Edition"
#     )
#     ap.add_argument(
#         "target", nargs="?", default=None, help="Image/PDF/folder (blank → GUI)"
#     )
#     ap.add_argument(
#         "--debug",
#         action="store_true",
#         help="Save debug-annotated images and print token table",
#     )
#     ap.add_argument(
#         "--llm",
#         metavar="HF_TOKEN",
#         default=None,
#         help="HuggingFace API token — enables Llama-3.1-8B post-processing",
#     )
#     ap.add_argument(
#         "--lang",
#         metavar="LANG",
#         default="en",
#         help="PaddleOCR language code: en (default), hi (Hindi), "
#         "ml (Malayalam), ch (Chinese), fr, de, etc.",
#     )
#     ap.add_argument("--no-json", action="store_true")
#     ap.add_argument("--no-csv", action="store_true")
#     ap.add_argument("--no-xlsx", action="store_true")
#     args = ap.parse_args()

#     # Apply language override
#     CFG["PADDLE_LANG"] = args.lang

#     print(
#         f"\n  GPU: {'ON' if GPU_OK else 'OFF (CPU)'}  "
#         f"PaddleOCR: {'YES' if PADDLE_OK else 'NO (pip install paddlepaddle paddleocr)'}  "
#         f"Tesseract: {'YES' if TESS_OK else 'NO'}  "
#         f"phonenumbers: {'YES' if PH_OK else 'NO'}  "
#         f"spaCy: {'YES' if SPACY_OK else 'NO'}  "
#         f"LLM: {'YES' if (args.llm and LLM_OK) else 'NO'}"
#     )

#     paths = collect_paths(args.target) if args.target else gui_pick()
#     if not paths:
#         print("  No files found. Exiting.")
#         sys.exit(0)

#     print(f"\n  {len(paths)} file(s) to process.\n")
#     results = []
#     out_dir = os.path.dirname(os.path.abspath(paths[0]))

#     import tempfile

#     for i, path in enumerate(paths, 1):
#         fname = os.path.basename(path)
#         print(f"  [{i}/{len(paths)}] {fname}")

#         if path.lower().endswith(".pdf"):
#             pages = pdf_to_images(path)
#             for pi, page_img in enumerate(pages):
#                 with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
#                     cv2.imwrite(tf.name, page_img)
#                     r = process_card(tf.name, debug=args.debug, hf_token=args.llm)
#                 r["file"] = f"{fname}_page{pi + 1}"
#                 print_result(r)
#                 results.append(r)
#                 os.unlink(tf.name)
#             continue

#         r = process_card(path, debug=args.debug, hf_token=args.llm)
#         print_result(r)
#         results.append(r)

#     if not results:
#         print("  No results.")
#         return

#     green = sum(r["quality"] == "GREEN" for r in results)
#     yellow = sum(r["quality"] == "YELLOW" for r in results)
#     red = sum(r["quality"] == "RED" for r in results)
#     print(f"\n  Summary: 🟢 {green}  🟡 {yellow}  🔴 {red}")

#     if not args.no_xlsx:
#         save_excel(results, out_dir)
#     if not args.no_json:
#         save_json(results, out_dir)
#     if not args.no_csv:
#         save_csv(results, out_dir)
#     print("\n  Done.")


# if __name__ == "__main__":
#     main()


import os
import re
import cv2
import math
import json
import traceback
import numpy as np
import pandas as pd
import torch
import phonenumbers
import tldextract
from datetime import datetime
from rapidfuzz import fuzz
from spellchecker import SpellChecker
from paddleocr import PaddleOCR
from phonenumbers import PhoneNumberMatcher
import spacy

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
CFG = {
    "OCR_CONF_THRESH": 0.18,
    "PADDLE_LANG": "en",
    "DEBUG": True,
}

# ══════════════════════════════════════════════════════════════════════════════
# GLOBALS
# ══════════════════════════════════════════════════════════════════════════════
USE_GPU = torch.cuda.is_available()
SPELL = SpellChecker()

# ══════════════════════════════════════════════════════════════════════════════
# LOAD SPACY
# ══════════════════════════════════════════════════════════════════════════════
try:
    NLP = spacy.load("en_core_web_sm")
    SPACY_OK = True
except Exception:
    NLP = None
    SPACY_OK = False

# ══════════════════════════════════════════════════════════════════════════════
# LOAD PADDLEOCR
# ══════════════════════════════════════════════════════════════════════════════
PADDLE_READER = None


def get_paddle_reader(lang="en"):
    global PADDLE_READER
    if PADDLE_READER is None:
        print(f"[INFO] Loading PaddleOCR ({lang})...")
        PADDLE_READER = PaddleOCR(lang=lang, use_gpu=USE_GPU, show_log=False)
        print("[INFO] PaddleOCR Loaded")
    return PADDLE_READER


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blur)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharp = cv2.filter2D(enhanced, -1, kernel)
    return sharp


# ══════════════════════════════════════════════════════════════════════════════
# DESKEW
# ══════════════════════════════════════════════════════════════════════════════
def deskew(img):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10
    )

    if lines is None:
        return img

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        if abs(angle) < 45:
            angles.append(angle)

    if not angles:
        return img

    median_angle = np.median(angles)
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(
        img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


# ══════════════════════════════════════════════════════════════════════════════
# OCR
# ══════════════════════════════════════════════════════════════════════════════
def run_paddle_ocr(img):
    reader = get_paddle_reader(CFG["PADDLE_LANG"])
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    result = reader.ocr(img)
    tokens = []

    try:
        if result and result[0]:
            for page in result:
                for item in page:
                    bbox = item[0]
                    text = item[1][0]
                    conf = item[1][1]
                    if conf < CFG["OCR_CONF_THRESH"]:
                        continue
                    tokens.append({"text": text.strip(), "conf": conf, "bbox": bbox})
    except Exception as e:
        print(f"[ERROR] OCR Parsing Failed: {e}")
        traceback.print_exc()

    return tokens


# ══════════════════════════════════════════════════════════════════════════════
# RECONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════
def reconstruct(tokens):
    fixed = []
    for t in tokens:
        text = t["text"]
        text = re.sub(r"\s+@\s+", "@", text)
        text = re.sub(r"\s+\.\s+", ".", text)
        if re.fullmatch(r"(?:[A-Z]\s+){2,}[A-Z]", text):
            text = text.replace(" ", "")
        text = re.sub(r"([A-Z]{2,})\s+([A-Z]{2,})", r"\1\2", text)
        fixed.append(text)
    return fixed


# ══════════════════════════════════════════════════════════════════════════════
# EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════
def extract_fields(lines):
    raw_text = "\n".join(lines)
    data = {
        "Name": "",
        "Job_Title": "",
        "Company": "",
        "Email": "",
        "Phone": "",
        "Website": "",
        "Address": "",
        "LinkedIn": "",
        "Twitter": "",
    }

    # EMAIL
    emails = re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", raw_text)
    if emails:
        data["Email"] = emails[0]

    # PHONE
    phones = []
    try:
        for match in PhoneNumberMatcher(raw_text, None):
            phones.append(
                phonenumbers.format_number(
                    match.number, phonenumbers.PhoneNumberFormat.INTERNATIONAL
                )
            )
    except:
        pass
    if phones:
        data["Phone"] = phones[0]

    # WEBSITE
    websites = re.findall(
        r"(?:https?://)?(?:www\.)?[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", raw_text
    )
    websites = [w for w in websites if "@" not in w]
    if websites:
        data["Website"] = websites[0]

    # NLP
    if SPACY_OK and NLP:
        doc = NLP(raw_text)
        for ent in doc.ents:
            if ent.label_ == "PERSON" and not data["Name"]:
                data["Name"] = ent.text
            elif ent.label_ == "ORG" and not data["Company"]:
                data["Company"] = ent.text
            elif ent.label_ in ["GPE", "LOC"] and not data["Address"]:
                data["Address"] = ent.text

    # DESIGNATION
    designations = [
        "CEO",
        "CTO",
        "Founder",
        "Engineer",
        "Manager",
        "Director",
        "Developer",
        "Consultant",
        "Analyst",
    ]
    for line in lines:
        if any(d.lower() in line.lower() for d in designations):
            data["Job_Title"] = line
            break

    # FALLBACK NAME
    if not data["Name"]:
        for line in lines:
            if 2 <= len(line.split()) <= 3:
                if line.isupper() or line.istitle():
                    data["Name"] = line
                    break
    return data


# ══════════════════════════════════════════════════════════════════════════════
# QUALITY SCORE
# ══════════════════════════════════════════════════════════════════════════════
def quality_score(data):
    score = sum(1 for v in data.values() if v)
    if score >= 6:
        return "🟢 GREEN"
    elif score >= 3:
        return "🟡 YELLOW"
    return "🔴 RED"


# ══════════════════════════════════════════════════════════════════════════════
# PROCESS SINGLE CARD
# ══════════════════════════════════════════════════════════════════════════════
def process_card(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError("Image could not be loaded")
    proc = preprocess(img)
    proc = deskew(proc)
    tokens = run_paddle_ocr(proc)
    if CFG["DEBUG"]:
        print(f"[DEBUG] Tokens Detected: {len(tokens)}")
    lines = reconstruct(tokens)
    data = extract_fields(lines)
    data["QUALITY"] = quality_score(data)
    return data


# ══════════════════════════════════════════════════════════════════════════════
# SAVE OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════
def save_outputs(results):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df = pd.DataFrame(results)
    xlsx_path = f"cards_{timestamp}.xlsx"
    json_path = f"cards_{timestamp}.json"
    csv_path = f"cards_{timestamp}.csv"
    df.to_excel(xlsx_path, index=False)
    df.to_json(json_path, orient="records", indent=4)
    df.to_csv(csv_path, index=False)
    print(f"\n[XLSX] → {xlsx_path}\n[JSON] → {json_path}\n[CSV]  → {csv_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    target = input("Enter image path: ").strip()
    if not os.path.exists(target):
        print("[ERROR] File not found")
        return

    try:
        result = process_card(target)
        print("\n" + "═" * 70)
        for k, v in result.items():
            print(f"{k:<12}: {v}")
        print("═" * 70)
        save_outputs([result])
    except Exception as e:
        print(f"[ERROR] {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
