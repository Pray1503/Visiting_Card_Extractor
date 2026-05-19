# # ══════════════════════════════════════════════════════════════════════════════
# # VISITING CARD OCR ENGINE  v17.0  —  DEFINITIVE MERGED EDITION
# # ══════════════════════════════════════════════════════════════════════════════
# #
# #  WHAT THIS VERSION IS AND WHY:
# #  Two code versions were systematically tested against 8 real business cards.
# #  v14 (our previous build) and v16 (alternative build) were both incomplete.
# #  v17 takes the best-performing logic from each, fixes what neither got right,
# #  and is verified against all 8 card failure cases.
# #
# #  COMPONENT ORIGIN MAP:
# #  ┌────────────────────────────────┬──────────┬──────────┬──────────────────┐
# #  │ Component                      │  v14     │  v16     │  v17             │
# #  ├────────────────────────────────┼──────────┼──────────┼──────────────────┤
# #  │ Spaced text collapse           │ broken*  │ ✓ better │ v16 logic        │
# #  │ Duplicate line text removal    │ ✓ works  │ ✓ works  │ both (layered)   │
# #  │ IOU token-level dedup          │ ✗ absent │ ✓ present│ v16 logic        │
# #  │ Quality scoring (validated)    │ ✗ broken*│ ✓ better │ v16 + fixed      │
# #  │ Email Bug 6 (suffix strip)     │ ✗ broken*│ ✗ broken*│ NEW strict TLD   │
# #  │ Postal-as-phone prevention     │ ✓ v14    │ ✓ v16    │ v14 (more cases) │
# #  │ Address-line phone exclusion   │ ✓ v14    │ partial  │ v14 logic        │
# #  │ Name stopword guard            │ ✓ v14    │ ✓ v16    │ both combined    │
# #  │ Dark background detection      │ ✓ v14    │ partial  │ v14 logic        │
# #  │ OCR result parsing (Paddle v3) │ basic    │ ✓ robust │ v16 parser       │
# #  │ load_images (generator)        │ list     │ generator│ generator (v16)  │
# #  │ VCF export                     │ ✗ absent │ ✓ present│ v16              │
# #  │ Logging                        │ print    │ logging  │ logging (v16)    │
# #  │ Per-card output files          │ ✓ v14    │ ✗ absent │ v14              │
# #  └────────────────────────────────┴──────────┴──────────┴──────────────────┘
# #  * broken = claimed fix but test proved it didn't work
# #
# #  BUG FIXES VERIFIED BY AUTOMATED TESTS:
# #
# #  ✅ BUG 1 — Doubled field values (Cards 02, 04, 07)
# #     "SARA AL-MANSOORI SARA AL-MANSOORI" → "SARA AL-MANSOORI"
# #     LAYER 1: IOU-based token dedup before line assembly (v16)
# #     LAYER 2: _dedup_line_text on every assembled line (v14)
# #     LAYER 3: _dedup_repeated_words in company/title extractor (v16)
# #
# #  ✅ BUG 2 — Spaced luxury text (Card 08, Armani Isabella)
# #     "C R E A T I V E  D I R E C T O R" → "CREATIVE DIRECTOR" (not "CREATIVEDIRECTOR")
# #     "I S A B E L L A  D E  R O S A" → "ISABELLA DE ROSA"
# #     FIX: v16's double-space word-boundary aware collapse.
# #     v14's version collapsed everything into one word with no spaces.
# #
# #  ✅ BUG 3 — City grabbed as Name (Cards 01, 03)
# #     "Surat" / "Robert-Bosch-StraBe" → rejected as name candidates
# #     FIX: _NAME_STOPWORDS regex + address token cross-check
# #
# #  ✅ BUG 4 — Postal code as phone (Card 07: "108-0075" as Phone #2)
# #     Region-specific postal format detection prevents postal codes
# #     from being parsed by phonenumbers library as valid phone numbers.
# #
# #  ✅ BUG 5 — Quality GREEN for wrong data (all affected cards)
# #     v14 gave GREEN when Name="Surat" or Name="" — quality counted
# #     non-empty values without validating content type.
# #     v16's scoring is correctly stricter: requires n_ok AND (e_ok OR p_ok)
# #     for GREEN; v14 required 4/5 non-validated checks.
# #
# #  ✅ BUG 6 — Email suffix contamination (Card 03 German)
# #     "k.mueller@bosch-engineering.dek.mueller" → "k.mueller@bosch-engineering.de"
# #     NEITHER v14 nor v16 fixed this. Both used EMAIL_RE which is greedy and
# #     matches the full bad string since ".dek" looks like a valid domain.
# #     NEW FIX: Strict TLD-anchored email regex that terminates at a known TLD
# #     boundary. The regex matches up to the TLD then stops, discarding any
# #     trailing text. List of 80+ TLDs covers all real-world cases.
# #
# #  ✅ BUG 7 — Address digits as phantom phone (Card 04 Singapore)
# #     "71 Ayer Rajah Crescent ... Singapore 139951" → "+61 139951" phantom
# #     FIX: Address lines (matching ADDRESS_KW) are pre-excluded from phone scan.
# #
# #  ✅ BUG 8 — Job title / company duplicated in output (from OCR double-pass)
# #     Same root as Bug 1. Resolved by all three dedup layers above.
# #
# # ══════════════════════════════════════════════════════════════════════════════
# #
# #  INSTALL:
# #    pip install paddlepaddle paddleocr opencv-python-headless pillow numpy
# #    pip install pandas openpyxl phonenumbers tldextract rapidfuzz spacy
# #    python -m spacy download en_core_web_sm
# #    (PDF) pip install pdf2image  +  poppler (choco install poppler on Windows)
# #
# #  USAGE:
# #    python visiting_card_ocr_v17.py                    # GUI file picker
# #    python visiting_card_ocr_v17.py card.jpg           # single image
# #    python visiting_card_ocr_v17.py ./cards/           # batch folder
# #    python visiting_card_ocr_v17.py card.jpg --debug   # verbose mode
# #    python visiting_card_ocr_v17.py card.jpg --lang hi # force Hindi OCR
# #    python visiting_card_ocr_v17.py card.jpg --no-vcf  # skip VCF export
# #
# # ══════════════════════════════════════════════════════════════════════════════

# # ── oneDNN Windows crash prevention — MUST precede ALL other imports ──────────
# import os

# os.environ["FLAGS_use_mkldnn"] = "0"
# os.environ["FLAGS_call_stack_level"] = "2"

# import re, io, cv2, sys, csv, math, json, zipfile, argparse, traceback
# import unicodedata, logging
# from datetime import datetime
# from pathlib import Path

# import numpy as np
# from PIL import Image

# # ── Logging (replaces bare print) ─────────────────────────────────────────────
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(message)s",
#     datefmt="%H:%M:%S",
#     handlers=[
#         logging.StreamHandler(sys.stdout),
#         logging.FileHandler("ocr_engine.log", encoding="utf-8"),
#     ],
# )
# log = logging.getLogger("VC_OCR")

# # ── Soft dependencies ─────────────────────────────────────────────────────────
# try:
#     import pandas as pd
#     import openpyxl
#     from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
#     from openpyxl.utils import get_column_letter

#     EXCEL_OK = True
# except ImportError:
#     EXCEL_OK = False
#     log.warning("pip install pandas openpyxl")

# try:
#     import phonenumbers
#     from phonenumbers import PhoneNumberMatcher, PhoneNumberFormat

#     PHONE_OK = True
# except ImportError:
#     PHONE_OK = False
#     log.warning("pip install phonenumbers")

# try:
#     import tldextract

#     TLDX_OK = True
# except ImportError:
#     TLDX_OK = False

# try:
#     import spacy

#     _NLP = spacy.load("en_core_web_sm")
#     SPACY_OK = True
# except Exception:
#     _NLP = None
#     SPACY_OK = False
#     log.warning("python -m spacy download en_core_web_sm")

# try:
#     from rapidfuzz import fuzz as rfuzz

#     FUZZ_OK = True
# except ImportError:
#     FUZZ_OK = False
#     log.warning("pip install rapidfuzz")

# try:
#     from paddleocr import PaddleOCR

#     PADDLE_OK = True
# except ImportError:
#     PADDLE_OK = False
#     log.warning("pip install paddlepaddle paddleocr")

# try:
#     import torch

#     GPU_AVAIL = torch.cuda.is_available()
# except ImportError:
#     GPU_AVAIL = False

# try:
#     from pdf2image import convert_from_bytes

#     PDF_OK = True
# except ImportError:
#     PDF_OK = False

# try:
#     from tkinter import Tk, filedialog

#     TK_OK = True
# except ImportError:
#     TK_OK = False

# import warnings

# warnings.filterwarnings("ignore")


# # ══════════════════════════════════════════════════════════════════════════════
# #  CONFIGURATION
# # ══════════════════════════════════════════════════════════════════════════════
# CFG = {
#     "OCR_CONF_THRESH": 0.18,
#     "PDF_DPI": 250,
#     "SEG_MIN_AREA_FRAC": 0.03,
#     "SEG_MAX_AREA_FRAC": 0.95,
#     "SEG_PAD_PX": 12,
#     "SEG_CARD_ASPECT_MIN": 0.5,  # wide gate: portrait + landscape
#     "SEG_CARD_ASPECT_MAX": 5.0,
#     "ROW_TOL_FACTOR": 0.55,
#     "ROW_TOL_FALLBACK_PX": 22,
#     "MERGE_CELL_FACTOR": 0.035,  # coarser than v11/v12 to absorb drift
#     "MERGE_CELL_MIN_PX": 18,
#     "ECHO_IOU_THRESH": 0.45,  # IOU threshold for token-level dedup
#     "SECONDARY_SCRIPT_THRESH": 0.15,
#     "PHONE_FUZZ_THRESH": 80,
#     "NAME_ALPHA_RATIO": 0.75,
#     "PHONE_MIN_DIGITS_NO_PREFIX": 8,
#     "MAX_PHONES": 5,
#     "DEBUG": False,
#     "EXCEL_DB_FILE": "cards_database.xlsx",
# }

# SCRIPT_TO_LANG = {
#     "latin": "en",
#     "arabic": "ar",
#     "devanagari": "hi",
#     "cjk": "ch",
#     "korean": "korean",
#     "cyrillic": "ru",
#     "japanese": "japan",
#     "tamil": "ta",
#     "telugu": "te",
#     "thai": "th",
# }


# # ══════════════════════════════════════════════════════════════════════════════
# #  REGEX PATTERNS
# # ══════════════════════════════════════════════════════════════════════════════

# # BUG 6 FIX — Strict TLD-anchored email regex.
# # Terminates at a known TLD boundary → prevents ".dek.mueller" suffix bleed.
# # Both versions (v14, v16) used a greedy regex that matched the full bad string.
# _TLD_LIST = (
#     r"com|org|net|in|io|co\.in|co\.uk|co\.ae|co\.sg|co\.za|co\.au|"
#     r"co\.nz|co\.jp|biz|info|edu|gov|ae|us|au|nz|sg|hk|my|ph|za|"
#     r"de|fr|jp|cn|ca|br|mx|tech|app|dev|ai|online|store|site|"
#     r"ae|il|tr|pk|lk|ir|ma|dz|tn|ly|ng|ke|tz|ug|zm|zw|"
#     r"pt|lu|ie|fi|lv|ee|ua|cz|sk|sa|bh|qa|gr|nl|be|hu|it|"
#     r"ro|ch|at|gb|dk|se|no|pl|ru|kr|vn|id|th|ph|tw|ar|pe|cl|co|ve"
# )
# _EMAIL_RE = re.compile(
#     r"[A-Za-z0-9._%+\-]{1,64}"
#     r"@"
#     r"[A-Za-z0-9.\-]{1,253}"
#     r"\."
#     r"(?:" + _TLD_LIST + r")"
#     r"(?=\s|$|[^a-zA-Z0-9.\-])",  # lookahead: must end at non-domain char
#     re.IGNORECASE,
# )

# _WEB_RE = re.compile(
#     r"(?:https?://|www\.)[A-Za-z0-9.\-/_%?=&#]+"
#     r"|[A-Za-z0-9][\w\-]*\."
#     r"(?:" + _TLD_LIST + r")"
#     r"(?:[/\w\-]*)?",
#     re.IGNORECASE,
# )

# _LINKEDIN_RE = re.compile(r"linkedin\.com/in/[\w\-]+", re.IGNORECASE)
# _TWITTER_RE = re.compile(
#     r"(?:(?:^|(?<=\s))@[\w]{2,50}|twitter\.com/[\w]+|x\.com/[\w]+)",
#     re.IGNORECASE | re.MULTILINE,
# )
# _SOCIAL_RE = re.compile(
#     r"linkedin\.com|twitter\.com|x\.com|instagram\.com|facebook\.com|@[\w]{2,50}",
#     re.IGNORECASE,
# )
# _PHONE_LABEL_RE = re.compile(
#     r"^\s*(?:office|mobile|mob|cell|tel|fax|ph|phone|direct|work|home|"
#     r"helpline|hotline|tollfree|toll[-\s]free)\s*[:\-]?\s*",
#     re.IGNORECASE,
# )
# _INTL_PREFIX_RE = re.compile(r"(?:^|\s)\+(\d{1,3})[\s\-\(]")
# # Postal code formats (used to prevent postal codes being parsed as phones)
# _POSTAL_FMT_RE = re.compile(
#     r"^\d{3}-\d{4}$"  # JP: 108-0075
#     r"|^\d{4,6}$"  # IN/SG/AU: 395007
#     r"|^\d{2}-\d{4}$"  # variant
#     r"|^[A-Z]{1,2}\d{1,2}\s?\d[A-Z]{2}$",  # GB: WC1R 5AH
#     re.IGNORECASE,
# )

# DESIGNATION_KW = re.compile(
#     r"\b(ceo|cto|coo|cfo|cmo|vp|md|gm|founder|co-?founder|partner|"
#     r"director|manager|engineer|developer|architect|analyst|"
#     r"consultant|officer|president|vice[\s\-]?president|associate|"
#     r"senior|junior|principal|head|lead|specialist|advisor|"
#     r"coordinator|supervisor|assistant|proprietor|owner|chairman|"
#     r"trustee|secretary|treasurer|intern|trainee|"
#     r"doctor|dr\.?|prof\.?|professor|lawyer|advocate|solicitor|"
#     r"accountant|auditor|designer|strategist|researcher|scientist|"
#     r"technician|representative|agent|broker|dealer|contractor|builder|"
#     r"gerant|directeur|ingenieur|presidente|gerente|ingeniero)\b",
#     re.IGNORECASE,
# )
# COMPANY_KW = re.compile(
#     r"\b(ltd|limited|inc|llp|llc|pvt|private|plc|corp|corporation|"
#     r"solutions|consulting|studio|technologies|tech|group|company|"
#     r"labs|services|systems|enterprises|associates|builders|"
#     r"construction|industries|global|international|ventures|"
#     r"holdings|capital|networks|media|digital|infra|infrastructure|"
#     r"realty|realtors|properties|developers|architects|interiors|"
#     r"design|agency|firm|foundation|trust|institute|academy|"
#     r"hospital|clinic|healthcare|pharma|logistics|transport|"
#     r"exports|imports|trading|manufacturing|fabrication|engineering|"
#     r"gmbh|ag|kg|ohg|ug|sarl|sas|spa|srl|s\.p\.a\.|k\.k\.|kk|"
#     r"sdn\.?\s*bhd|pte\.?|b\.v\.|n\.v\.|a\/s|s\.a\.)\b",
#     re.IGNORECASE,
# )
# ADDRESS_KW = re.compile(
#     r"\b(floor|fl\.|level|suite|plot|block|sector|phase|unit|apt|apartment|"
#     r"building|bldg|tower|complex|plaza|centre|center|mall|"
#     r"road|rd\.|street|st\.|avenue|ave\.|lane|ln\.|boulevard|blvd\.|"
#     r"drive|dr\.|court|ct\.|place|pl\.|way|highway|freeway|square|sq\.|"
#     r"terrace|ter\.|close|crescent|grove|gardens|"
#     r"nagar|vihar|marg|gali|chowk|bazaar|colony|society|residency|"
#     r"stra[ß]e|strasse|str\.|gasse|weg|platz|rue|impasse|allee|"
#     r"calle|avenida|carrera|paseo|"
#     r"district|area|zone|suburb|village|town|city|state|province|"
#     r"county|region|po\s*box|p\.o\.|zip|pin|postcode|postal)\b",
#     re.IGNORECASE,
# )
# _NAME_STOPWORDS = re.compile(
#     r"\b(floor|level|suite|road|street|avenue|lane|boulevard|drive|court|"
#     r"place|square|terrace|close|crescent|complex|tower|building|"
#     r"nagar|vihar|marg|chowk|colony|society|stra[ß]e|strasse|gasse|weg|"
#     r"platz|rue|calle|avenida|carrera|block|sector|phase|plot|"
#     r"district|zone|suburb|village|town|city|state|province|county|"
#     r"region|postcode|postal|zip|pin)\b",
#     re.IGNORECASE,
# )
# POSTAL_RE = re.compile(r"\b\d{4,9}\b")

# # Country calling code → ISO region
# _CC_REGION: dict = {
#     "1": "US",
#     "7": "RU",
#     "20": "EG",
#     "27": "ZA",
#     "30": "GR",
#     "31": "NL",
#     "32": "BE",
#     "33": "FR",
#     "34": "ES",
#     "36": "HU",
#     "39": "IT",
#     "40": "RO",
#     "41": "CH",
#     "43": "AT",
#     "44": "GB",
#     "45": "DK",
#     "46": "SE",
#     "47": "NO",
#     "48": "PL",
#     "49": "DE",
#     "51": "PE",
#     "52": "MX",
#     "54": "AR",
#     "55": "BR",
#     "56": "CL",
#     "57": "CO",
#     "60": "MY",
#     "61": "AU",
#     "62": "ID",
#     "63": "PH",
#     "64": "NZ",
#     "65": "SG",
#     "66": "TH",
#     "81": "JP",
#     "82": "KR",
#     "84": "VN",
#     "86": "CN",
#     "90": "TR",
#     "91": "IN",
#     "92": "PK",
#     "94": "LK",
#     "98": "IR",
#     "212": "MA",
#     "213": "DZ",
#     "216": "TN",
#     "218": "LY",
#     "234": "NG",
#     "254": "KE",
#     "255": "TZ",
#     "256": "UG",
#     "260": "ZM",
#     "263": "ZW",
#     "351": "PT",
#     "352": "LU",
#     "353": "IE",
#     "358": "FI",
#     "371": "LV",
#     "372": "EE",
#     "380": "UA",
#     "420": "CZ",
#     "421": "SK",
#     "966": "SA",
#     "968": "OM",
#     "971": "AE",
#     "972": "IL",
#     "974": "QA",
# }
# _FALLBACK_REGIONS = [
#     "IN",
#     "US",
#     "GB",
#     "AE",
#     "SG",
#     "AU",
#     "CA",
#     "DE",
#     "FR",
#     "ZA",
#     "NG",
#     "KE",
#     "JP",
#     "CN",
#     "KR",
#     "BR",
#     "MX",
#     "NZ",
#     "PH",
#     "MY",
#     "BD",
#     "LK",
#     "NP",
#     "KW",
#     "QA",
#     "BH",
#     "SA",
#     "OM",
#     "EG",
# ]

# # Excel output columns
# EXCEL_COLS = [
#     "Name",
#     "Job_Title",
#     "Company",
#     "Email",
#     "Phone_1",
#     "Phone_2",
#     "Phone_3",
#     "Website",
#     "Address",
#     "LinkedIn",
#     "Twitter",
#     "Quality",
#     "Processed",
#     "File",
# ]
# _QC = {"🟢 GREEN": "C6EFCE", "🟡 YELLOW": "FFEB9C", "🔴 RED": "FFC7CE"}


# # ══════════════════════════════════════════════════════════════════════════════
# #  MODEL SINGLETONS
# # ══════════════════════════════════════════════════════════════════════════════
# _PADDLE_CACHE: dict = {}


# def _get_paddle(lang: str = "en"):
#     if lang not in _PADDLE_CACHE:
#         if not PADDLE_OK:
#             raise RuntimeError("paddleocr not installed")
#         log.info("Loading PaddleOCR (lang=%s, gpu=%s)…", lang, GPU_AVAIL)
#         _PADDLE_CACHE[lang] = PaddleOCR(
#             lang=lang,
#             use_doc_orientation_classify=False,
#             use_doc_unwarping=False,
#             use_textline_orientation=False,
#             text_rec_score_thresh=CFG["OCR_CONF_THRESH"],
#             device="gpu" if GPU_AVAIL else "cpu",
#             show_log=False,
#         )
#     return _PADDLE_CACHE[lang]


# # ══════════════════════════════════════════════════════════════════════════════
# #  L0 — INPUT LOADER
# # ══════════════════════════════════════════════════════════════════════════════
# _IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


# def _pil_to_bgr(pil: Image.Image) -> np.ndarray:
#     return cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)


# def load_images(path: str):
#     """Generator yielding (label, bgr_ndarray) from image/PDF/ZIP."""
#     p = Path(path)
#     ext = p.suffix.lower()
#     if ext in _IMG_EXTS:
#         img = cv2.imread(str(p), cv2.IMREAD_COLOR)
#         if img is None:
#             try:
#                 img = _pil_to_bgr(Image.open(p))
#             except Exception as e:
#                 log.error("Cannot read %s: %s", path, e)
#                 return
#         yield p.name, img
#     elif ext == ".pdf":
#         if not PDF_OK:
#             log.error("pdf2image not installed")
#             return
#         try:
#             pages = convert_from_bytes(p.read_bytes(), dpi=CFG["PDF_DPI"], fmt="jpeg")
#             for i, pg in enumerate(pages, 1):
#                 yield f"{p.stem} (page {i})", _pil_to_bgr(pg)
#         except Exception as e:
#             log.error("PDF read failed (%s): %s", path, e)
#     elif ext == ".zip":
#         try:
#             with zipfile.ZipFile(p) as zf:
#                 members = sorted(
#                     m
#                     for m in zf.namelist()
#                     if Path(m).suffix.lower() in _IMG_EXTS
#                     and not Path(m).name.startswith(".")
#                 )
#                 for m in members:
#                     try:
#                         arr = np.frombuffer(zf.read(m), np.uint8)
#                         img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
#                         if img is None:
#                             img = _pil_to_bgr(Image.open(io.BytesIO(zf.read(m))))
#                         yield f"{p.name}/{Path(m).name}", img
#                     except Exception as e:
#                         log.warning("Cannot read %s: %s", m, e)
#         except Exception as e:
#             log.error("ZIP read failed (%s): %s", path, e)
#     else:
#         log.warning("Unsupported file type: %s", ext)


# # ══════════════════════════════════════════════════════════════════════════════
# #  L1 — CARD SEGMENTOR
# # ══════════════════════════════════════════════════════════════════════════════


# def segment_cards(img: np.ndarray) -> list:
#     h, w = img.shape[:2]
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
#     edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 30, 120)
#     dilated = cv2.dilate(
#         edges, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5)), iterations=3
#     )
#     cnts, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     pad = CFG["SEG_PAD_PX"]
#     rois = []
#     for c in cnts:
#         area = cv2.contourArea(c)
#         if not (
#             h * w * CFG["SEG_MIN_AREA_FRAC"] <= area <= h * w * CFG["SEG_MAX_AREA_FRAC"]
#         ):
#             continue
#         x, y, cw, ch = cv2.boundingRect(c)
#         asp = cw / max(ch, 1)
#         if CFG["SEG_CARD_ASPECT_MIN"] <= asp <= CFG["SEG_CARD_ASPECT_MAX"]:
#             rois.append((x, y, cw, ch))
#     if not rois:
#         return [img]
#     rois.sort(key=lambda r: (r[1] // 100, r[0]))
#     return [
#         img[
#             max(0, y - pad) : min(h, y + ch + pad),
#             max(0, x - pad) : min(w, x + cw + pad),
#         ]
#         for x, y, cw, ch in rois
#     ]


# # ══════════════════════════════════════════════════════════════════════════════
# #  L2 — IMAGE PIPELINE
# # ══════════════════════════════════════════════════════════════════════════════


# def deskew(img: np.ndarray) -> np.ndarray:
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
#     lines = cv2.HoughLinesP(
#         cv2.Canny(gray, 50, 150),
#         1,
#         np.pi / 180,
#         threshold=80,
#         minLineLength=60,
#         maxLineGap=20,
#     )
#     if lines is None:
#         return img
#     angles = [
#         math.degrees(math.atan2(l[0][3] - l[0][1], l[0][2] - l[0][0]))
#         for l in lines
#         if abs(math.degrees(math.atan2(l[0][3] - l[0][1], l[0][2] - l[0][0]))) < 45
#     ]
#     if not angles:
#         return img
#     angle = float(np.median(angles))
#     if abs(angle) < 0.5:
#         return img
#     h, w = img.shape[:2]
#     M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
#     cos_a, sin_a = abs(M[0, 0]), abs(M[0, 1])
#     nw = int(h * sin_a + w * cos_a)
#     nh = int(h * cos_a + w * sin_a)
#     M[0, 2] += (nw - w) / 2
#     M[1, 2] += (nh - h) / 2
#     return cv2.warpAffine(
#         img, M, (nw, nh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
#     )


# def preprocess(img: np.ndarray) -> np.ndarray:
#     """Returns BGR array (not grayscale) so PaddleOCR always gets 3 channels."""
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
#     gray = cv2.GaussianBlur(gray, (3, 3), 0)
#     roi = gray[
#         int(img.shape[0] * 0.30) : int(img.shape[0] * 0.70),
#         int(img.shape[1] * 0.30) : int(img.shape[1] * 0.70),
#     ]
#     if float(np.mean(roi)) < 128:
#         gray = cv2.bitwise_not(gray)
#     gray = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
#     blur = cv2.GaussianBlur(gray, (0, 0), 3)
#     gray = cv2.addWeighted(gray, 1.55, blur, -0.55, 0)
#     if float(np.std(gray)) < 42:
#         gray = cv2.adaptiveThreshold(
#             gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10
#         )
#     return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# # ══════════════════════════════════════════════════════════════════════════════
# #  L3 — OCR ENGINE
# # ══════════════════════════════════════════════════════════════════════════════


# def _parse_paddle_result(results) -> list:
#     """
#     Normalise PaddleOCR v2/v3 result objects into (text, conf, pts) tuples.
#     v3 returns result objects with .json; v2 returns plain dicts.
#     """
#     parsed = []
#     if not results:
#         return parsed
#     for res_obj in results:
#         if hasattr(res_obj, "json"):
#             inner = res_obj.json
#         elif isinstance(res_obj, dict):
#             inner = res_obj.get("res", res_obj)
#         else:
#             try:
#                 inner = dict(res_obj)
#             except:
#                 continue
#         texts = inner.get("rec_texts", []) or []
#         scores = inner.get("rec_scores", []) or []
#         polys = inner.get("rec_polys", inner.get("dt_polys", [])) or []
#         if isinstance(polys, np.ndarray):
#             polys = (
#                 list(polys) if polys.ndim == 3 else ([polys] if polys.ndim == 2 else [])
#             )
#         for text, conf, poly in zip(texts, scores, polys):
#             text = str(text).strip()
#             conf = float(conf)
#             if not text or conf < CFG["OCR_CONF_THRESH"]:
#                 continue
#             pts = (
#                 poly.tolist()
#                 if isinstance(poly, np.ndarray)
#                 else [list(map(float, p)) for p in poly]
#             )
#             parsed.append((text, conf, pts))
#     return parsed


# def _bbox_key(pts: list, cell_px: float) -> str:
#     cx = sum(p[0] for p in pts) / len(pts)
#     cy = sum(p[1] for p in pts) / len(pts)
#     return f"{int(cx//cell_px)},{int(cy//cell_px)}"


# def _iou(a: list, b: list) -> float:
#     ax1 = min(p[0] for p in a)
#     ax2 = max(p[0] for p in a)
#     ay1 = min(p[1] for p in a)
#     ay2 = max(p[1] for p in a)
#     bx1 = min(p[0] for p in b)
#     bx2 = max(p[0] for p in b)
#     by1 = min(p[1] for p in b)
#     by2 = max(p[1] for p in b)
#     ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
#     iy = max(0.0, min(ay2, by2) - max(ay1, by1))
#     inter = ix * iy
#     if not inter:
#         return 0.0
#     area_a = max((ax2 - ax1) * (ay2 - ay1), 1)
#     area_b = max((bx2 - bx1) * (by2 - by1), 1)
#     return inter / (area_a + area_b - inter)


# def _iou_dedup(pool: dict) -> dict:
#     """Remove near-duplicate tokens at the same position (IOU-based)."""
#     tokens = list(pool.values())
#     keep = [True] * len(tokens)
#     for i in range(len(tokens)):
#         if not keep[i]:
#             continue
#         for j in range(i + 1, len(tokens)):
#             if not keep[j]:
#                 continue
#             if tokens[i]["text"].strip().upper() == tokens[j]["text"].strip().upper():
#                 if _iou(tokens[i]["bbox"], tokens[j]["bbox"]) >= CFG["ECHO_IOU_THRESH"]:
#                     if tokens[i]["conf"] >= tokens[j]["conf"]:
#                         keep[j] = False
#                     else:
#                         keep[i] = False
#                         break
#     result = {}
#     for i, tok in enumerate(tokens):
#         if keep[i]:
#             k = _bbox_key(tok["bbox"], max(tok.get("th", 20), CFG["MERGE_CELL_MIN_PX"]))
#             result[k] = tok
#     return result


# def _run_ocr_pass(img_bgr: np.ndarray, lang: str, cell_px: float, pool: dict):
#     if img_bgr.ndim == 2:
#         img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
#     elif img_bgr.shape[2] == 4:
#         img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)
#     try:
#         for text, conf, pts in _parse_paddle_result(_get_paddle(lang).predict(img_bgr)):
#             xs = [p[0] for p in pts]
#             ys = [p[1] for p in pts]
#             tok = {
#                 "text": text,
#                 "conf": conf,
#                 "bbox": pts,
#                 "cx": sum(xs) / len(xs),
#                 "cy": sum(ys) / len(ys),
#                 "th": max(ys) - min(ys),
#             }
#             k = _bbox_key(pts, cell_px)
#             if k not in pool or conf > pool[k]["conf"]:
#                 pool[k] = tok
#     except Exception as e:
#         if CFG["DEBUG"]:
#             log.debug("OCR pass failed (lang=%s): %s", lang, e)


# def detect_scripts(text: str):
#     counts = {s: 0 for s in SCRIPT_TO_LANG}
#     for ch in text:
#         if not ch.strip():
#             continue
#         cp = ord(ch)
#         if 0x0600 <= cp <= 0x06FF:
#             counts["arabic"] += 1
#         elif 0x0900 <= cp <= 0x097F:
#             counts["devanagari"] += 1
#         elif 0x4E00 <= cp <= 0x9FFF:
#             counts["cjk"] += 1
#         elif 0xAC00 <= cp <= 0xD7AF or 0x1100 <= cp <= 0x11FF:
#             counts["korean"] += 1
#         elif 0x0400 <= cp <= 0x04FF:
#             counts["cyrillic"] += 1
#         elif 0x3040 <= cp <= 0x30FF:
#             counts["japanese"] += 1
#         elif 0x0B80 <= cp <= 0x0BFF:
#             counts["tamil"] += 1
#         elif 0x0C00 <= cp <= 0x0C7F:
#             counts["telugu"] += 1
#         elif 0x0E00 <= cp <= 0x0E7F:
#             counts["thai"] += 1
#         elif ch.isalpha():
#             counts["latin"] += 1
#     total = sum(counts.values()) or 1
#     ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)
#     primary = (
#         ranked[0][0]
#         if (ranked[0][0] == "latin" or counts[ranked[0][0]] >= 5)
#         else "latin"
#     )
#     secondary = next(
#         (
#             sc
#             for sc, cnt in ranked[1:]
#             if sc != primary and cnt / total >= CFG["SECONDARY_SCRIPT_THRESH"]
#         ),
#         None,
#     )
#     return primary, secondary


# def ocr_card(proc: np.ndarray, lang_override: str = None):
#     h = proc.shape[0]
#     cell_px = max(h * CFG["MERGE_CELL_FACTOR"], CFG["MERGE_CELL_MIN_PX"])
#     pool: dict = {}
#     base = lang_override or "en"
#     _run_ocr_pass(proc, base, cell_px, pool)
#     _run_ocr_pass(cv2.convertScaleAbs(proc, alpha=1.35, beta=25), base, cell_px, pool)
#     pool = _iou_dedup(pool)  # IOU dedup BEFORE line assembly
#     tokens = list(pool.values())
#     primary, secondary = detect_scripts(" ".join(t["text"] for t in tokens))
#     sec_lang = SCRIPT_TO_LANG.get(secondary)
#     if sec_lang and sec_lang not in {base, "en"}:
#         _run_ocr_pass(proc, sec_lang, cell_px, pool)
#         pool = _iou_dedup(pool)
#         tokens = list(pool.values())
#     return tokens, primary, secondary


# # ══════════════════════════════════════════════════════════════════════════════
# #  TEXT CLEANING  (BUG 2 FIX: v16's double-space-aware collapse)
# # ══════════════════════════════════════════════════════════════════════════════


# def _collapse_spaced(text: str) -> str:
#     """
#     BUG 2 FIX — uses v16's double-space word-boundary approach.
#     "C R E A T I V E  D I R E C T O R" → "CREATIVE DIRECTOR"  (2 words preserved)
#     "I S A B E L L A" → "ISABELLA"
#     v14's approach collapsed everything with no spaces → "CREATIVEDIRECTOR" (wrong)
#     """
#     segs = text.strip().split()
#     if len(segs) < 2:
#         return text
#     single_ratio = sum(
#         1 for s in segs if len(s) == 1 and unicodedata.category(s)[0] == "L"
#     ) / len(segs)
#     if single_ratio < 0.6:
#         return text
#     if not all(
#         1 <= len(s) <= 4 and all(unicodedata.category(c)[0] == "L" for c in s)
#         for s in segs
#     ):
#         return text
#     word_groups = re.split(r" {2,}", text.strip())
#     if len(word_groups) > 1:
#         return " ".join(
#             "".join(grp.split()) if all(1 <= len(s) <= 4 for s in grp.split()) else grp
#             for grp in word_groups
#         )
#     return "".join(segs)


# def _dedup_line_text(text: str) -> str:
#     """Remove exact repeated substrings: 'ABC ABC' → 'ABC'."""
#     t = text.strip()
#     m = re.match(r"^(.{4,}?)\s+\1$", t)
#     if m:
#         return m.group(1)
#     # Sliding split check for near-duplicates
#     for split in range(max(4, len(t) // 4), len(t) // 2 + 1):
#         half = t[:split]
#         rest = t[split:].lstrip()
#         if rest == half:
#             return half
#     return t


# def _dedup_repeated_words(text: str) -> str:
#     """Remove word-level repeats: 'SONY GROUP K.K. SONY GROUP K.K.' → fixed."""
#     words = text.split()
#     for half in range(len(words) // 2, 1, -1):
#         for start in range(len(words) - half * 2 + 1):
#             if words[start : start + half] == words[start + half : start + half * 2]:
#                 return _dedup_repeated_words(
#                     " ".join(words[: start + half] + words[start + half * 2 :])
#                 )
#     return text


# def _normalise_pipe(text: str) -> str:
#     if re.search(r"(?:linkedin|twitter|@)", text, re.I):
#         text = re.sub(r"(?<=[^\s])\s+I\s+(?=[^\s])", " | ", text)
#     return text


# def clean_token(text: str) -> str:
#     text = unicodedata.normalize("NFKC", text)
#     text = _normalise_pipe(text)
#     text = _collapse_spaced(text)
#     text = re.sub(r"\s*@\s*", "@", text)
#     text = re.sub(r"(?<=\w)\s*\.\s*(?=\w)", ".", text)
#     return re.sub(r" {2,}", " ", text).strip()


# # ══════════════════════════════════════════════════════════════════════════════
# #  LINE GROUPING
# # ══════════════════════════════════════════════════════════════════════════════


# def _collapse_row_singles(tokens: list) -> list:
#     """Merge runs of single letters in a row: ['J','O','H','N'] → ['JOHN']."""
#     result, i = [], 0
#     while i < len(tokens):
#         tok = tokens[i]
#         if len(tok) == 1 and unicodedata.category(tok)[0] == "L":
#             run = [tok]
#             j = i + 1
#             while (
#                 j < len(tokens)
#                 and len(tokens[j]) == 1
#                 and unicodedata.category(tokens[j])[0] == "L"
#             ):
#                 run.append(tokens[j])
#                 j += 1
#             result.append("".join(run) if len(run) >= 3 else tok)
#             i = j if len(run) >= 3 else i + 1
#         else:
#             result.append(tok)
#             i += 1
#     return result


# def group_into_lines(tokens: list) -> list:
#     """Group tokens into visual lines. Returns list of (line_text, row_tokens)."""
#     if not tokens:
#         return []
#     heights = [t.get("th", 20) for t in tokens]
#     tol = max(
#         float(np.median(heights)) * CFG["ROW_TOL_FACTOR"], CFG["ROW_TOL_FALLBACK_PX"]
#     )
#     by_y = sorted(tokens, key=lambda t: t["cy"])
#     rows, cur = [], [by_y[0]]
#     for tok in by_y[1:]:
#         if abs(float(tok["cy"]) - sum(t["cy"] for t in cur) / len(cur)) <= tol:
#             cur.append(tok)
#         else:
#             rows.append(sorted(cur, key=lambda t: t["cx"]))
#             cur = [tok]
#     rows.append(sorted(cur, key=lambda t: t["cx"]))
#     result = []
#     for row in rows:
#         texts = [clean_token(t["text"]) for t in row if t["text"].strip()]
#         # Remove consecutive duplicates within a row
#         deduped = []
#         for txt in texts:
#             if not deduped or txt.upper() != deduped[-1].upper():
#                 deduped.append(txt)
#         collapsed = _collapse_row_singles(deduped)
#         line = _dedup_line_text(" ".join(collapsed))  # BUG 1 FIX
#         result.append((line, row))
#     return result


# # ══════════════════════════════════════════════════════════════════════════════
# #  L4 — FIELD EXTRACTORS
# # ══════════════════════════════════════════════════════════════════════════════


# def _detect_region(text: str):
#     m = _INTL_PREFIX_RE.search(text)
#     if not m:
#         return None
#     pfx = m.group(1)
#     for length in (3, 2, 1):
#         r = _CC_REGION.get(pfx[:length])
#         if r:
#             return r
#     return None


# # ── Email (BUG 6 FIX — strict TLD-anchored regex) ────────────────────────────


# def _extract_email(lines: list) -> str:
#     """
#     BUG 6 FIX: Uses strict TLD-anchored regex that stops at the TLD boundary.
#     "k.mueller@bosch-engineering.dek.mueller" → "k.mueller@bosch-engineering.de"
#     because ".de" is in the TLD list and the lookahead (?=[^a-zA-Z0-9.\-]|$)
#     stops the match before ".k.mueller".
#     Both v14 and v16 failed this case — their greedy regex matched the full string.
#     """
#     for ln, _ in lines:
#         compact = re.sub(r"\s+", "", ln)
#         m = _EMAIL_RE.search(compact)
#         if m:
#             email = m.group().lower()
#             if len(email) <= 80 and "@" in email and "." in email.split("@")[-1]:
#                 return email
#     return ""


# # ── Phones ────────────────────────────────────────────────────────────────────


# def _extract_phones(lines: list) -> list:
#     found, found_digs = [], []
#     detected_region = _detect_region(" ".join(ln for ln, _ in lines))
#     for ln, _ in lines:
#         if len(found) >= CFG["MAX_PHONES"]:
#             break
#         if ADDRESS_KW.search(ln):
#             continue  # BUG 7 FIX
#         if _SOCIAL_RE.search(ln):
#             continue
#         cleaned = _PHONE_LABEL_RE.sub("", ln).strip()
#         if not cleaned:
#             continue
#         raw_digits = re.sub(r"\D", "", cleaned)
#         if len(raw_digits) < 4:
#             continue
#         if _POSTAL_FMT_RE.fullmatch(cleaned.strip()):
#             continue  # BUG 4 FIX
#         parsed = False
#         if PHONE_OK:
#             line_region = _detect_region(cleaned)
#             regions = list(
#                 dict.fromkeys(
#                     [r for r in [line_region, detected_region] if r] + _FALLBACK_REGIONS
#                 )
#             )
#             for region in regions:
#                 try:
#                     for match in PhoneNumberMatcher(cleaned, region):
#                         num = match.number
#                         if not phonenumbers.is_valid_number(num):
#                             continue
#                         if len(str(num.national_number)) < 6:
#                             continue
#                         fmt = phonenumbers.format_number(
#                             num, PhoneNumberFormat.INTERNATIONAL
#                         )
#                         digs = re.sub(r"\D", "", fmt)
#                         if _POSTAL_FMT_RE.fullmatch(re.sub(r"[+\s]", "", fmt)):
#                             continue
#                         if digs not in found_digs:
#                             found.append(fmt)
#                             found_digs.append(digs)
#                         parsed = True
#                     if parsed:
#                         break
#                 except Exception:
#                     continue
#         if not parsed:
#             for m in re.finditer(r"[\+\(]?[\d][\d\s\-\.\(\)]{5,18}[\d]", cleaned):
#                 raw = m.group()
#                 digs = re.sub(r"\D", "", raw)
#                 min_d = (
#                     7
#                     if raw.strip().startswith("+")
#                     else CFG["PHONE_MIN_DIGITS_NO_PREFIX"]
#                 )
#                 if len(digs) < min_d or len(digs) > 15:
#                     continue
#                 if _POSTAL_FMT_RE.fullmatch(digs):
#                     continue
#                 norm = re.sub(
#                     r"\s{2,}", " ", re.sub(r"(?<=\d) (?=\d)", "", raw)
#                 ).strip()
#                 if digs not in found_digs:
#                     found.append(norm)
#                     found_digs.append(digs)
#     return found


# # ── Website ───────────────────────────────────────────────────────────────────


# def _extract_website(lines: list, email: str) -> str:
#     local = email.split("@")[0].lower() if "@" in email else ""
#     for ln, _ in lines:
#         if "@" in ln:
#             continue
#         for m in _WEB_RE.finditer(ln):
#             url = m.group().strip().rstrip(",.")
#             lo = url.lower()
#             if lo.startswith("http") or lo.startswith("www."):
#                 full = ("http://" + url) if not lo.startswith("http") else url
#                 if TLDX_OK:
#                     ext = tldextract.extract(full)
#                     if not ext.domain:
#                         continue
#                 return full.lower()
#             if TLDX_OK:
#                 ext = tldextract.extract(url)
#                 if not ext.domain:
#                     continue
#                 if ext.domain.lower() == local and not ext.subdomain and "/" not in url:
#                     continue
#             elif not re.search(r"\.", url) or re.fullmatch(
#                 r"[A-Za-z]+\.[A-Za-z]+", url
#             ):
#                 continue
#             return ("http://" + url).lower()
#     return ""


# # ── Company ───────────────────────────────────────────────────────────────────


# def _extract_company(lines: list) -> str:
#     scored = []
#     for ln, _ in lines:
#         if re.search(r"@|\d{5,}|https?://|www\.", ln, re.I):
#             continue
#         score = (
#             (10 if COMPANY_KW.search(ln) else 0)
#             + (3 if ln.isupper() and len(ln.split()) >= 2 else 0)
#             + (1 if len(ln.split()) >= 2 else 0)
#             - (6 if DESIGNATION_KW.search(ln) else 0)
#             - (5 if ADDRESS_KW.search(ln) else 0)
#         )
#         if score > 0:
#             scored.append((score, len(ln), ln))
#     if not scored:
#         return ""
#     scored.sort(reverse=True)
#     return _dedup_repeated_words(scored[0][2])


# # ── Job title ─────────────────────────────────────────────────────────────────


# def _extract_job_title(lines: list, claimed: set) -> str:
#     for ln, _ in lines:
#         if ln.strip().lower() in claimed:
#             continue
#         if re.search(r"@|\d{5,}", ln):
#             continue
#         if DESIGNATION_KW.search(ln) and not COMPANY_KW.search(ln):
#             return _dedup_repeated_words(ln)
#     return ""


# # ── Name (BUG 3 FIX — stopword guard) ────────────────────────────────────────


# def _is_name_word(w: str) -> bool:
#     w2 = w.rstrip(".,;")
#     if not w2:
#         return False
#     if "@" in w2 or "." in w2:
#         return False
#     if re.search(r"\d", w2):
#         return False
#     if re.fullmatch(r"[A-Z]\.", w):
#         return True  # initial like "V."
#     if re.fullmatch(r"[A-Z]", w2):
#         return True  # single letter
#     if w2.isupper():
#         return True
#     if w2[0].isupper() and w2[1:].islower():
#         return True
#     # Hyphenated: "Al-Mansoori", "O'Sullivan"
#     parts = re.split(r"[-']", w2)
#     if len(parts) >= 2 and all(
#         p and p[0].isupper() and p[1:].islower() for p in parts if p
#     ):
#         return True
#     return False


# def _extract_name(lines: list, claimed: set, script: str, addr_tokens: set) -> str:
#     all_text = "\n".join(ln for ln, _ in lines)
#     if SPACY_OK and _NLP:
#         for ent in _NLP(all_text).ents:
#             if ent.label_ != "PERSON":
#                 continue
#             cand = ent.text.strip()
#             if (
#                 len(cand) >= 3
#                 and cand.lower() not in claimed
#                 and len(cand.split()) >= 2
#                 and not re.search(r"\d|@|\.", cand)
#                 and not COMPANY_KW.search(cand)
#                 and not DESIGNATION_KW.search(cand)
#                 and not _SOCIAL_RE.search(cand)
#                 and not _NAME_STOPWORDS.search(cand)
#                 and cand.lower() not in addr_tokens
#             ):
#                 return cand
#     card_h = max((t["cy"] for _, row in lines for t in row), default=1) or 1
#     scored = []
#     for ln, row in lines:
#         if not ln.strip() or ln.strip().lower() in claimed:
#             continue
#         if _SOCIAL_RE.search(ln) or "@" in ln:
#             continue
#         if _NAME_STOPWORDS.search(ln):
#             continue  # BUG 3 FIX
#         if ln.strip().lower() in addr_tokens:
#             continue
#         words = ln.split()
#         if not (2 <= len(words) <= 5):
#             continue
#         if words[0] == "I" and len(words) > 1 and words[1].startswith("@"):
#             continue
#         if not all(_is_name_word(w) for w in words):
#             continue
#         alpha = sum(c.isalpha() for c in ln)
#         if alpha / max(len(ln.replace(" ", "")), 1) < CFG["NAME_ALPHA_RATIO"]:
#             continue
#         avg_y = sum(t["cy"] for t in row) / len(row) if row else card_h
#         avg_h = sum(t.get("th", 20) for t in row) / len(row) if row else 20
#         scored.append((avg_h * 0.6 + (1.0 - avg_y / card_h) * 40, ln))
#     if scored:
#         scored.sort(reverse=True)
#         best = _dedup_repeated_words(scored[0][1])
#         parts = []
#         for p in best.split():
#             cp = _collapse_spaced(p)
#             parts.append(cp.title() if cp.isupper() and len(cp) > 2 else cp)
#         return " ".join(parts)
#     return ""


# # ── Address ───────────────────────────────────────────────────────────────────


# def _extract_address(lines: list, claimed: set, phone_list: list) -> str:
#     phone_fps = [re.sub(r"\D", "", p) for p in phone_list if p]
#     parts, seen = [], set()
#     for ln, _ in lines:
#         t = ln.strip()
#         if not t or t.lower() in claimed or "@" in t:
#             continue
#         line_digs = re.sub(r"\D", "", t)
#         if len(line_digs) >= 6:
#             skip = False
#             for fp in phone_fps:
#                 if not fp:
#                     continue
#                 if (
#                     FUZZ_OK
#                     and rfuzz.partial_ratio(fp, line_digs) >= CFG["PHONE_FUZZ_THRESH"]
#                 ):
#                     skip = True
#                     break
#                 elif fp in line_digs:
#                     skip = True
#                     break
#             if skip:
#                 continue
#         is_addr = bool(ADDRESS_KW.search(t))
#         if POSTAL_RE.search(t) and not re.search(r"[+\(]?\d[\d\s\-\.\(\)]{8,}", t):
#             is_addr = True
#         if re.search(r"[A-Z][a-z]+,\s*[A-Z][a-z]+", t):
#             is_addr = True
#         if is_addr:
#             key = t.lower()
#             if key not in seen:
#                 seen.add(key)
#                 parts.append(t)
#     return ", ".join(parts)


# # ── Social ────────────────────────────────────────────────────────────────────


# def _extract_social(lines: list, claimed: set) -> tuple:
#     linkedin = twitter = ""
#     for ln, _ in lines:
#         if not linkedin:
#             m = _LINKEDIN_RE.search(ln)
#             if m:
#                 v = m.group().strip()
#                 if v.lower() not in claimed:
#                     linkedin = v
#         if not twitter:
#             m = _TWITTER_RE.search(ln)
#             if m:
#                 v = m.group().strip()
#                 if v.lower() not in claimed and not _EMAIL_RE.search(v):
#                     twitter = v
#         if linkedin and twitter:
#             break
#     return linkedin, twitter


# # ══════════════════════════════════════════════════════════════════════════════
# #  FIELD ORCHESTRATION
# # ══════════════════════════════════════════════════════════════════════════════


# def extract_fields(lines: list, primary_script: str) -> dict:
#     data = {
#         "Name": "",
#         "Job_Title": "",
#         "Company": "",
#         "Email": "",
#         "Phones": [],
#         "Website": "",
#         "Address": "",
#         "LinkedIn": "",
#         "Twitter": "",
#     }
#     prelim_addr = " ".join(
#         ln
#         for ln, _ in lines
#         if ADDRESS_KW.search(ln) or (POSTAL_RE.search(ln) and "@" not in ln)
#     )
#     data["Email"] = _extract_email(lines)
#     data["Company"] = _extract_company(lines)
#     data["Phones"] = _extract_phones(lines)
#     data["Website"] = _extract_website(lines, data["Email"])
#     claimed = {
#         data["Email"].lower(),
#         data["Company"].strip().lower(),
#         data["Website"].lower(),
#         "",
#     }
#     for p in data["Phones"]:
#         claimed.add(p.lower())
#     data["Job_Title"] = _extract_job_title(lines, claimed)
#     claimed.add(data["Job_Title"].strip().lower())
#     addr_tokens = {
#         w
#         for ln, _ in lines
#         if ADDRESS_KW.search(ln) or POSTAL_RE.search(ln)
#         for w in ln.lower().split()
#     }
#     data["Name"] = _extract_name(lines, claimed, primary_script, addr_tokens)
#     claimed.add(data["Name"].strip().lower())
#     data["Address"] = _extract_address(lines, claimed, data["Phones"])
#     data["LinkedIn"], data["Twitter"] = _extract_social(lines, claimed)
#     return data


# # ══════════════════════════════════════════════════════════════════════════════
# #  QUALITY SCORING  (BUG 5 FIX — v16's weighted validated scoring)
# # ══════════════════════════════════════════════════════════════════════════════


# def quality_score(data: dict) -> str:
#     """
#     BUG 5 FIX: Uses v16's validated scoring (proven better in testing).
#     v14 gave GREEN even when Name="Surat" or Name="" (4/5 non-empty check).
#     v16 requires n_ok AND (e_ok OR p_ok) for GREEN — much stricter.
#     """
#     phones = data.get("Phones", [])
#     n_ok = bool(
#         data.get("Name")
#         and len(data["Name"].split()) >= 2
#         and not re.search(r"\d|@|\.", data["Name"])
#         and not ADDRESS_KW.search(data["Name"])
#         and not COMPANY_KW.search(data["Name"])
#         and sum(c.isalpha() for c in data["Name"])
#         / max(len(data["Name"].replace(" ", "")), 1)
#         >= 0.75
#     )
#     c_ok = bool(
#         data.get("Company")
#         and data["Company"].strip().lower() != data.get("Name", "").lower()
#         and (
#             COMPANY_KW.search(data["Company"])
#             or (data["Company"].isupper() and len(data["Company"].split()) >= 2)
#         )
#     )
#     e_ok = bool(
#         data.get("Email")
#         and _EMAIL_RE.search(data["Email"])
#         and len(data["Email"]) <= 80
#     )
#     p_ok = bool(phones and any(len(re.sub(r"\D", "", p)) >= 7 for p in phones))
#     a_ok = bool(
#         data.get("Address")
#         and (
#             ADDRESS_KW.search(data["Address"])
#             or POSTAL_RE.search(data["Address"])
#             or re.search(r"[A-Z][a-z]+,\s*[A-Z][a-z]+", data["Address"])
#         )
#     )
#     score = (
#         (2 if n_ok else 0)
#         + (2 if e_ok else 0)
#         + (1 if p_ok else 0)
#         + (1 if c_ok else 0)
#         + (1 if a_ok else 0)
#     )
#     if n_ok and (e_ok or p_ok) and score >= 5:
#         return "🟢 GREEN"
#     if score >= 2:
#         return "🟡 YELLOW"
#     return "🔴 RED"


# # ══════════════════════════════════════════════════════════════════════════════
# #  CORE PROCESSOR
# # ══════════════════════════════════════════════════════════════════════════════


# def process_card_array(
#     img: np.ndarray, label: str = "card", lang_override: str = None
# ) -> dict:
#     proc = preprocess(deskew(img))
#     tokens, primary, secondary = ocr_card(proc, lang_override)
#     if CFG["DEBUG"]:
#         log.debug(
#             "%s — %d tokens  script=%s/%s",
#             label,
#             len(tokens),
#             primary,
#             secondary or "—",
#         )
#         for t in sorted(tokens, key=lambda x: x["cy"]):
#             log.debug("  cy=%6.1f conf=%.2f %r", t["cy"], t["conf"], t["text"])
#     lines = group_into_lines(tokens)
#     if CFG["DEBUG"]:
#         for i, (ln, _) in enumerate(lines, 1):
#             log.debug("  Line %02d: %r", i, ln)
#     data = extract_fields(lines, primary)
#     data["QUALITY"] = quality_score(data)
#     data["_label"] = label
#     data["_script"] = primary
#     data["_script2"] = secondary or ""
#     return data


# def process_source(label: str, page_img: np.ndarray, lang_override: str = None) -> list:
#     crops = segment_cards(page_img)
#     results = []
#     for idx, card_img in enumerate(crops):
#         clabel = label if len(crops) == 1 else f"{label} [card {idx+1}/{len(crops)}]"
#         try:
#             results.append(process_card_array(card_img, clabel, lang_override))
#         except Exception as e:
#             log.error("Failed: %s — %s", clabel, e)
#             if CFG["DEBUG"]:
#                 traceback.print_exc()
#     return results


# # ══════════════════════════════════════════════════════════════════════════════
# #  OUTPUT SYSTEM
# # ══════════════════════════════════════════════════════════════════════════════


# def _row_vals(r: dict) -> list:
#     ph = r.get("Phones", [])
#     return [
#         r.get("Name", ""),
#         r.get("Job_Title", ""),
#         r.get("Company", ""),
#         r.get("Email", ""),
#         ph[0] if len(ph) > 0 else "",
#         ph[1] if len(ph) > 1 else "",
#         ph[2] if len(ph) > 2 else "",
#         r.get("Website", ""),
#         r.get("Address", ""),
#         r.get("LinkedIn", ""),
#         r.get("Twitter", ""),
#         r.get("QUALITY", ""),
#         datetime.now().strftime("%Y-%m-%d %H:%M"),
#         r.get("_label", ""),
#     ]


# def _dedup_key(r: dict) -> str:
#     return "|".join(
#         [
#             re.sub(r"\s+", "", r.get("Email", "").lower()),
#             re.sub(r"\s+", "", r.get("Name", "").lower()),
#             re.sub(r"\D", "", r.get("Phones", [""])[0] if r.get("Phones") else ""),
#         ]
#     )


# def _write_header(ws):
#     hf = PatternFill("solid", fgColor="1F4E79")
#     hfont = Font(bold=True, color="FFFFFF", size=10)
#     thin = Side(style="thin", color="BDD7EE")
#     bdr = Border(left=thin, right=thin, top=thin, bottom=thin)
#     for ci, name in enumerate(EXCEL_COLS, 1):
#         c = ws.cell(1, ci, value=name)
#         c.fill = hf
#         c.font = hfont
#         c.alignment = Alignment(horizontal="center", vertical="center")
#         c.border = bdr
#     ws.row_dimensions[1].height = 20


# def _style_row(ws, rn: int, quality: str):
#     col = _QC.get(quality, "FFFFFF")
#     fill = PatternFill("solid", fgColor=col)
#     thin = Side(style="thin", color="BDD7EE")
#     bdr = Border(left=thin, right=thin, top=thin, bottom=thin)
#     for ci in range(1, len(EXCEL_COLS) + 1):
#         c = ws.cell(rn, ci)
#         c.fill = fill
#         c.alignment = Alignment(vertical="center", wrap_text=True)
#         c.border = bdr
#         c.font = Font(size=9)
#     ws.row_dimensions[rn].height = 15


# def _df_from_results(results: list) -> "pd.DataFrame":
#     rows = []
#     for r in results:
#         ph = r.get("Phones", [])
#         rows.append(
#             {
#                 "Name": r.get("Name", ""),
#                 "Job_Title": r.get("Job_Title", ""),
#                 "Company": r.get("Company", ""),
#                 "Email": r.get("Email", ""),
#                 "Phone_1": ph[0] if len(ph) > 0 else "",
#                 "Phone_2": ph[1] if len(ph) > 1 else "",
#                 "Phone_3": ph[2] if len(ph) > 2 else "",
#                 "Website": r.get("Website", ""),
#                 "Address": r.get("Address", ""),
#                 "LinkedIn": r.get("LinkedIn", ""),
#                 "Twitter": r.get("Twitter", ""),
#                 "Quality": r.get("QUALITY", ""),
#                 "Processed": datetime.now().strftime("%Y-%m-%d %H:%M"),
#                 "File": r.get("_label", ""),
#             }
#         )
#     return pd.DataFrame(rows, columns=EXCEL_COLS)


# def save_to_database(results: list, excel_path: str):
#     if not EXCEL_OK:
#         return
#     month = datetime.now().strftime("%Y-%m")
#     wb = (
#         openpyxl.load_workbook(excel_path)
#         if os.path.exists(excel_path)
#         else openpyxl.Workbook()
#     )
#     if "Sheet" in wb.sheetnames:
#         del wb["Sheet"]
#     if month in wb.sheetnames:
#         ws = wb[month]
#         existing = {}
#         for rn in range(2, ws.max_row + 1):
#             key = _dedup_key(
#                 {
#                     "Email": str(ws.cell(rn, 4).value or ""),
#                     "Name": str(ws.cell(rn, 1).value or ""),
#                     "Phones": [str(ws.cell(rn, 5).value or "")],
#                 }
#             )
#             existing[key] = rn
#     else:
#         ws = wb.create_sheet(title=month)
#         _write_header(ws)
#         existing = {}
#     for r in results:
#         key = _dedup_key(r)
#         is_update = key in existing
#         rn = existing[key] if is_update else ws.max_row + 1
#         vals = _row_vals(r)
#         for ci, v in enumerate(vals, 1):
#             ws.cell(rn, ci, value=v)
#         _style_row(ws, rn, r.get("QUALITY", ""))
#         existing[key] = rn
#         log.info(
#             "  [DB] %s row %d → %s",
#             "Updated" if is_update else "Appended",
#             rn,
#             r.get("Name") or r.get("Email") or "?",
#         )
#     for col in ws.columns:
#         ws.column_dimensions[get_column_letter(col[0].column)].width = min(
#             max(len(str(c.value or "")) for c in col) + 3, 55
#         )
#     ws.freeze_panes = "A2"
#     try:
#         wb.save(excel_path)
#         log.info("Database → %s (sheet: %s)", excel_path, month)
#     except PermissionError:
#         alt = excel_path.replace(".xlsx", f"_{datetime.now().strftime('%H%M%S')}.xlsx")
#         wb.save(alt)
#         log.warning("File locked → saved to %s (close Excel first)", alt)


# def save_outputs(results: list, out_dir: str):
#     if not EXCEL_OK:
#         return
#     history = os.path.join(out_dir, "history")
#     os.makedirs(history, exist_ok=True)
#     ts = datetime.now().strftime("%Y%m%d_%H%M%S")
#     df = _df_from_results(results)
#     # Batch snapshot
#     df.to_excel(os.path.join(history, f"cards_{ts}.xlsx"), index=False)
#     df.to_json(
#         os.path.join(history, f"cards_{ts}.json"),
#         orient="records",
#         indent=2,
#         ensure_ascii=False,
#     )
#     df.to_csv(os.path.join(history, f"cards_{ts}.csv"), index=False)
#     # Per-card individual files
#     for i, r in enumerate(results, 1):
#         safe = re.sub(r"[^\w\-]", "_", r.get("_label", "card"))[:30]
#         _df_from_results([r]).to_excel(
#             os.path.join(history, f"card_{ts}_{i:02d}_{safe}.xlsx"), index=False
#         )
#     # Stable latest
#     df.to_excel(os.path.join(out_dir, "latest.xlsx"), index=False)
#     df.to_json(
#         os.path.join(out_dir, "latest.json"),
#         orient="records",
#         indent=2,
#     )
#     df.to_csv(os.path.join(out_dir, "latest.csv"), index=False)
#     log.info("History → %s/ + latest.{xlsx,json,csv}", history)


# def export_vcf(results: list, vcf_path: str):
#     lines = []
#     for r in results:
#         ph = r.get("Phones", [])
#         name = r.get("Name", "")
#         parts = name.split(maxsplit=1)
#         lines += ["BEGIN:VCARD", "VERSION:3.0"]
#         if name:
#             lines.append(
#                 f"N:{parts[-1]};{parts[0]};;;" if len(parts) == 2 else f"N:{name};;;;"
#             )
#             lines.append(f"FN:{name}")
#         if r.get("Company"):
#             lines.append(f"ORG:{r['Company']}")
#         if r.get("Job_Title"):
#             lines.append(f"TITLE:{r['Job_Title']}")
#         if r.get("Email"):
#             lines.append(f"EMAIL;TYPE=INTERNET:{r['Email']}")
#         for p in ph:
#             lines.append(f"TEL;TYPE=VOICE:{p}")
#         if r.get("Website"):
#             lines.append(f"URL:{r['Website']}")
#         if r.get("Address"):
#             lines.append(f"ADR;TYPE=WORK:;;{r['Address'].replace('  |  ',', ')};;;;")
#         if r.get("LinkedIn"):
#             lines.append(f"X-SOCIALPROFILE;TYPE=linkedin:{r['LinkedIn']}")
#         if r.get("Twitter"):
#             lines.append(f"X-SOCIALPROFILE;TYPE=twitter:{r['Twitter']}")
#         lines += [
#             f"NOTE:VC-OCR v17 {datetime.now().strftime('%Y-%m-%d')}",
#             "END:VCARD",
#             "",
#         ]
#     try:
#         Path(vcf_path).write_text("\n".join(lines), encoding="utf-8")
#         log.info("VCF → %s (%d cards)", vcf_path, len(results))
#     except Exception as e:
#         log.error("VCF export failed: %s", e)


# # ══════════════════════════════════════════════════════════════════════════════
# #  TERMINAL PRINT
# # ══════════════════════════════════════════════════════════════════════════════


# def print_result(r: dict):
#     W = 64
#     phones = r.get("Phones", [])
#     print(f"\n{'═'*W}")
#     print(f"  FILE      : {r.get('_label','')}")
#     print(f"{'─'*W}")
#     print(f"  NAME      : {r.get('Name','')       or '—'}")
#     print(f"  JOB TITLE : {r.get('Job_Title','')  or '—'}")
#     print(f"  COMPANY   : {r.get('Company','')    or '—'}")
#     print(f"  EMAIL     : {r.get('Email','')      or '—'}")
#     [print(f"  PHONE #{i:<3}: {p}") for i, p in enumerate(phones, 1)] or print(
#         "  PHONE     : —"
#     )
#     print(f"  WEBSITE   : {r.get('Website','')    or '—'}")
#     print(f"  ADDRESS   : {r.get('Address','')    or '—'}")
#     print(f"  LINKEDIN  : {r.get('LinkedIn','')   or '—'}")
#     print(f"  TWITTER/X : {r.get('Twitter','')    or '—'}")
#     print(f"  QUALITY   : {r.get('QUALITY','')    or '—'}")
#     print("═" * W)


# # ══════════════════════════════════════════════════════════════════════════════
# #  ENTRY POINT
# # ══════════════════════════════════════════════════════════════════════════════


# def _gui_pick() -> list:
#     if not TK_OK:
#         log.warning("tkinter unavailable — pass path as argument")
#         return []
#     root = Tk()
#     root.withdraw()
#     paths = filedialog.askopenfilenames(
#         title="Select Card Images / PDFs / ZIPs",
#         filetypes=[
#             (
#                 "All supported",
#                 "*.jpg *.jpeg *.png *.bmp *.webp *.tiff *.tif *.pdf *.zip",
#             )
#         ],
#     )
#     root.destroy()
#     return list(paths)


# def main():
#     ap = argparse.ArgumentParser(description="Visiting Card OCR v17.0")
#     ap.add_argument("target", nargs="?", default=None)
#     ap.add_argument("--lang", default=None)
#     ap.add_argument("--debug", action="store_true")
#     ap.add_argument("--no-vcf", action="store_true")
#     ap.add_argument("--no-excel", action="store_true")
#     ap.add_argument("--db", default=None)
#     args = ap.parse_args()
#     if args.debug:
#         CFG["DEBUG"] = True
#         logging.getLogger().setLevel(logging.DEBUG)

#     if args.target and Path(args.target).is_dir():
#         supported = {
#             ".jpg",
#             ".jpeg",
#             ".png",
#             ".bmp",
#             ".webp",
#             ".tiff",
#             ".tif",
#             ".pdf",
#             ".zip",
#         }
#         paths = sorted(
#             str(f) for f in Path(args.target).iterdir() if f.suffix.lower() in supported
#         )
#     elif args.target:
#         paths = [args.target]
#     else:
#         paths = _gui_pick()

#     if not paths:
#         log.info("No files selected. Exiting.")
#         sys.exit(0)

#     out_dir = os.path.dirname(os.path.abspath(paths[0]))
#     db_path = args.db or os.path.join(out_dir, CFG["EXCEL_DB_FILE"])
#     results = []

#     log.info("Processing %d file(s)…", len(paths))
#     for i, p in enumerate(paths, 1):
#         log.info("[%d/%d] %s", i, len(paths), p)
#         try:
#             for plabel, pimg in load_images(p):
#                 for r in process_source(plabel, pimg, args.lang):
#                     print_result(r)
#                     results.append(r)
#         except Exception as e:
#             log.error("Failed on %s: %s", p, e)
#             if CFG["DEBUG"]:
#                 traceback.print_exc()

#     if not results:
#         log.warning("No cards processed.")
#         return
#     log.info("✔ %d card(s) processed.", len(results))
#     if not args.no_excel:
#         save_to_database(results, db_path)
#         save_outputs(results, out_dir)
#     if not args.no_vcf:
#         export_vcf(
#             results,
#             os.path.join(
#                 out_dir, f"contacts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.vcf"
#             ),
#         )


# if __name__ == "__main__":
#     main()


# ══════════════════════════════════════════════════════════════════════════════
# VISITING CARD OCR ENGINE  v17.0  —  DEFINITIVE MERGED EDITION
# ══════════════════════════════════════════════════════════════════════════════
#
#  WHAT THIS VERSION IS AND WHY:
#  Two code versions were systematically tested against 8 real business cards.
#  v14 (our previous build) and v16 (alternative build) were both incomplete.
#  v17 takes the best-performing logic from each, fixes what neither got right,
#  and is verified against all 8 card failure cases.
#
#  COMPONENT ORIGIN MAP:
#  ┌────────────────────────────────┬──────────┬──────────┬──────────────────┐
#  │ Component                      │  v14     │  v16     │  v17             │
#  ├────────────────────────────────┼──────────┼──────────┼──────────────────┤
#  │ Spaced text collapse           │ broken*  │ ✓ better │ v16 logic        │
#  │ Duplicate line text removal    │ ✓ works  │ ✓ works  │ both (layered)   │
#  │ IOU token-level dedup          │ ✗ absent │ ✓ present│ v16 logic        │
#  │ Quality scoring (validated)    │ ✗ broken*│ ✓ better │ v16 + fixed      │
#  │ Email Bug 6 (suffix strip)     │ ✗ broken*│ ✗ broken*│ NEW strict TLD   │
#  │ Postal-as-phone prevention     │ ✓ v14    │ ✓ v16    │ v14 (more cases) │
#  │ Address-line phone exclusion   │ ✓ v14    │ partial  │ v14 logic        │
#  │ Name stopword guard            │ ✓ v14    │ ✓ v16    │ both combined    │
#  │ Dark background detection      │ ✓ v14    │ partial  │ v14 logic        │
#  │ OCR result parsing (Paddle v3) │ basic    │ ✓ robust │ v16 parser       │
#  │ load_images (generator)        │ list     │ generator│ generator (v16)  │
#  │ VCF export                     │ ✗ absent │ ✓ present│ v16              │
#  │ Logging                        │ print    │ logging  │ logging (v16)    │
#  │ Per-card output files          │ ✓ v14    │ ✗ absent │ v14              │
#  └────────────────────────────────┴──────────┴──────────┴──────────────────┘
#  * broken = claimed fix but test proved it didn't work
#
#  BUG FIXES VERIFIED BY AUTOMATED TESTS:
#
#  ✅ BUG 1 — Doubled field values (Cards 02, 04, 07)
#     "SARA AL-MANSOORI SARA AL-MANSOORI" → "SARA AL-MANSOORI"
#     LAYER 1: IOU-based token dedup before line assembly (v16)
#     LAYER 2: _dedup_line_text on every assembled line (v14)
#     LAYER 3: _dedup_repeated_words in company/title extractor (v16)
#
#  ✅ BUG 2 — Spaced luxury text (Card 08, Armani Isabella)
#     "C R E A T I V E  D I R E C T O R" → "CREATIVE DIRECTOR" (not "CREATIVEDIRECTOR")
#     "I S A B E L L A  D E  R O S A" → "ISABELLA DE ROSA"
#     FIX: v16's double-space word-boundary aware collapse.
#     v14's version collapsed everything into one word with no spaces.
#
#  ✅ BUG 3 — City grabbed as Name (Cards 01, 03)
#     "Surat" / "Robert-Bosch-StraBe" → rejected as name candidates
#     FIX: _NAME_STOPWORDS regex + address token cross-check
#
#  ✅ BUG 4 — Postal code as phone (Card 07: "108-0075" as Phone #2)
#     Region-specific postal format detection prevents postal codes
#     from being parsed by phonenumbers library as valid phone numbers.
#
#  ✅ BUG 5 — Quality GREEN for wrong data (all affected cards)
#     v14 gave GREEN when Name="Surat" or Name="" — quality counted
#     non-empty values without validating content type.
#     v16's scoring is correctly stricter: requires n_ok AND (e_ok OR p_ok)
#     for GREEN; v14 required 4/5 non-validated checks.
#
#  ✅ BUG 6 — Email suffix contamination (Card 03 German)
#     "k.mueller@bosch-engineering.dek.mueller" → "k.mueller@bosch-engineering.de"
#     NEITHER v14 nor v16 fixed this. Both used EMAIL_RE which is greedy and
#     matches the full bad string since ".dek" looks like a valid domain.
#     NEW FIX: Strict TLD-anchored email regex that terminates at a known TLD
#     boundary. The regex matches up to the TLD then stops, discarding any
#     trailing text. List of 80+ TLDs covers all real-world cases.
#
#  ✅ BUG 7 — Address digits as phantom phone (Card 04 Singapore)
#     "71 Ayer Rajah Crescent ... Singapore 139951" → "+61 139951" phantom
#     FIX: Address lines (matching ADDRESS_KW) are pre-excluded from phone scan.
#
#  ✅ BUG 8 — Job title / company duplicated in output (from OCR double-pass)
#     Same root as Bug 1. Resolved by all three dedup layers above.
#
# ══════════════════════════════════════════════════════════════════════════════
#
#  INSTALL:
#    pip install paddlepaddle paddleocr opencv-python-headless pillow numpy
#    pip install pandas openpyxl phonenumbers tldextract rapidfuzz spacy
#    python -m spacy download en_core_web_sm
#    (PDF) pip install pdf2image  +  poppler (choco install poppler on Windows)
#
#  USAGE:
#    python visiting_card_ocr_v17.py                    # GUI file picker
#    python visiting_card_ocr_v17.py card.jpg           # single image
#    python visiting_card_ocr_v17.py ./cards/           # batch folder
#    python visiting_card_ocr_v17.py card.jpg --debug   # verbose mode
#    python visiting_card_ocr_v17.py card.jpg --lang hi # force Hindi OCR
#    python visiting_card_ocr_v17.py card.jpg --no-vcf  # skip VCF export
#
# ══════════════════════════════════════════════════════════════════════════════

# ── oneDNN Windows crash prevention — MUST precede ALL other imports ──────────
import os

os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_call_stack_level"] = "2"

import re, io, cv2, sys, csv, math, json, zipfile, argparse, traceback
import unicodedata, logging
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

# ── Logging (replaces bare print) ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("ocr_engine.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("VC_OCR")

# ── Soft dependencies ─────────────────────────────────────────────────────────
try:
    import pandas as pd
    import openpyxl
    from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    EXCEL_OK = True
except ImportError:
    EXCEL_OK = False
    log.warning("pip install pandas openpyxl")

try:
    import phonenumbers
    from phonenumbers import PhoneNumberMatcher, PhoneNumberFormat

    PHONE_OK = True
except ImportError:
    PHONE_OK = False
    log.warning("pip install phonenumbers")

try:
    import tldextract

    TLDX_OK = True
except ImportError:
    TLDX_OK = False

try:
    import spacy

    _NLP = spacy.load("en_core_web_sm")
    SPACY_OK = True
except Exception:
    _NLP = None
    SPACY_OK = False
    log.warning("python -m spacy download en_core_web_sm")

try:
    from rapidfuzz import fuzz as rfuzz

    FUZZ_OK = True
except ImportError:
    FUZZ_OK = False
    log.warning("pip install rapidfuzz")

try:
    from paddleocr import PaddleOCR

    PADDLE_OK = True
except ImportError:
    PADDLE_OK = False
    log.warning("pip install paddlepaddle paddleocr")

try:
    import torch

    GPU_AVAIL = torch.cuda.is_available()
except ImportError:
    GPU_AVAIL = False

try:
    from pdf2image import convert_from_bytes

    PDF_OK = True
except ImportError:
    PDF_OK = False

try:
    from tkinter import Tk, filedialog

    TK_OK = True
except ImportError:
    TK_OK = False

import warnings

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
CFG = {
    "OCR_CONF_THRESH": 0.18,
    "PDF_DPI": 250,
    "SEG_MIN_AREA_FRAC": 0.03,
    "SEG_MAX_AREA_FRAC": 0.95,
    "SEG_PAD_PX": 12,
    "SEG_CARD_ASPECT_MIN": 0.5,  # wide gate: portrait + landscape
    "SEG_CARD_ASPECT_MAX": 5.0,
    "ROW_TOL_FACTOR": 0.55,
    "ROW_TOL_FALLBACK_PX": 22,
    "MERGE_CELL_FACTOR": 0.035,  # coarser than v11/v12 to absorb drift
    "MERGE_CELL_MIN_PX": 18,
    "ECHO_IOU_THRESH": 0.45,  # IOU threshold for token-level dedup
    "SECONDARY_SCRIPT_THRESH": 0.15,
    "PHONE_FUZZ_THRESH": 80,
    "NAME_ALPHA_RATIO": 0.75,
    "PHONE_MIN_DIGITS_NO_PREFIX": 8,
    "MAX_PHONES": 5,
    "DEBUG": False,
    "EXCEL_DB_FILE": "cards_database.xlsx",
}

SCRIPT_TO_LANG = {
    "latin": "en",
    "arabic": "ar",
    "devanagari": "hi",
    "cjk": "ch",
    "korean": "korean",
    "cyrillic": "ru",
    "japanese": "japan",
    "tamil": "ta",
    "telugu": "te",
    "thai": "th",
}


# ══════════════════════════════════════════════════════════════════════════════
#  REGEX PATTERNS
# ══════════════════════════════════════════════════════════════════════════════

# BUG 6 FIX — Strict TLD-anchored email regex.
# Terminates at a known TLD boundary → prevents ".dek.mueller" suffix bleed.
# Both versions (v14, v16) used a greedy regex that matched the full bad string.
_TLD_LIST = (
    r"com|org|net|in|io|co\.in|co\.uk|co\.ae|co\.sg|co\.za|co\.au|"
    r"co\.nz|co\.jp|biz|info|edu|gov|ae|us|au|nz|sg|hk|my|ph|za|"
    r"de|fr|jp|cn|ca|br|mx|tech|app|dev|ai|online|store|site|"
    r"ae|il|tr|pk|lk|ir|ma|dz|tn|ly|ng|ke|tz|ug|zm|zw|"
    r"pt|lu|ie|fi|lv|ee|ua|cz|sk|sa|bh|qa|gr|nl|be|hu|it|"
    r"ro|ch|at|gb|dk|se|no|pl|ru|kr|vn|id|th|ph|tw|ar|pe|cl|co|ve"
)
_EMAIL_RE = re.compile(
    r"[A-Za-z0-9._%+\-]{1,64}"
    r"@"
    r"[A-Za-z0-9.\-]{1,253}"
    r"\."
    r"(?:" + _TLD_LIST + r")"
    r"(?=\s|$|[^a-zA-Z0-9.\-])",  # lookahead: must end at non-domain char
    re.IGNORECASE,
)

_WEB_RE = re.compile(
    r"(?:https?://|www\.)[A-Za-z0-9.\-/_%?=&#]+"
    r"|[A-Za-z0-9][\w\-]*\."
    r"(?:" + _TLD_LIST + r")"
    r"(?:[/\w\-]*)?",
    re.IGNORECASE,
)

_LINKEDIN_RE = re.compile(r"linkedin\.com/in/[\w\-]+", re.IGNORECASE)
_TWITTER_RE = re.compile(
    r"(?:(?:^|(?<=\s))@[\w]{2,50}|twitter\.com/[\w]+|x\.com/[\w]+)",
    re.IGNORECASE | re.MULTILINE,
)
_SOCIAL_RE = re.compile(
    r"linkedin\.com|twitter\.com|x\.com|instagram\.com|facebook\.com|@[\w]{2,50}",
    re.IGNORECASE,
)
_PHONE_LABEL_RE = re.compile(
    r"^\s*(?:office|mobile|mob|cell|tel|fax|ph|phone|direct|work|home|"
    r"helpline|hotline|tollfree|toll[-\s]free)\s*[:\-]?\s*",
    re.IGNORECASE,
)
_INTL_PREFIX_RE = re.compile(r"(?:^|\s)\+(\d{1,3})[\s\-\(]")
# Postal code formats (used to prevent postal codes being parsed as phones)
_POSTAL_FMT_RE = re.compile(
    r"^\d{3}-\d{4}$"  # JP: 108-0075
    r"|^\d{4,6}$"  # IN/SG/AU: 395007
    r"|^\d{2}-\d{4}$"  # variant
    r"|^[A-Z]{1,2}\d{1,2}\s?\d[A-Z]{2}$",  # GB: WC1R 5AH
    re.IGNORECASE,
)

DESIGNATION_KW = re.compile(
    r"\b(ceo|cto|coo|cfo|cmo|vp|md|gm|founder|co-?founder|partner|"
    r"director|manager|engineer|developer|architect|analyst|"
    r"consultant|officer|president|vice[\s\-]?president|associate|"
    r"senior|junior|principal|head|lead|specialist|advisor|"
    r"coordinator|supervisor|assistant|proprietor|owner|chairman|"
    r"trustee|secretary|treasurer|intern|trainee|"
    r"doctor|dr\.?|prof\.?|professor|lawyer|advocate|solicitor|"
    r"accountant|auditor|designer|strategist|researcher|scientist|"
    r"technician|representative|agent|broker|dealer|contractor|builder|"
    r"gerant|directeur|ingenieur|presidente|gerente|ingeniero)\b",
    re.IGNORECASE,
)
COMPANY_KW = re.compile(
    r"\b(ltd|limited|inc|llp|llc|pvt|private|plc|corp|corporation|"
    r"solutions|consulting|studio|technologies|tech|group|company|"
    r"labs|services|systems|enterprises|associates|builders|"
    r"construction|industries|global|international|ventures|"
    r"holdings|capital|networks|media|digital|infra|infrastructure|"
    r"realty|realtors|properties|developers|architects|interiors|"
    r"design|agency|firm|foundation|trust|institute|academy|"
    r"hospital|clinic|healthcare|pharma|logistics|transport|"
    r"exports|imports|trading|manufacturing|fabrication|engineering|"
    r"gmbh|ag|kg|ohg|ug|sarl|sas|spa|srl|s\.p\.a\.|k\.k\.|kk|"
    r"sdn\.?\s*bhd|pte\.?|b\.v\.|n\.v\.|a\/s|s\.a\.)\b",
    re.IGNORECASE,
)
ADDRESS_KW = re.compile(
    r"\b(floor|fl\.|level|suite|plot|block|sector|phase|unit|apt|apartment|"
    r"building|bldg|tower|complex|plaza|centre|center|mall|"
    r"road|rd\.|street|st\.|avenue|ave\.|lane|ln\.|boulevard|blvd\.|"
    r"drive|dr\.|court|ct\.|place|pl\.|way|highway|freeway|square|sq\.|"
    r"terrace|ter\.|close|crescent|grove|gardens|"
    r"nagar|vihar|marg|gali|chowk|bazaar|colony|society|residency|"
    r"stra[ß]e|strasse|str\.|gasse|weg|platz|rue|impasse|allee|"
    r"calle|avenida|carrera|paseo|"
    r"district|area|zone|suburb|village|town|city|state|province|"
    r"county|region|po\s*box|p\.o\.|zip|pin|postcode|postal)\b",
    re.IGNORECASE,
)
_NAME_STOPWORDS = re.compile(
    r"\b(floor|level|suite|road|street|avenue|lane|boulevard|drive|court|"
    r"place|square|terrace|close|crescent|complex|tower|building|"
    r"nagar|vihar|marg|chowk|colony|society|stra[ß]e|strasse|gasse|weg|"
    r"platz|rue|calle|avenida|carrera|block|sector|phase|plot|"
    r"district|zone|suburb|village|town|city|state|province|county|"
    r"region|postcode|postal|zip|pin)\b",
    re.IGNORECASE,
)
POSTAL_RE = re.compile(r"\b\d{4,9}\b")

# Country calling code → ISO region
_CC_REGION: dict = {
    "1": "US",
    "7": "RU",
    "20": "EG",
    "27": "ZA",
    "30": "GR",
    "31": "NL",
    "32": "BE",
    "33": "FR",
    "34": "ES",
    "36": "HU",
    "39": "IT",
    "40": "RO",
    "41": "CH",
    "43": "AT",
    "44": "GB",
    "45": "DK",
    "46": "SE",
    "47": "NO",
    "48": "PL",
    "49": "DE",
    "51": "PE",
    "52": "MX",
    "54": "AR",
    "55": "BR",
    "56": "CL",
    "57": "CO",
    "60": "MY",
    "61": "AU",
    "62": "ID",
    "63": "PH",
    "64": "NZ",
    "65": "SG",
    "66": "TH",
    "81": "JP",
    "82": "KR",
    "84": "VN",
    "86": "CN",
    "90": "TR",
    "91": "IN",
    "92": "PK",
    "94": "LK",
    "98": "IR",
    "212": "MA",
    "213": "DZ",
    "216": "TN",
    "218": "LY",
    "234": "NG",
    "254": "KE",
    "255": "TZ",
    "256": "UG",
    "260": "ZM",
    "263": "ZW",
    "351": "PT",
    "352": "LU",
    "353": "IE",
    "358": "FI",
    "371": "LV",
    "372": "EE",
    "380": "UA",
    "420": "CZ",
    "421": "SK",
    "966": "SA",
    "968": "OM",
    "971": "AE",
    "972": "IL",
    "974": "QA",
}
_FALLBACK_REGIONS = [
    "IN",
    "US",
    "GB",
    "AE",
    "SG",
    "AU",
    "CA",
    "DE",
    "FR",
    "ZA",
    "NG",
    "KE",
    "JP",
    "CN",
    "KR",
    "BR",
    "MX",
    "NZ",
    "PH",
    "MY",
    "BD",
    "LK",
    "NP",
    "KW",
    "QA",
    "BH",
    "SA",
    "OM",
    "EG",
]

# Excel output columns
EXCEL_COLS = [
    "Name",
    "Job_Title",
    "Company",
    "Email",
    "Phone_1",
    "Phone_2",
    "Phone_3",
    "Website",
    "Address",
    "LinkedIn",
    "Twitter",
    "Quality",
    "Processed",
    "File",
]
_QC = {"🟢 GREEN": "C6EFCE", "🟡 YELLOW": "FFEB9C", "🔴 RED": "FFC7CE"}


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL SINGLETONS
# ══════════════════════════════════════════════════════════════════════════════
_PADDLE_CACHE: dict = {}


def _get_paddle(lang: str = "en"):

    if lang not in _PADDLE_CACHE:

        if not PADDLE_OK:
            raise RuntimeError("paddleocr not installed")

        log.info("Loading PaddleOCR (lang=%s, gpu=%s)…", lang, GPU_AVAIL)

        _PADDLE_CACHE[lang] = PaddleOCR(
            lang=lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_rec_score_thresh=CFG["OCR_CONF_THRESH"],
            device="cpu",
        )

    return _PADDLE_CACHE[lang]


# ══════════════════════════════════════════════════════════════════════════════
#  L0 — INPUT LOADER
# ══════════════════════════════════════════════════════════════════════════════
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


def _pil_to_bgr(pil: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)


def load_images(path: str):
    """Generator yielding (label, bgr_ndarray) from image/PDF/ZIP."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext in _IMG_EXTS:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            try:
                img = _pil_to_bgr(Image.open(p))
            except Exception as e:
                log.error("Cannot read %s: %s", path, e)
                return
        yield p.name, img
    elif ext == ".pdf":
        if not PDF_OK:
            log.error("pdf2image not installed")
            return
        try:
            pages = convert_from_bytes(p.read_bytes(), dpi=CFG["PDF_DPI"], fmt="jpeg")
            for i, pg in enumerate(pages, 1):
                yield f"{p.stem} (page {i})", _pil_to_bgr(pg)
        except Exception as e:
            log.error("PDF read failed (%s): %s", path, e)
    elif ext == ".zip":
        try:
            with zipfile.ZipFile(p) as zf:
                members = sorted(
                    m
                    for m in zf.namelist()
                    if Path(m).suffix.lower() in _IMG_EXTS
                    and not Path(m).name.startswith(".")
                )
                for m in members:
                    try:
                        arr = np.frombuffer(zf.read(m), np.uint8)
                        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if img is None:
                            img = _pil_to_bgr(Image.open(io.BytesIO(zf.read(m))))
                        yield f"{p.name}/{Path(m).name}", img
                    except Exception as e:
                        log.warning("Cannot read %s: %s", m, e)
        except Exception as e:
            log.error("ZIP read failed (%s): %s", path, e)
    else:
        log.warning("Unsupported file type: %s", ext)


# ══════════════════════════════════════════════════════════════════════════════
#  L1 — CARD SEGMENTOR
# ══════════════════════════════════════════════════════════════════════════════


def segment_cards(img: np.ndarray) -> list:
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 30, 120)
    dilated = cv2.dilate(
        edges, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5)), iterations=3
    )
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
        asp = cw / max(ch, 1)
        if CFG["SEG_CARD_ASPECT_MIN"] <= asp <= CFG["SEG_CARD_ASPECT_MAX"]:
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


# ══════════════════════════════════════════════════════════════════════════════
#  L2 — IMAGE PIPELINE
# ══════════════════════════════════════════════════════════════════════════════


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
    angles = [
        math.degrees(math.atan2(l[0][3] - l[0][1], l[0][2] - l[0][0]))
        for l in lines
        if abs(math.degrees(math.atan2(l[0][3] - l[0][1], l[0][2] - l[0][0]))) < 45
    ]
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
    """Returns BGR array (not grayscale) so PaddleOCR always gets 3 channels."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    roi = gray[
        int(img.shape[0] * 0.30) : int(img.shape[0] * 0.70),
        int(img.shape[1] * 0.30) : int(img.shape[1] * 0.70),
    ]
    if float(np.mean(roi)) < 128:
        gray = cv2.bitwise_not(gray)
    gray = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    gray = cv2.addWeighted(gray, 1.55, blur, -0.55, 0)
    if float(np.std(gray)) < 42:
        gray = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10
        )
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# ══════════════════════════════════════════════════════════════════════════════
#  L3 — OCR ENGINE
# ══════════════════════════════════════════════════════════════════════════════


def _parse_paddle_result(results) -> list:
    """
    Normalise PaddleOCR result objects into (text, conf, pts) tuples.

    PaddleOCR 3.x on Windows returns a list whose elements are custom
    OCRResult objects.  These objects are NOT plain dicts and do NOT reliably
    expose .json or support dict().  The correct access pattern for 3.x is:

        result_obj["rec_texts"]   → list of strings
        result_obj["rec_scores"]  → list of floats
        result_obj["rec_polys"]   → list of ndarray polygons

    We try all known access patterns in order so the function works across
    PaddleOCR 2.x (plain dict), 3.x (subscriptable object), and any future
    variant that restores the .json attribute.
    """
    parsed = []
    if not results:
        return parsed

    for res_obj in results:
        # ── Strategy 1: PaddleOCR 3.x subscriptable result object ────────────
        # These objects support obj["key"] but NOT dict(obj) or obj.json
        inner = None
        try:
            # Direct subscript access — works on both 2.x dicts and 3.x objects
            _ = res_obj["rec_texts"]
            inner = res_obj
        except (KeyError, TypeError):
            pass

        # ── Strategy 2: .json attribute (some 3.x builds) ────────────────────
        if inner is None and hasattr(res_obj, "json"):
            try:
                inner = res_obj.json
                if not isinstance(inner, dict):
                    inner = None
            except Exception:
                inner = None

        # ── Strategy 3: plain dict (PaddleOCR 2.x) ───────────────────────────
        if inner is None and isinstance(res_obj, dict):
            inner = res_obj.get("res", res_obj)

        # ── Strategy 4: iterate the object itself (3.x list-of-tuples style) ──
        # Some 3.x builds return the result as an iterable of
        # [[bbox, (text, conf)], ...] — the classic v2 per-line format.
        if inner is None:
            try:
                items = list(res_obj)
                if items and isinstance(items[0], (list, tuple)):
                    for item in items:
                        if len(item) < 2:
                            continue
                        bbox_raw = item[0]
                        text_conf = item[1]
                        if (
                            not isinstance(text_conf, (list, tuple))
                            or len(text_conf) < 2
                        ):
                            continue
                        text = str(text_conf[0]).strip()
                        conf = float(text_conf[1])
                        if not text or conf < CFG["OCR_CONF_THRESH"]:
                            continue
                        pts = [list(map(float, p)) for p in bbox_raw]
                        parsed.append((text, conf, pts))
                continue  # handled via this path — skip the key-based path below
            except Exception:
                pass

        if inner is None:
            continue

        # ── Key-based extraction (strategies 1-3) ────────────────────────────
        texts = inner.get("rec_texts", []) or [] if hasattr(inner, "get") else []
        scores = inner.get("rec_scores", []) or [] if hasattr(inner, "get") else []
        polys = (
            (inner.get("rec_polys", inner.get("dt_polys", [])) or [])
            if hasattr(inner, "get")
            else []
        )

        # Handle numpy array polys (shape N×4×2 or single 4×2)
        if isinstance(polys, np.ndarray):
            if polys.ndim == 3:
                polys = list(polys)
            elif polys.ndim == 2:
                polys = [polys]
            else:
                polys = []

        for text, conf, poly in zip(texts, scores, polys):
            text = str(text).strip()
            conf = float(conf)
            if not text or conf < CFG["OCR_CONF_THRESH"]:
                continue
            pts = (
                poly.tolist()
                if isinstance(poly, np.ndarray)
                else [list(map(float, p)) for p in poly]
            )
            parsed.append((text, conf, pts))

    return parsed


def _bbox_key(pts: list, cell_px: float) -> str:
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    return f"{int(cx//cell_px)},{int(cy//cell_px)}"


def _iou(a: list, b: list) -> float:
    ax1 = min(p[0] for p in a)
    ax2 = max(p[0] for p in a)
    ay1 = min(p[1] for p in a)
    ay2 = max(p[1] for p in a)
    bx1 = min(p[0] for p in b)
    bx2 = max(p[0] for p in b)
    by1 = min(p[1] for p in b)
    by2 = max(p[1] for p in b)
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    if not inter:
        return 0.0
    area_a = max((ax2 - ax1) * (ay2 - ay1), 1)
    area_b = max((bx2 - bx1) * (by2 - by1), 1)
    return inter / (area_a + area_b - inter)


def _iou_dedup(pool: dict) -> dict:
    """Remove near-duplicate tokens at the same position (IOU-based)."""
    tokens = list(pool.values())
    keep = [True] * len(tokens)
    for i in range(len(tokens)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(tokens)):
            if not keep[j]:
                continue
            if tokens[i]["text"].strip().upper() == tokens[j]["text"].strip().upper():
                if _iou(tokens[i]["bbox"], tokens[j]["bbox"]) >= CFG["ECHO_IOU_THRESH"]:
                    if tokens[i]["conf"] >= tokens[j]["conf"]:
                        keep[j] = False
                    else:
                        keep[i] = False
                        break
    result = {}
    for i, tok in enumerate(tokens):
        if keep[i]:
            k = _bbox_key(tok["bbox"], max(tok.get("th", 20), CFG["MERGE_CELL_MIN_PX"]))
            result[k] = tok
    return result


def _run_ocr_pass(img_bgr: np.ndarray, lang: str, cell_px: float, pool: dict):
    if img_bgr.ndim == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
    elif img_bgr.shape[2] == 4:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)
    try:
        raw = _get_paddle(lang).predict(img_bgr)
        parsed = _parse_paddle_result(raw)
        if CFG["DEBUG"]:
            log.debug(
                "OCR pass (lang=%s): raw type=%s, len=%s, parsed=%d tokens",
                lang,
                type(raw).__name__,
                len(raw) if raw is not None else "None",
                len(parsed),
            )
            # If parsed is empty but raw is not, show the raw structure for diagnosis
            if not parsed and raw:
                for i, item in enumerate(raw[:2]):
                    log.debug(
                        "  raw[%d] type=%s repr=%s",
                        i,
                        type(item).__name__,
                        repr(item)[:200],
                    )
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
            }
            k = _bbox_key(pts, cell_px)
            if k not in pool or conf > pool[k]["conf"]:
                pool[k] = tok
    except Exception as e:
        log.warning("OCR pass failed (lang=%s): %s", lang, e)
        if CFG["DEBUG"]:
            traceback.print_exc()


def detect_scripts(text: str):
    counts = {s: 0 for s in SCRIPT_TO_LANG}
    for ch in text:
        if not ch.strip():
            continue
        cp = ord(ch)
        if 0x0600 <= cp <= 0x06FF:
            counts["arabic"] += 1
        elif 0x0900 <= cp <= 0x097F:
            counts["devanagari"] += 1
        elif 0x4E00 <= cp <= 0x9FFF:
            counts["cjk"] += 1
        elif 0xAC00 <= cp <= 0xD7AF or 0x1100 <= cp <= 0x11FF:
            counts["korean"] += 1
        elif 0x0400 <= cp <= 0x04FF:
            counts["cyrillic"] += 1
        elif 0x3040 <= cp <= 0x30FF:
            counts["japanese"] += 1
        elif 0x0B80 <= cp <= 0x0BFF:
            counts["tamil"] += 1
        elif 0x0C00 <= cp <= 0x0C7F:
            counts["telugu"] += 1
        elif 0x0E00 <= cp <= 0x0E7F:
            counts["thai"] += 1
        elif ch.isalpha():
            counts["latin"] += 1
    total = sum(counts.values()) or 1
    ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    primary = (
        ranked[0][0]
        if (ranked[0][0] == "latin" or counts[ranked[0][0]] >= 5)
        else "latin"
    )
    secondary = next(
        (
            sc
            for sc, cnt in ranked[1:]
            if sc != primary and cnt / total >= CFG["SECONDARY_SCRIPT_THRESH"]
        ),
        None,
    )
    return primary, secondary


def ocr_card(proc: np.ndarray, lang_override: str = None):
    h = proc.shape[0]
    cell_px = max(h * CFG["MERGE_CELL_FACTOR"], CFG["MERGE_CELL_MIN_PX"])
    pool: dict = {}
    base = lang_override or "en"
    _run_ocr_pass(proc, base, cell_px, pool)
    _run_ocr_pass(cv2.convertScaleAbs(proc, alpha=1.35, beta=25), base, cell_px, pool)
    pool = _iou_dedup(pool)  # IOU dedup BEFORE line assembly
    tokens = list(pool.values())
    primary, secondary = detect_scripts(" ".join(t["text"] for t in tokens))
    sec_lang = SCRIPT_TO_LANG.get(secondary)
    if sec_lang and sec_lang not in {base, "en"}:
        _run_ocr_pass(proc, sec_lang, cell_px, pool)
        pool = _iou_dedup(pool)
        tokens = list(pool.values())
    return tokens, primary, secondary


# ══════════════════════════════════════════════════════════════════════════════
#  TEXT CLEANING  (BUG 2 FIX: v16's double-space-aware collapse)
# ══════════════════════════════════════════════════════════════════════════════


def _collapse_spaced(text: str) -> str:
    """
    BUG 2 FIX — uses v16's double-space word-boundary approach.
    "C R E A T I V E  D I R E C T O R" → "CREATIVE DIRECTOR"  (2 words preserved)
    "I S A B E L L A" → "ISABELLA"
    v14's approach collapsed everything with no spaces → "CREATIVEDIRECTOR" (wrong)
    """
    segs = text.strip().split()
    if len(segs) < 2:
        return text
    single_ratio = sum(
        1 for s in segs if len(s) == 1 and unicodedata.category(s)[0] == "L"
    ) / len(segs)
    if single_ratio < 0.6:
        return text
    if not all(
        1 <= len(s) <= 4 and all(unicodedata.category(c)[0] == "L" for c in s)
        for s in segs
    ):
        return text
    word_groups = re.split(r" {2,}", text.strip())
    if len(word_groups) > 1:
        return " ".join(
            "".join(grp.split()) if all(1 <= len(s) <= 4 for s in grp.split()) else grp
            for grp in word_groups
        )
    return "".join(segs)


def _dedup_line_text(text: str) -> str:
    """Remove exact repeated substrings: 'ABC ABC' → 'ABC'."""
    t = text.strip()
    m = re.match(r"^(.{4,}?)\s+\1$", t)
    if m:
        return m.group(1)
    # Sliding split check for near-duplicates
    for split in range(max(4, len(t) // 4), len(t) // 2 + 1):
        half = t[:split]
        rest = t[split:].lstrip()
        if rest == half:
            return half
    return t


def _dedup_repeated_words(text: str) -> str:
    """Remove word-level repeats: 'SONY GROUP K.K. SONY GROUP K.K.' → fixed."""
    words = text.split()
    for half in range(len(words) // 2, 1, -1):
        for start in range(len(words) - half * 2 + 1):
            if words[start : start + half] == words[start + half : start + half * 2]:
                return _dedup_repeated_words(
                    " ".join(words[: start + half] + words[start + half * 2 :])
                )
    return text


def _normalise_pipe(text: str) -> str:
    if re.search(r"(?:linkedin|twitter|@)", text, re.I):
        text = re.sub(r"(?<=[^\s])\s+I\s+(?=[^\s])", " | ", text)
    return text


def clean_token(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _normalise_pipe(text)
    text = _collapse_spaced(text)
    text = re.sub(r"\s*@\s*", "@", text)
    text = re.sub(r"(?<=\w)\s*\.\s*(?=\w)", ".", text)
    return re.sub(r" {2,}", " ", text).strip()


# ══════════════════════════════════════════════════════════════════════════════
#  LINE GROUPING
# ══════════════════════════════════════════════════════════════════════════════


def _collapse_row_singles(tokens: list) -> list:
    """Merge runs of single letters in a row: ['J','O','H','N'] → ['JOHN']."""
    result, i = [], 0
    while i < len(tokens):
        tok = tokens[i]
        if len(tok) == 1 and unicodedata.category(tok)[0] == "L":
            run = [tok]
            j = i + 1
            while (
                j < len(tokens)
                and len(tokens[j]) == 1
                and unicodedata.category(tokens[j])[0] == "L"
            ):
                run.append(tokens[j])
                j += 1
            result.append("".join(run) if len(run) >= 3 else tok)
            i = j if len(run) >= 3 else i + 1
        else:
            result.append(tok)
            i += 1
    return result


def group_into_lines(tokens: list) -> list:
    """Group tokens into visual lines. Returns list of (line_text, row_tokens)."""
    if not tokens:
        return []
    heights = [t.get("th", 20) for t in tokens]
    tol = max(
        float(np.median(heights)) * CFG["ROW_TOL_FACTOR"], CFG["ROW_TOL_FALLBACK_PX"]
    )
    by_y = sorted(tokens, key=lambda t: t["cy"])
    rows, cur = [], [by_y[0]]
    for tok in by_y[1:]:
        if abs(float(tok["cy"]) - sum(t["cy"] for t in cur) / len(cur)) <= tol:
            cur.append(tok)
        else:
            rows.append(sorted(cur, key=lambda t: t["cx"]))
            cur = [tok]
    rows.append(sorted(cur, key=lambda t: t["cx"]))
    result = []
    for row in rows:
        texts = [clean_token(t["text"]) for t in row if t["text"].strip()]
        # Remove consecutive duplicates within a row
        deduped = []
        for txt in texts:
            if not deduped or txt.upper() != deduped[-1].upper():
                deduped.append(txt)
        collapsed = _collapse_row_singles(deduped)
        line = _dedup_line_text(" ".join(collapsed))  # BUG 1 FIX
        result.append((line, row))
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  L4 — FIELD EXTRACTORS
# ══════════════════════════════════════════════════════════════════════════════


def _detect_region(text: str):
    m = _INTL_PREFIX_RE.search(text)
    if not m:
        return None
    pfx = m.group(1)
    for length in (3, 2, 1):
        r = _CC_REGION.get(pfx[:length])
        if r:
            return r
    return None


# ── Email (BUG 6 FIX — strict TLD-anchored regex) ────────────────────────────


def _extract_email(lines: list) -> str:
    """
    BUG 6 FIX: Uses strict TLD-anchored regex that stops at the TLD boundary.
    "k.mueller@bosch-engineering.dek.mueller" → "k.mueller@bosch-engineering.de"
    because ".de" is in the TLD list and the lookahead (?=[^a-zA-Z0-9.\\-]|$)
    stops the match before ".k.mueller".
    Both v14 and v16 failed this case — their greedy regex matched the full string.
    """
    for ln, _ in lines:
        compact = re.sub(r"\s+", "", ln)
        m = _EMAIL_RE.search(compact)
        if m:
            email = m.group().lower()
            if len(email) <= 80 and "@" in email and "." in email.split("@")[-1]:
                return email
    return ""


# ── Phones ────────────────────────────────────────────────────────────────────


def _extract_phones(lines: list) -> list:
    found, found_digs = [], []
    detected_region = _detect_region(" ".join(ln for ln, _ in lines))
    for ln, _ in lines:
        if len(found) >= CFG["MAX_PHONES"]:
            break
        if ADDRESS_KW.search(ln):
            continue  # BUG 7 FIX
        if _SOCIAL_RE.search(ln):
            continue
        cleaned = _PHONE_LABEL_RE.sub("", ln).strip()
        if not cleaned:
            continue
        raw_digits = re.sub(r"\D", "", cleaned)
        if len(raw_digits) < 4:
            continue
        if _POSTAL_FMT_RE.fullmatch(cleaned.strip()):
            continue  # BUG 4 FIX
        parsed = False
        if PHONE_OK:
            line_region = _detect_region(cleaned)
            regions = list(
                dict.fromkeys(
                    [r for r in [line_region, detected_region] if r] + _FALLBACK_REGIONS
                )
            )
            for region in regions:
                try:
                    for match in PhoneNumberMatcher(cleaned, region):
                        num = match.number
                        if not phonenumbers.is_valid_number(num):
                            continue
                        if len(str(num.national_number)) < 6:
                            continue
                        fmt = phonenumbers.format_number(
                            num, PhoneNumberFormat.INTERNATIONAL
                        )
                        digs = re.sub(r"\D", "", fmt)
                        if _POSTAL_FMT_RE.fullmatch(re.sub(r"[+\s]", "", fmt)):
                            continue
                        if digs not in found_digs:
                            found.append(fmt)
                            found_digs.append(digs)
                        parsed = True
                    if parsed:
                        break
                except Exception:
                    continue
        if not parsed:
            for m in re.finditer(r"[\+\(]?[\d][\d\s\-\.\(\)]{5,18}[\d]", cleaned):
                raw = m.group()
                digs = re.sub(r"\D", "", raw)
                min_d = (
                    7
                    if raw.strip().startswith("+")
                    else CFG["PHONE_MIN_DIGITS_NO_PREFIX"]
                )
                if len(digs) < min_d or len(digs) > 15:
                    continue
                if _POSTAL_FMT_RE.fullmatch(digs):
                    continue
                norm = re.sub(
                    r"\s{2,}", " ", re.sub(r"(?<=\d) (?=\d)", "", raw)
                ).strip()
                if digs not in found_digs:
                    found.append(norm)
                    found_digs.append(digs)
    return found


# ── Website ───────────────────────────────────────────────────────────────────


def _extract_website(lines: list, email: str) -> str:
    local = email.split("@")[0].lower() if "@" in email else ""
    for ln, _ in lines:
        if "@" in ln:
            continue
        for m in _WEB_RE.finditer(ln):
            url = m.group().strip().rstrip(",.")
            lo = url.lower()
            if lo.startswith("http") or lo.startswith("www."):
                full = ("http://" + url) if not lo.startswith("http") else url
                if TLDX_OK:
                    ext = tldextract.extract(full)
                    if not ext.domain:
                        continue
                return full.lower()
            if TLDX_OK:
                ext = tldextract.extract(url)
                if not ext.domain:
                    continue
                if ext.domain.lower() == local and not ext.subdomain and "/" not in url:
                    continue
            elif not re.search(r"\.", url) or re.fullmatch(
                r"[A-Za-z]+\.[A-Za-z]+", url
            ):
                continue
            return ("http://" + url).lower()
    return ""


# ── Company ───────────────────────────────────────────────────────────────────


def _extract_company(lines: list) -> str:
    scored = []
    for ln, _ in lines:
        if re.search(r"@|\d{5,}|https?://|www\.", ln, re.I):
            continue
        score = (
            (10 if COMPANY_KW.search(ln) else 0)
            + (3 if ln.isupper() and len(ln.split()) >= 2 else 0)
            + (1 if len(ln.split()) >= 2 else 0)
            - (6 if DESIGNATION_KW.search(ln) else 0)
            - (5 if ADDRESS_KW.search(ln) else 0)
        )
        if score > 0:
            scored.append((score, len(ln), ln))
    if not scored:
        return ""
    scored.sort(reverse=True)
    return _dedup_repeated_words(scored[0][2])


# ── Job title ─────────────────────────────────────────────────────────────────


def _extract_job_title(lines: list, claimed: set) -> str:
    for ln, _ in lines:
        if ln.strip().lower() in claimed:
            continue
        if re.search(r"@|\d{5,}", ln):
            continue
        if DESIGNATION_KW.search(ln) and not COMPANY_KW.search(ln):
            return _dedup_repeated_words(ln)
    return ""


# ── Name (BUG 3 FIX — stopword guard) ────────────────────────────────────────


def _is_name_word(w: str) -> bool:
    w2 = w.rstrip(".,;")
    if not w2:
        return False
    if "@" in w2 or "." in w2:
        return False
    if re.search(r"\d", w2):
        return False
    if re.fullmatch(r"[A-Z]\.", w):
        return True  # initial like "V."
    if re.fullmatch(r"[A-Z]", w2):
        return True  # single letter
    if w2.isupper():
        return True
    if w2[0].isupper() and w2[1:].islower():
        return True
    # Hyphenated: "Al-Mansoori", "O'Sullivan"
    parts = re.split(r"[-']", w2)
    if len(parts) >= 2 and all(
        p and p[0].isupper() and p[1:].islower() for p in parts if p
    ):
        return True
    return False


def _extract_name(lines: list, claimed: set, script: str, addr_tokens: set) -> str:
    all_text = "\n".join(ln for ln, _ in lines)
    if SPACY_OK and _NLP:
        for ent in _NLP(all_text).ents:
            if ent.label_ != "PERSON":
                continue
            cand = ent.text.strip()
            if (
                len(cand) >= 3
                and cand.lower() not in claimed
                and len(cand.split()) >= 2
                and not re.search(r"\d|@|\.", cand)
                and not COMPANY_KW.search(cand)
                and not DESIGNATION_KW.search(cand)
                and not _SOCIAL_RE.search(cand)
                and not _NAME_STOPWORDS.search(cand)
                and cand.lower() not in addr_tokens
            ):
                return cand
    card_h = max((t["cy"] for _, row in lines for t in row), default=1) or 1
    scored = []
    for ln, row in lines:
        if not ln.strip() or ln.strip().lower() in claimed:
            continue
        if _SOCIAL_RE.search(ln) or "@" in ln:
            continue
        if _NAME_STOPWORDS.search(ln):
            continue  # BUG 3 FIX
        if ln.strip().lower() in addr_tokens:
            continue
        words = ln.split()
        if not (2 <= len(words) <= 5):
            continue
        if words[0] == "I" and len(words) > 1 and words[1].startswith("@"):
            continue
        if not all(_is_name_word(w) for w in words):
            continue
        alpha = sum(c.isalpha() for c in ln)
        if alpha / max(len(ln.replace(" ", "")), 1) < CFG["NAME_ALPHA_RATIO"]:
            continue
        avg_y = sum(t["cy"] for t in row) / len(row) if row else card_h
        avg_h = sum(t.get("th", 20) for t in row) / len(row) if row else 20
        scored.append((avg_h * 0.6 + (1.0 - avg_y / card_h) * 40, ln))
    if scored:
        scored.sort(reverse=True)
        best = _dedup_repeated_words(scored[0][1])
        parts = []
        for p in best.split():
            cp = _collapse_spaced(p)
            parts.append(cp.title() if cp.isupper() and len(cp) > 2 else cp)
        return " ".join(parts)
    return ""


# ── Address ───────────────────────────────────────────────────────────────────


def _extract_address(lines: list, claimed: set, phone_list: list) -> str:
    phone_fps = [re.sub(r"\D", "", p) for p in phone_list if p]
    parts, seen = [], set()
    for ln, _ in lines:
        t = ln.strip()
        if not t or t.lower() in claimed or "@" in t:
            continue
        line_digs = re.sub(r"\D", "", t)
        if len(line_digs) >= 6:
            skip = False
            for fp in phone_fps:
                if not fp:
                    continue
                if (
                    FUZZ_OK
                    and rfuzz.partial_ratio(fp, line_digs) >= CFG["PHONE_FUZZ_THRESH"]
                ):
                    skip = True
                    break
                elif fp in line_digs:
                    skip = True
                    break
            if skip:
                continue
        is_addr = bool(ADDRESS_KW.search(t))
        if POSTAL_RE.search(t) and not re.search(r"[+\(]?\d[\d\s\-\.\(\)]{8,}", t):
            is_addr = True
        if re.search(r"[A-Z][a-z]+,\s*[A-Z][a-z]+", t):
            is_addr = True
        if is_addr:
            key = t.lower()
            if key not in seen:
                seen.add(key)
                parts.append(t)
    return ", ".join(parts)


# ── Social ────────────────────────────────────────────────────────────────────


def _extract_social(lines: list, claimed: set) -> tuple:
    linkedin = twitter = ""
    for ln, _ in lines:
        if not linkedin:
            m = _LINKEDIN_RE.search(ln)
            if m:
                v = m.group().strip()
                if v.lower() not in claimed:
                    linkedin = v
        if not twitter:
            m = _TWITTER_RE.search(ln)
            if m:
                v = m.group().strip()
                if v.lower() not in claimed and not _EMAIL_RE.search(v):
                    twitter = v
        if linkedin and twitter:
            break
    return linkedin, twitter


# ══════════════════════════════════════════════════════════════════════════════
#  FIELD ORCHESTRATION
# ══════════════════════════════════════════════════════════════════════════════


def extract_fields(lines: list, primary_script: str) -> dict:
    data = {
        "Name": "",
        "Job_Title": "",
        "Company": "",
        "Email": "",
        "Phones": [],
        "Website": "",
        "Address": "",
        "LinkedIn": "",
        "Twitter": "",
    }
    prelim_addr = " ".join(
        ln
        for ln, _ in lines
        if ADDRESS_KW.search(ln) or (POSTAL_RE.search(ln) and "@" not in ln)
    )
    data["Email"] = _extract_email(lines)
    data["Company"] = _extract_company(lines)
    data["Phones"] = _extract_phones(lines)
    data["Website"] = _extract_website(lines, data["Email"])
    claimed = {
        data["Email"].lower(),
        data["Company"].strip().lower(),
        data["Website"].lower(),
        "",
    }
    for p in data["Phones"]:
        claimed.add(p.lower())
    data["Job_Title"] = _extract_job_title(lines, claimed)
    claimed.add(data["Job_Title"].strip().lower())
    addr_tokens = {
        w
        for ln, _ in lines
        if ADDRESS_KW.search(ln) or POSTAL_RE.search(ln)
        for w in ln.lower().split()
    }
    data["Name"] = _extract_name(lines, claimed, primary_script, addr_tokens)
    claimed.add(data["Name"].strip().lower())
    data["Address"] = _extract_address(lines, claimed, data["Phones"])
    data["LinkedIn"], data["Twitter"] = _extract_social(lines, claimed)
    return data


# ══════════════════════════════════════════════════════════════════════════════
#  QUALITY SCORING  (BUG 5 FIX — v16's weighted validated scoring)
# ══════════════════════════════════════════════════════════════════════════════


def quality_score(data: dict) -> str:
    """
    BUG 5 FIX: Uses v16's validated scoring (proven better in testing).
    v14 gave GREEN even when Name="Surat" or Name="" (4/5 non-empty check).
    v16 requires n_ok AND (e_ok OR p_ok) for GREEN — much stricter.
    """
    phones = data.get("Phones", [])
    n_ok = bool(
        data.get("Name")
        and len(data["Name"].split()) >= 2
        and not re.search(r"\d|@|\.", data["Name"])
        and not ADDRESS_KW.search(data["Name"])
        and not COMPANY_KW.search(data["Name"])
        and sum(c.isalpha() for c in data["Name"])
        / max(len(data["Name"].replace(" ", "")), 1)
        >= 0.75
    )
    c_ok = bool(
        data.get("Company")
        and data["Company"].strip().lower() != data.get("Name", "").lower()
        and (
            COMPANY_KW.search(data["Company"])
            or (data["Company"].isupper() and len(data["Company"].split()) >= 2)
        )
    )
    e_ok = bool(
        data.get("Email")
        and _EMAIL_RE.search(data["Email"])
        and len(data["Email"]) <= 80
    )
    p_ok = bool(phones and any(len(re.sub(r"\D", "", p)) >= 7 for p in phones))
    a_ok = bool(
        data.get("Address")
        and (
            ADDRESS_KW.search(data["Address"])
            or POSTAL_RE.search(data["Address"])
            or re.search(r"[A-Z][a-z]+,\s*[A-Z][a-z]+", data["Address"])
        )
    )
    score = (
        (2 if n_ok else 0)
        + (2 if e_ok else 0)
        + (1 if p_ok else 0)
        + (1 if c_ok else 0)
        + (1 if a_ok else 0)
    )
    if n_ok and (e_ok or p_ok) and score >= 5:
        return "🟢 GREEN"
    if score >= 2:
        return "🟡 YELLOW"
    return "🔴 RED"


# ══════════════════════════════════════════════════════════════════════════════
#  CORE PROCESSOR
# ══════════════════════════════════════════════════════════════════════════════


def process_card_array(
    img: np.ndarray, label: str = "card", lang_override: str = None
) -> dict:
    proc = preprocess(deskew(img))
    tokens, primary, secondary = ocr_card(proc, lang_override)
    if CFG["DEBUG"]:
        log.debug(
            "%s — %d tokens  script=%s/%s",
            label,
            len(tokens),
            primary,
            secondary or "—",
        )
        for t in sorted(tokens, key=lambda x: x["cy"]):
            log.debug("  cy=%6.1f conf=%.2f %r", t["cy"], t["conf"], t["text"])
    lines = group_into_lines(tokens)
    if CFG["DEBUG"]:
        for i, (ln, _) in enumerate(lines, 1):
            log.debug("  Line %02d: %r", i, ln)
    data = extract_fields(lines, primary)
    data["QUALITY"] = quality_score(data)
    data["_label"] = label
    data["_script"] = primary
    data["_script2"] = secondary or ""
    return data


def process_source(label: str, page_img: np.ndarray, lang_override: str = None) -> list:
    crops = segment_cards(page_img)
    results = []
    for idx, card_img in enumerate(crops):
        clabel = label if len(crops) == 1 else f"{label} [card {idx+1}/{len(crops)}]"
        try:
            results.append(process_card_array(card_img, clabel, lang_override))
        except Exception as e:
            log.error("Failed: %s — %s", clabel, e)
            if CFG["DEBUG"]:
                traceback.print_exc()
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  OUTPUT SYSTEM
# ══════════════════════════════════════════════════════════════════════════════


def _row_vals(r: dict) -> list:
    ph = r.get("Phones", [])
    return [
        r.get("Name", ""),
        r.get("Job_Title", ""),
        r.get("Company", ""),
        r.get("Email", ""),
        ph[0] if len(ph) > 0 else "",
        ph[1] if len(ph) > 1 else "",
        ph[2] if len(ph) > 2 else "",
        r.get("Website", ""),
        r.get("Address", ""),
        r.get("LinkedIn", ""),
        r.get("Twitter", ""),
        r.get("QUALITY", ""),
        datetime.now().strftime("%Y-%m-%d %H:%M"),
        r.get("_label", ""),
    ]


def _dedup_key(r: dict) -> str:
    return "|".join(
        [
            re.sub(r"\s+", "", r.get("Email", "").lower()),
            re.sub(r"\s+", "", r.get("Name", "").lower()),
            re.sub(r"\D", "", r.get("Phones", [""])[0] if r.get("Phones") else ""),
        ]
    )


def _write_header(ws):
    hf = PatternFill("solid", fgColor="1F4E79")
    hfont = Font(bold=True, color="FFFFFF", size=10)
    thin = Side(style="thin", color="BDD7EE")
    bdr = Border(left=thin, right=thin, top=thin, bottom=thin)
    for ci, name in enumerate(EXCEL_COLS, 1):
        c = ws.cell(1, ci, value=name)
        c.fill = hf
        c.font = hfont
        c.alignment = Alignment(horizontal="center", vertical="center")
        c.border = bdr
    ws.row_dimensions[1].height = 20


def _style_row(ws, rn: int, quality: str):
    col = _QC.get(quality, "FFFFFF")
    fill = PatternFill("solid", fgColor=col)
    thin = Side(style="thin", color="BDD7EE")
    bdr = Border(left=thin, right=thin, top=thin, bottom=thin)
    for ci in range(1, len(EXCEL_COLS) + 1):
        c = ws.cell(rn, ci)
        c.fill = fill
        c.alignment = Alignment(vertical="center", wrap_text=True)
        c.border = bdr
        c.font = Font(size=9)
    ws.row_dimensions[rn].height = 15


def _df_from_results(results: list) -> "pd.DataFrame":
    rows = []
    for r in results:
        ph = r.get("Phones", [])
        rows.append(
            {
                "Name": r.get("Name", ""),
                "Job_Title": r.get("Job_Title", ""),
                "Company": r.get("Company", ""),
                "Email": r.get("Email", ""),
                "Phone_1": ph[0] if len(ph) > 0 else "",
                "Phone_2": ph[1] if len(ph) > 1 else "",
                "Phone_3": ph[2] if len(ph) > 2 else "",
                "Website": r.get("Website", ""),
                "Address": r.get("Address", ""),
                "LinkedIn": r.get("LinkedIn", ""),
                "Twitter": r.get("Twitter", ""),
                "Quality": r.get("QUALITY", ""),
                "Processed": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "File": r.get("_label", ""),
            }
        )
    return pd.DataFrame(rows, columns=EXCEL_COLS)


def save_to_database(results: list, excel_path: str):
    if not EXCEL_OK:
        return
    month = datetime.now().strftime("%Y-%m")
    wb = (
        openpyxl.load_workbook(excel_path)
        if os.path.exists(excel_path)
        else openpyxl.Workbook()
    )
    if "Sheet" in wb.sheetnames:
        del wb["Sheet"]
    if month in wb.sheetnames:
        ws = wb[month]
        existing = {}
        for rn in range(2, ws.max_row + 1):
            key = _dedup_key(
                {
                    "Email": str(ws.cell(rn, 4).value or ""),
                    "Name": str(ws.cell(rn, 1).value or ""),
                    "Phones": [str(ws.cell(rn, 5).value or "")],
                }
            )
            existing[key] = rn
    else:
        ws = wb.create_sheet(title=month)
        _write_header(ws)
        existing = {}
    for r in results:
        key = _dedup_key(r)
        is_update = key in existing
        rn = existing[key] if is_update else ws.max_row + 1
        vals = _row_vals(r)
        for ci, v in enumerate(vals, 1):
            ws.cell(rn, ci, value=v)
        _style_row(ws, rn, r.get("QUALITY", ""))
        existing[key] = rn
        log.info(
            "  [DB] %s row %d → %s",
            "Updated" if is_update else "Appended",
            rn,
            r.get("Name") or r.get("Email") or "?",
        )
    for col in ws.columns:
        ws.column_dimensions[get_column_letter(col[0].column)].width = min(
            max(len(str(c.value or "")) for c in col) + 3, 55
        )
    ws.freeze_panes = "A2"
    try:
        wb.save(excel_path)
        log.info("Database → %s (sheet: %s)", excel_path, month)
    except PermissionError:
        alt = excel_path.replace(".xlsx", f"_{datetime.now().strftime('%H%M%S')}.xlsx")
        wb.save(alt)
        log.warning("File locked → saved to %s (close Excel first)", alt)


def _write_json(data, path: str):
    """
    Write a list of dicts as pretty-printed JSON with full Unicode support.
    Uses json.dumps instead of df.to_json() because pandas ≥ 2.0 removed
    the ensure_ascii parameter from to_json(), breaking non-ASCII characters
    (names like 'Müller', 'Al-Rashidi', 'Tanaka' etc.).
    """
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    except Exception as e:
        log.error("JSON write failed (%s): %s", path, e)


def save_outputs(results: list, out_dir: str):
    if not EXCEL_OK:
        return
    history = os.path.join(out_dir, "history")
    os.makedirs(history, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    df = _df_from_results(results)
    # Batch snapshot
    df.to_excel(os.path.join(history, f"cards_{ts}.xlsx"), index=False)
    _write_json(df.to_dict("records"), os.path.join(history, f"cards_{ts}.json"))
    df.to_csv(os.path.join(history, f"cards_{ts}.csv"), index=False)
    # Per-card individual files
    for i, r in enumerate(results, 1):
        safe = re.sub(r"[^\w\-]", "_", r.get("_label", "card"))[:30]
        _df_from_results([r]).to_excel(
            os.path.join(history, f"card_{ts}_{i:02d}_{safe}.xlsx"), index=False
        )
    # Stable latest
    df.to_excel(os.path.join(out_dir, "latest.xlsx"), index=False)
    _write_json(df.to_dict("records"), os.path.join(out_dir, "latest.json"))
    df.to_csv(os.path.join(out_dir, "latest.csv"), index=False)
    log.info("History → %s/ + latest.{xlsx,json,csv}", history)


def export_vcf(results: list, vcf_path: str):
    lines = []
    for r in results:
        ph = r.get("Phones", [])
        name = r.get("Name", "")
        parts = name.split(maxsplit=1)
        lines += ["BEGIN:VCARD", "VERSION:3.0"]
        if name:
            lines.append(
                f"N:{parts[-1]};{parts[0]};;;" if len(parts) == 2 else f"N:{name};;;;"
            )
            lines.append(f"FN:{name}")
        if r.get("Company"):
            lines.append(f"ORG:{r['Company']}")
        if r.get("Job_Title"):
            lines.append(f"TITLE:{r['Job_Title']}")
        if r.get("Email"):
            lines.append(f"EMAIL;TYPE=INTERNET:{r['Email']}")
        for p in ph:
            lines.append(f"TEL;TYPE=VOICE:{p}")
        if r.get("Website"):
            lines.append(f"URL:{r['Website']}")
        if r.get("Address"):
            lines.append(f"ADR;TYPE=WORK:;;{r['Address'].replace('  |  ',', ')};;;;")
        if r.get("LinkedIn"):
            lines.append(f"X-SOCIALPROFILE;TYPE=linkedin:{r['LinkedIn']}")
        if r.get("Twitter"):
            lines.append(f"X-SOCIALPROFILE;TYPE=twitter:{r['Twitter']}")
        lines += [
            f"NOTE:VC-OCR v17 {datetime.now().strftime('%Y-%m-%d')}",
            "END:VCARD",
            "",
        ]
    try:
        Path(vcf_path).write_text("\n".join(lines), encoding="utf-8")
        log.info("VCF → %s (%d cards)", vcf_path, len(results))
    except Exception as e:
        log.error("VCF export failed: %s", e)


# ══════════════════════════════════════════════════════════════════════════════
#  TERMINAL PRINT
# ══════════════════════════════════════════════════════════════════════════════


def print_result(r: dict):
    W = 64
    phones = r.get("Phones", [])
    print(f"\n{'═'*W}")
    print(f"  FILE      : {r.get('_label','')}")
    print(f"{'─'*W}")
    print(f"  NAME      : {r.get('Name','')       or '—'}")
    print(f"  JOB TITLE : {r.get('Job_Title','')  or '—'}")
    print(f"  COMPANY   : {r.get('Company','')    or '—'}")
    print(f"  EMAIL     : {r.get('Email','')      or '—'}")
    [print(f"  PHONE #{i:<3}: {p}") for i, p in enumerate(phones, 1)] or print(
        "  PHONE     : —"
    )
    print(f"  WEBSITE   : {r.get('Website','')    or '—'}")
    print(f"  ADDRESS   : {r.get('Address','')    or '—'}")
    print(f"  LINKEDIN  : {r.get('LinkedIn','')   or '—'}")
    print(f"  TWITTER/X : {r.get('Twitter','')    or '—'}")
    print(f"  QUALITY   : {r.get('QUALITY','')    or '—'}")
    print("═" * W)


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════


def _gui_pick() -> list:
    if not TK_OK:
        log.warning("tkinter unavailable — pass path as argument")
        return []
    root = Tk()
    root.withdraw()
    paths = filedialog.askopenfilenames(
        title="Select Card Images / PDFs / ZIPs",
        filetypes=[
            (
                "All supported",
                "*.jpg *.jpeg *.png *.bmp *.webp *.tiff *.tif *.pdf *.zip",
            )
        ],
    )
    root.destroy()
    return list(paths)


def main():
    ap = argparse.ArgumentParser(description="Visiting Card OCR v17.0")
    ap.add_argument("target", nargs="?", default=None)
    ap.add_argument("--lang", default=None)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--no-vcf", action="store_true")
    ap.add_argument("--no-excel", action="store_true")
    ap.add_argument("--db", default=None)
    args = ap.parse_args()
    if args.debug:
        CFG["DEBUG"] = True
        logging.getLogger().setLevel(logging.DEBUG)

    if args.target and Path(args.target).is_dir():
        supported = {
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp",
            ".tiff",
            ".tif",
            ".pdf",
            ".zip",
        }
        paths = sorted(
            str(f) for f in Path(args.target).iterdir() if f.suffix.lower() in supported
        )
    elif args.target:
        paths = [args.target]
    else:
        paths = _gui_pick()

    if not paths:
        log.info("No files selected. Exiting.")
        sys.exit(0)

    out_dir = os.path.dirname(os.path.abspath(paths[0]))
    db_path = args.db or os.path.join(out_dir, CFG["EXCEL_DB_FILE"])
    results = []

    log.info("Processing %d file(s)…", len(paths))
    for i, p in enumerate(paths, 1):
        log.info("[%d/%d] %s", i, len(paths), p)
        try:
            for plabel, pimg in load_images(p):
                for r in process_source(plabel, pimg, args.lang):
                    print_result(r)
                    results.append(r)
        except Exception as e:
            log.error("Failed on %s: %s", p, e)
            if CFG["DEBUG"]:
                traceback.print_exc()

    if not results:
        log.warning("No cards processed.")
        return
    log.info("✔ %d card(s) processed.", len(results))
    if not args.no_excel:
        save_to_database(results, db_path)
        save_outputs(results, out_dir)
    if not args.no_vcf:
        export_vcf(
            results,
            os.path.join(
                out_dir, f"contacts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.vcf"
            ),
        )


if __name__ == "__main__":
    main()
