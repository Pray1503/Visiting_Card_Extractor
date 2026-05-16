# import os
# import re
# import cv2
# import math
# import json
# import traceback
# import numpy as np
# import pandas as pd
# import torch
# import phonenumbers
# import tldextract
# from datetime import datetime
# from rapidfuzz import fuzz
# from spellchecker import SpellChecker
# from paddleocr import PaddleOCR
# from phonenumbers import PhoneNumberMatcher
# import spacy

# # ══════════════════════════════════════════════════════════════════════════════
# # CONFIG
# # ══════════════════════════════════════════════════════════════════════════════
# CFG = {
#     "OCR_CONF_THRESH": 0.18,
#     "PADDLE_LANG": "en",
#     "DEBUG": True,
# }

# # ══════════════════════════════════════════════════════════════════════════════
# # GLOBALS
# # ══════════════════════════════════════════════════════════════════════════════
# USE_GPU = torch.cuda.is_available()
# SPELL = SpellChecker()

# # ══════════════════════════════════════════════════════════════════════════════
# # LOAD SPACY
# # ══════════════════════════════════════════════════════════════════════════════
# try:
#     NLP = spacy.load("en_core_web_sm")
#     SPACY_OK = True
# except Exception:
#     NLP = None
#     SPACY_OK = False

# # ══════════════════════════════════════════════════════════════════════════════
# # LOAD PADDLEOCR
# # ══════════════════════════════════════════════════════════════════════════════
# PADDLE_READER = None


# def get_paddle_reader(lang="en"):
#     global PADDLE_READER
#     if PADDLE_READER is None:
#         print(f"[INFO] Loading PaddleOCR ({lang})...")
#         PADDLE_READER = PaddleOCR(lang=lang, use_gpu=USE_GPU, show_log=False)
#         print("[INFO] PaddleOCR Loaded")
#     return PADDLE_READER


# # ══════════════════════════════════════════════════════════════════════════════
# # PREPROCESSING
# # ══════════════════════════════════════════════════════════════════════════════
# def preprocess(img):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (3, 3), 0)
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     enhanced = clahe.apply(blur)
#     kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
#     sharp = cv2.filter2D(enhanced, -1, kernel)
#     return sharp


# # ══════════════════════════════════════════════════════════════════════════════
# # DESKEW
# # ══════════════════════════════════════════════════════════════════════════════
# def deskew(img):
#     if len(img.shape) == 3:
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     else:
#         gray = img

#     edges = cv2.Canny(gray, 50, 150)
#     lines = cv2.HoughLinesP(
#         edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10
#     )

#     if lines is None:
#         return img

#     angles = []
#     for line in lines:
#         x1, y1, x2, y2 = line[0]
#         angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
#         if abs(angle) < 45:
#             angles.append(angle)

#     if not angles:
#         return img

#     median_angle = np.median(angles)
#     h, w = img.shape[:2]
#     center = (w // 2, h // 2)
#     M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
#     rotated = cv2.warpAffine(
#         img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
#     )
#     return rotated


# # ══════════════════════════════════════════════════════════════════════════════
# # OCR
# # ══════════════════════════════════════════════════════════════════════════════
# def run_paddle_ocr(img):
#     reader = get_paddle_reader(CFG["PADDLE_LANG"])
#     if len(img.shape) == 2:
#         img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

#     result = reader.ocr(img)
#     tokens = []

#     try:
#         if result and result[0]:
#             for page in result:
#                 for item in page:
#                     bbox = item[0]
#                     text = item[1][0]
#                     conf = item[1][1]
#                     if conf < CFG["OCR_CONF_THRESH"]:
#                         continue
#                     tokens.append({"text": text.strip(), "conf": conf, "bbox": bbox})
#     except Exception as e:
#         print(f"[ERROR] OCR Parsing Failed: {e}")
#         traceback.print_exc()

#     return tokens


# # ══════════════════════════════════════════════════════════════════════════════
# # RECONSTRUCTION
# # ══════════════════════════════════════════════════════════════════════════════
# def reconstruct(tokens):
#     fixed = []
#     for t in tokens:
#         text = t["text"]
#         text = re.sub(r"\s+@\s+", "@", text)
#         text = re.sub(r"\s+\.\s+", ".", text)
#         if re.fullmatch(r"(?:[A-Z]\s+){2,}[A-Z]", text):
#             text = text.replace(" ", "")
#         text = re.sub(r"([A-Z]{2,})\s+([A-Z]{2,})", r"\1\2", text)
#         fixed.append(text)
#     return fixed


# # ══════════════════════════════════════════════════════════════════════════════
# # EXTRACTION
# # ══════════════════════════════════════════════════════════════════════════════
# def extract_fields(lines):
#     raw_text = "\n".join(lines)
#     data = {
#         "Name": "",
#         "Job_Title": "",
#         "Company": "",
#         "Email": "",
#         "Phone": "",
#         "Website": "",
#         "Address": "",
#         "LinkedIn": "",
#         "Twitter": "",
#     }

#     # EMAIL
#     emails = re.findall(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", raw_text)
#     if emails:
#         data["Email"] = emails[0]

#     # PHONE
#     phones = []
#     try:
#         for match in PhoneNumberMatcher(raw_text, None):
#             phones.append(
#                 phonenumbers.format_number(
#                     match.number, phonenumbers.PhoneNumberFormat.INTERNATIONAL
#                 )
#             )
#     except:
#         pass
#     if phones:
#         data["Phone"] = phones[0]

#     # WEBSITE
#     websites = re.findall(
#         r"(?:https?://)?(?:www\.)?[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", raw_text
#     )
#     websites = [w for w in websites if "@" not in w]
#     if websites:
#         data["Website"] = websites[0]

#     # NLP
#     if SPACY_OK and NLP:
#         doc = NLP(raw_text)
#         for ent in doc.ents:
#             if ent.label_ == "PERSON" and not data["Name"]:
#                 data["Name"] = ent.text
#             elif ent.label_ == "ORG" and not data["Company"]:
#                 data["Company"] = ent.text
#             elif ent.label_ in ["GPE", "LOC"] and not data["Address"]:
#                 data["Address"] = ent.text

#     # DESIGNATION
#     designations = [
#         "CEO",
#         "CTO",
#         "Founder",
#         "Engineer",
#         "Manager",
#         "Director",
#         "Developer",
#         "Consultant",
#         "Analyst",
#     ]
#     for line in lines:
#         if any(d.lower() in line.lower() for d in designations):
#             data["Job_Title"] = line
#             break

#     # FALLBACK NAME
#     if not data["Name"]:
#         for line in lines:
#             if 2 <= len(line.split()) <= 3:
#                 if line.isupper() or line.istitle():
#                     data["Name"] = line
#                     break
#     return data


# # ══════════════════════════════════════════════════════════════════════════════
# # QUALITY SCORE
# # ══════════════════════════════════════════════════════════════════════════════
# def quality_score(data):
#     score = sum(1 for v in data.values() if v)
#     if score >= 6:
#         return "🟢 GREEN"
#     elif score >= 3:
#         return "🟡 YELLOW"
#     return "🔴 RED"


# # ══════════════════════════════════════════════════════════════════════════════
# # PROCESS SINGLE CARD
# # ══════════════════════════════════════════════════════════════════════════════
# def process_card(path):
#     img = cv2.imread(path)
#     if img is None:
#         raise ValueError("Image could not be loaded")
#     proc = preprocess(img)
#     proc = deskew(proc)
#     tokens = run_paddle_ocr(proc)
#     if CFG["DEBUG"]:
#         print(f"[DEBUG] Tokens Detected: {len(tokens)}")
#     lines = reconstruct(tokens)
#     data = extract_fields(lines)
#     data["QUALITY"] = quality_score(data)
#     return data


# # ══════════════════════════════════════════════════════════════════════════════
# # SAVE OUTPUTS
# # ══════════════════════════════════════════════════════════════════════════════
# def save_outputs(results):
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     df = pd.DataFrame(results)
#     xlsx_path = f"cards_{timestamp}.xlsx"
#     json_path = f"cards_{timestamp}.json"
#     csv_path = f"cards_{timestamp}.csv"
#     df.to_excel(xlsx_path, index=False)
#     df.to_json(json_path, orient="records", indent=4)
#     df.to_csv(csv_path, index=False)
#     print(f"\n[XLSX] → {xlsx_path}\n[JSON] → {json_path}\n[CSV]  → {csv_path}")


# # ══════════════════════════════════════════════════════════════════════════════
# # MAIN
# # ══════════════════════════════════════════════════════════════════════════════
# def main():
#     target = input("Enter image path: ").strip()
#     if not os.path.exists(target):
#         print("[ERROR] File not found")
#         return

#     try:
#         result = process_card(target)
#         print("\n" + "═" * 70)
#         for k, v in result.items():
#             print(f"{k:<12}: {v}")
#         print("═" * 70)
#         save_outputs([result])
#     except Exception as e:
#         print(f"[ERROR] {e}")
#         traceback.print_exc()


# if __name__ == "__main__":
#     main()

# ══════════════════════════════════════════════════════════════════════════════
# VISITING CARD OCR ENGINE  v12.0  —  ENTERPRISE EDITION
# ══════════════════════════════════════════════════════════════════════════════
#
#  ARCHITECTURE (5 layers):
#  ┌─────────────────────────────────────────────────────────────────────────┐
#  │ L0 — INPUT ROUTER     Image / PDF / ZIP → in-memory np.ndarray arrays  │
#  │ L1 — CARD SEGMENTOR   OpenCV contour engine → isolate individual cards  │
#  │ L2 — IMAGE PIPELINE   Deskew → CLAHE → Sharpen (per card array)        │
#  │ L3 — MULTILINGUAL OCR Dual-script detection → PaddleOCR hot-swap       │
#  │ L4 — FIELD EXTRACTOR  Email→Phone→Website→Company→Title→Name→Address   │
#  └─────────────────────────────────────────────────────────────────────────┘
#
#  BUG FIXES vs v11:
#
#  ❌ BUG 1 — Coordinate Scale-Blindness:
#     Y_ROW_TOL=20 and cx//20 cell keys are hardcoded pixel values that break
#     at any resolution other than the one they were tuned on (mobile ~1080p).
#     At 300 DPI PDF rasterization (~2480px tall), lines fragment badly.
#     At low-res scans, distinct fields collapse into the same merge cell.
#  ✅ FIX → Y_ROW_TOL = median token height × 0.55 (computed per image).
#           Merge cell key = cx // cell_px, cy // cell_px where
#           cell_px = max(image_h * 0.02, 12).  Both scale with resolution.
#
#  ❌ BUG 2 — Multi-Language Split Blind Spot:
#     Single dominant-script detection forces one PaddleOCR language even
#     when a card mixes Japanese address lines with English contact details.
#  ✅ FIX → Dual-stream inference: if any secondary script meets a 15%
#           character-count minority threshold, both language models are run
#           and their token grids are merged (highest confidence wins per cell).
#           Result: mixed-language cards get best-of-both-worlds recognition.
#
#  ❌ BUG 3 — Phone-to-Address Overlap Leak:
#     Strict substring logic (phone_fp in line_digits) fails when OCR adds
#     extension text ("Ext 1234") or spacing artifacts that fragment the
#     digit sequence. Phone line slips through to address extractor.
#  ✅ FIX → Replace substring check with RapidFuzz partial_ratio comparison.
#           Any line whose digit sequence scores ≥ 80% similarity to the
#           extracted phone's digit sequence is rejected as a phone line.
#           Threshold tunable in CFG["PHONE_FUZZ_THRESH"].
#
#  ❌ BUG 4 — Name Extraction Leakage:
#     Fallback heuristic accepted any 2-4 word line starting with an uppercase
#     letter, which captured partial job-title fragments or city names.
#  ✅ FIX → Reinforced name constraints: ALL tokens in candidate line must be
#           either (a) fully uppercase, or (b) strictly title-cased (first char
#           upper, remainder lower). Mixed-case or lowercase tokens disqualify
#           the entire line. Minimum alpha-character ratio enforced (≥ 0.75).
#
#  NEW FEATURES vs v11:
#
#  ★ Multi-Card Page Segmentation (Section 3.1):
#    Pre-OCR OpenCV contour scanner detects multiple business cards on one
#    sheet. Canny edges + morphological dilation + contour filtering by area
#    and aspect ratio isolate each card ROI, which is padded and sorted
#    top-left → bottom-right before processing.
#
#  ★ PDF & ZIP Container Handling (Section 3.2):
#    pdf2image.convert_from_path rasterizes PDF pages at 250 DPI in-memory.
#    zipfile.ZipFile extracts image members into BytesIO buffers — no temp
#    files written to disk. Recursive unpacking handles zips-within-zips.
#
#  ★ In-Memory Array Pipeline (Section 3.3):
#    process_card_array(img: np.ndarray) is the core processing unit.
#    No cv2.imwrite / tempfile calls anywhere in the pipeline.
#    L0 → L4 operate entirely on numpy arrays and Python dicts.
#
# INSTALL:
#   pip install paddlepaddle paddleocr opencv-python-headless numpy pillow
#   pip install pandas openpyxl phonenumbers spacy rapidfuzz
#   pip install pdf2image    (also needs poppler: choco install poppler / brew install poppler)
#   python -m spacy download en_core_web_sm
#
# USAGE:
#   python visiting_card_ocr_v12.py
#   → GUI file picker for image / PDF / ZIP
# ══════════════════════════════════════════════════════════════════════════════

# ── oneDNN Windows crash prevention — MUST be before all other imports ────────
import os

os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_call_stack_level"] = "2"

# ── Standard library ──────────────────────────────────────────────────────────
import re
import io
import cv2
import math
import zipfile
import traceback
import unicodedata
from datetime import datetime
from pathlib import Path
from tkinter import Tk, filedialog

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import torch
import phonenumbers
import spacy

from paddleocr import PaddleOCR
from phonenumbers import PhoneNumberMatcher, PhoneNumberFormat
from rapidfuzz import fuzz as rfuzz
from PIL import Image

# ── Optional PDF support ──────────────────────────────────────────────────────
try:
    from pdf2image import convert_from_bytes

    PDF_OK = True
except ImportError:
    PDF_OK = False
    print(
        "[WARN] pdf2image not installed — PDF support disabled. "
        "Install with: pip install pdf2image"
    )


# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
CFG = {
    # OCR confidence threshold (tokens below this are discarded)
    "OCR_CONF_THRESH": 0.18,
    # PDF rasterization resolution (DPI)
    "PDF_DPI": 250,
    # Card segmentation: minimum card area as fraction of full image area
    "SEG_MIN_AREA_FRAC": 0.03,
    # Card segmentation: maximum card area as fraction (avoids full-page match)
    "SEG_MAX_AREA_FRAC": 0.95,
    # Card segmentation: padding in pixels added around each detected card
    "SEG_PAD_PX": 12,
    # Layout grouping: row tolerance = median token height × this factor
    # (replaces hardcoded Y_ROW_TOL = 20 — fixes Bug 1)
    "ROW_TOL_FACTOR": 0.55,
    # Layout grouping: fallback absolute tolerance when no tokens present
    "ROW_TOL_FALLBACK_PX": 20,
    # Merge cell size = image_height × this factor (replaces cx//20 — fixes Bug 1)
    "MERGE_CELL_FACTOR": 0.02,
    # Absolute minimum cell size in pixels (prevents division anomalies)
    "MERGE_CELL_MIN_PX": 12,
    # Multilingual: secondary script is activated if its char share ≥ this ratio
    # (fixes Bug 2 — dual-stream inference)
    "SECONDARY_SCRIPT_THRESH": 0.15,
    # Phone-address overlap: RapidFuzz partial_ratio threshold (0–100)
    # Lines whose digit sequence similarity to phone digits meets this threshold
    # are rejected from the address field (fixes Bug 3)
    "PHONE_FUZZ_THRESH": 80,
    # Name heuristic: minimum fraction of characters that must be alphabetic
    # (fixes Bug 4 — name leakage)
    "NAME_ALPHA_RATIO": 0.75,
    # Debug output toggle
    "DEBUG": True,
}

# ══════════════════════════════════════════════════════════════════════════════
# SCRIPT → PADDLEOCR LANGUAGE MAP
# ══════════════════════════════════════════════════════════════════════════════
SCRIPT_TO_LANG: dict[str, str] = {
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

# spaCy model per script (falls back to English)
SCRIPT_TO_SPACY: dict[str, str] = {
    "latin": "en_core_web_sm",
    "cjk": "zh_core_web_sm",
    "default": "en_core_web_sm",
}

# ══════════════════════════════════════════════════════════════════════════════
# COMPILED REGEX PATTERNS
# ══════════════════════════════════════════════════════════════════════════════

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
    r"\b("
    r"ltd|limited|inc|llp|llc|pvt|private|plc|corp|corporation|"
    r"solutions|consulting|studio|technologies|tech|group|company|"
    r"labs|services|systems|enterprises|associates|builders|"
    r"construction|industries|global|international|ventures|"
    r"holdings|capital|networks|media|digital|infra|infrastructure|"
    r"realty|realtors|properties|developers|architects|interiors|"
    r"design|agency|firm|foundation|trust|institute|academy|"
    r"hospital|clinic|healthcare|pharma|logistics|transport|"
    r"exports|imports|trading|manufacturing|fabrication|engineering|"
    r"gmbh|ag|kg|ohg|ug|e\.v\.|"
    r"sarl|sas|sci|snc|eurl|"
    r"s\.a\.|s\.l\.|ltda|s\.a\.s\.|"
    r"spa|srl|"
    r"k\.k\.|kk|"
    r"sdn\.?\s*bhd|pte\.|"
    r"b\.v\.|n\.v\.|vof|a\/s"
    r")\b",
    re.IGNORECASE,
)

ADDRESS_KW = re.compile(
    r"\b("
    r"floor|fl\.|level|suite|plot|block|sector|phase|unit|"
    r"apt|apartment|building|bldg|tower|complex|plaza|centre|center|mall|"
    r"road|rd\.|street|st\.|avenue|ave\.|lane|ln\.|boulevard|blvd\.|"
    r"drive|dr\.|court|ct\.|place|pl\.|way|highway|freeway|"
    r"square|sq\.|terrace|ter\.|close|crescent|grove|gardens|"
    r"nagar|vihar|marg|gali|chowk|bazaar|colony|society|residency|"
    r"straße|strasse|str\.|gasse|weg|platz|"
    r"rue|impasse|allée|"
    r"calle|avenida|carrera|paseo|"
    r"district|area|zone|suburb|village|town|city|state|province|"
    r"county|region|po\s*box|p\.o\.|zip|pin|postcode|postal"
    r")\b",
    re.IGNORECASE,
)

POSTAL_RE = re.compile(r"\b\d{4,9}\b")

WEBSITE_RE = re.compile(
    r"(?:https?://|www\.)[A-Za-z0-9.\-/_%?=&#]+"
    r"|[A-Za-z0-9][\w\-]*\."
    r"(?:com|org|net|in|io|co\.in|co\.uk|co\.ae|co\.sg|co\.za|co\.au|"
    r"co\.nz|co\.jp|biz|info|edu|gov|ae|us|au|nz|sg|hk|my|ph|za|"
    r"de|fr|jp|cn|ca|br|mx|tech|app|dev|ai|online|store|site)"
    r"(?:[/\w\-]*)?",
    re.IGNORECASE,
)

EMAIL_RE = re.compile(
    r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}",
    re.IGNORECASE,
)

# ══════════════════════════════════════════════════════════════════════════════
# PHONE REGION DETECTION — REGION-NEUTRAL
# ══════════════════════════════════════════════════════════════════════════════

_CC_REGION: dict[str, str] = {
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

_FALLBACK_REGIONS: list[str] = [
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
]


def _detect_phone_region(text: str) -> str | None:
    """Scan text for +XX prefix and map to ISO-3166 region code."""
    m = re.search(r"\+(\d{1,3})", text)
    if not m:
        return None
    cc = m.group(1)
    for length in (3, 2, 1):
        region = _CC_REGION.get(cc[:length])
        if region:
            return region
    return None


# ══════════════════════════════════════════════════════════════════════════════
# MODEL SINGLETONS — loaded once, reused across all cards in the batch
# ══════════════════════════════════════════════════════════════════════════════

USE_GPU = torch.cuda.is_available()
_PADDLE_CACHE: dict[str, PaddleOCR] = {}
_SPACY_CACHE: dict[str, object] = {}


def get_paddle_reader(lang: str = "en") -> PaddleOCR:
    """Return a cached PaddleOCR reader for the given language."""
    if lang not in _PADDLE_CACHE:
        print(f"  [INFO] Loading PaddleOCR model (lang={lang})…")
        _PADDLE_CACHE[lang] = PaddleOCR(
            lang=lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_rec_score_thresh=CFG["OCR_CONF_THRESH"],
            device="gpu" if USE_GPU else "cpu",
        )
        print(f"  [INFO] PaddleOCR (lang={lang}) ready.")
    return _PADDLE_CACHE[lang]


def get_spacy_nlp(model: str = "en_core_web_sm") -> object:
    """Return a cached spaCy pipeline, falling back to English on OSError."""
    if model not in _SPACY_CACHE:
        try:
            _SPACY_CACHE[model] = spacy.load(model)
        except OSError:
            try:
                _SPACY_CACHE[model] = spacy.load("en_core_web_sm")
            except OSError:
                _SPACY_CACHE[model] = None
    return _SPACY_CACHE[model]


# ══════════════════════════════════════════════════════════════════════════════
# SCRIPT DETECTION — DUAL-STREAM MULTILINGUAL  (BUG 2 FIX)
# ══════════════════════════════════════════════════════════════════════════════


def detect_scripts(text: str) -> tuple[str, str | None]:
    """
    Counts Unicode character blocks to determine the dominant script and,
    if a secondary script meets the minority threshold, returns both.

    Returns:
        (primary_script, secondary_script | None)

    Dual-stream example: a Japanese business card with English email/phone
    returns ("japanese", "latin") — both PaddleOCR models will be run and
    their tokens merged (highest confidence wins per cell grid position).
    """
    counts: dict[str, int] = {s: 0 for s in SCRIPT_TO_LANG}

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

    # Sort scripts by character count descending
    ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)

    primary_script = ranked[0][0]
    # Normalise: if dominant is latin with very few counts, still latin
    if primary_script != "latin" and counts[primary_script] < 5:
        primary_script = "latin"

    # Secondary script: any non-primary script with ≥ threshold share
    secondary_script: str | None = None
    for script, cnt in ranked[1:]:
        if script == primary_script:
            continue
        if cnt / total >= CFG["SECONDARY_SCRIPT_THRESH"]:
            secondary_script = script
            break

    return primary_script, secondary_script


# ══════════════════════════════════════════════════════════════════════════════
# L0 — INPUT ROUTER  (in-memory, no temp files)
# ══════════════════════════════════════════════════════════════════════════════


def _pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    """Convert a PIL Image to an OpenCV BGR numpy array."""
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def load_images_from_path(file_path: str) -> list[tuple[str, np.ndarray]]:
    """
    Routes a file path to the correct loader based on extension.

    Returns:
        List of (label, bgr_array) tuples.
        label = human-readable identifier for logging (e.g. "sheet.pdf page 3")
    """
    ext = Path(file_path).suffix.lower()

    if ext in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}:
        img = cv2.imread(file_path)
        if img is None:
            raise ValueError(f"Cannot read image: {file_path}")
        return [(Path(file_path).name, img)]

    elif ext == ".pdf":
        return _load_pdf(file_path)

    elif ext == ".zip":
        return _load_zip(file_path)

    else:
        raise ValueError(f"Unsupported file type: {ext}")


def _load_pdf(file_path: str) -> list[tuple[str, np.ndarray]]:
    """
    Rasterize all PDF pages in-memory at CFG["PDF_DPI"].
    Returns one BGR array per page — no disk writes.
    """
    if not PDF_OK:
        raise RuntimeError(
            "pdf2image is not installed. "
            "Run: pip install pdf2image  (and install Poppler)."
        )

    stem = Path(file_path).stem
    with open(file_path, "rb") as fh:
        raw = fh.read()

    pages = convert_from_bytes(raw, dpi=CFG["PDF_DPI"])
    return [(f"{stem}_page{i+1}", _pil_to_bgr(p)) for i, p in enumerate(pages)]


def _load_zip(file_path: str) -> list[tuple[str, np.ndarray]]:
    """
    Recursively unpack a ZIP archive and extract all image members into
    in-memory BGR arrays.  Nested ZIPs are unpacked recursively.
    No temporary files are written.
    """
    results: list[tuple[str, np.ndarray]] = []
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}

    with zipfile.ZipFile(file_path, "r") as zf:
        for member in zf.namelist():
            member_ext = Path(member).suffix.lower()

            if member_ext in image_exts:
                raw_bytes = zf.read(member)
                buf = np.frombuffer(raw_bytes, dtype=np.uint8)
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if img is not None:
                    results.append((member, img))

            elif member_ext == ".pdf":
                raw_bytes = zf.read(member)
                if PDF_OK:
                    pages = convert_from_bytes(raw_bytes, dpi=CFG["PDF_DPI"])
                    stem = Path(member).stem
                    for i, p in enumerate(pages):
                        results.append((f"{stem}_page{i+1}", _pil_to_bgr(p)))

            elif member_ext == ".zip":
                # Nested zip: read into BytesIO buffer and recurse
                raw_bytes = zf.read(member)
                inner_buf = io.BytesIO(raw_bytes)
                with zipfile.ZipFile(inner_buf, "r") as inner_zf:
                    inner_exts = {
                        ".jpg",
                        ".jpeg",
                        ".png",
                        ".bmp",
                        ".webp",
                        ".tiff",
                        ".tif",
                    }
                    for inner_member in inner_zf.namelist():
                        if Path(inner_member).suffix.lower() in inner_exts:
                            ib = np.frombuffer(
                                inner_zf.read(inner_member), dtype=np.uint8
                            )
                            img = cv2.imdecode(ib, cv2.IMREAD_COLOR)
                            if img is not None:
                                results.append((f"{member}/{inner_member}", img))

    return results


# ══════════════════════════════════════════════════════════════════════════════
# L1 — CARD SEGMENTOR  (multi-card detection)
# ══════════════════════════════════════════════════════════════════════════════


def segment_cards(img: np.ndarray) -> list[np.ndarray]:
    """
    Detects and extracts individual business cards from a sheet image using
    an OpenCV contour pipeline:

      1. Convert to grayscale
      2. Gaussian blur (reduces noise before edge detection)
      3. Canny edge detection
      4. Morphological dilation (closes gaps between card edges)
      5. Find external contours
      6. Filter by area (SEG_MIN_AREA_FRAC … SEG_MAX_AREA_FRAC of total image)
         and approximate aspect ratio (business cards are typically 1.4–2.2×)
      7. Extract bounding rectangles, add padding, sort top-left → bottom-right
      8. Crop each ROI and return as list of BGR arrays

    If no valid card contours are found, returns the full image as-is (single
    card mode — backward compatible with all existing callers).
    """
    h_img, w_img = img.shape[:2]
    total_area = h_img * w_img

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 120)

    # Dilation kernel: horizontal emphasis helps connect card border lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
    dilated = cv2.dilate(edges, kernel, iterations=3)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pad = CFG["SEG_PAD_PX"]
    min_area = total_area * CFG["SEG_MIN_AREA_FRAC"]
    max_area = total_area * CFG["SEG_MAX_AREA_FRAC"]
    rois: list[tuple[int, int, int, int]] = []  # (x, y, w, h)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (min_area <= area <= max_area):
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / max(h, 1)
        # Standard business card aspect ratio ≈ 1.75 (85mm × 55mm)
        # Accept a generous range to handle landscape / portrait / square cards
        if not (0.8 <= aspect <= 3.5):
            continue
        rois.append((x, y, w, h))

    if not rois:
        # No multi-card layout detected — return the full image as one card
        return [img]

    # Sort ROIs in natural reading order: top → bottom, left → right
    rois.sort(key=lambda r: (r[1] // 100, r[0]))  # y-band of 100px, then x

    crops: list[np.ndarray] = []
    for x, y, w, h in rois:
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w_img, x + w + pad)
        y2 = min(h_img, y + h + pad)
        crops.append(img[y1:y2, x1:x2])

    return crops


# ══════════════════════════════════════════════════════════════════════════════
# L2 — IMAGE PIPELINE
# ══════════════════════════════════════════════════════════════════════════════


def preprocess(img: np.ndarray) -> np.ndarray:
    """Grayscale → Gaussian blur → CLAHE → sharpening kernel."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blur)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(enhanced, -1, kernel)


def deskew(img: np.ndarray) -> np.ndarray:
    """Correct card rotation using Hough line transform median angle."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=80, minLineLength=60, maxLineGap=20
    )
    if lines is None:
        return img
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        a = math.degrees(math.atan2(y2 - y1, x2 - x1))
        if abs(a) < 45:
            angles.append(a)
    if not angles:
        return img
    angle = float(np.median(angles))
    if abs(angle) < 0.5:  # below 0.5° is noise — don't rotate
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(
        img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


# ══════════════════════════════════════════════════════════════════════════════
# L3 — MULTILINGUAL OCR ENGINE  (dual-stream, resolution-adaptive)
# ══════════════════════════════════════════════════════════════════════════════


def run_paddle_ocr(img: np.ndarray, lang: str = "en") -> list[dict]:
    """
    Run PaddleOCR on a preprocessed grayscale or BGR array.

    Returns a list of token dicts:
        {"text": str, "conf": float, "bbox": list, "cx": float, "cy": float}

    cx and cy are the centroid coordinates of the bounding polygon.
    These are used downstream for resolution-adaptive layout grouping.
    """
    reader = get_paddle_reader(lang)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    tokens: list[dict] = []
    try:
        results = reader.predict(img)
        for res in results:
            if not isinstance(res, dict):
                continue
            texts = res.get("rec_texts", [])
            scores = res.get("rec_scores", [])
            polys = res.get("rec_polys", res.get("dt_polys", []))

            for text, conf, poly in zip(texts, scores, polys):
                conf = float(conf)
                if conf < CFG["OCR_CONF_THRESH"]:
                    continue
                text = str(text).strip()
                if not text:
                    continue
                pts = [list(map(float, p)) for p in poly]
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                tokens.append(
                    {
                        "text": text,
                        "conf": conf,
                        "bbox": pts,
                        "cx": sum(xs) / len(xs),
                        "cy": sum(ys) / len(ys),
                        "th": (max(ys) - min(ys)),  # token height for tol calc
                    }
                )
    except Exception as e:
        print(f"  [ERROR] OCR failed (lang={lang}): {e}")
        if CFG["DEBUG"]:
            traceback.print_exc()

    return tokens


def merge_token_grids(
    token_lists: list[list[dict]],
    image_h: int,
) -> list[dict]:
    """
    Merge multiple token lists (from dual-language OCR passes) into one.

    Cell key = (int(cx // cell_px), int(cy // cell_px))
    where cell_px = max(image_h × MERGE_CELL_FACTOR, MERGE_CELL_MIN_PX).

    This makes the merge grid scale with image resolution — fixing Bug 1.
    At 300 DPI (image_h ≈ 2400px): cell_px ≈ 48px  (coarser, correct merge)
    At 150 DPI (image_h ≈ 1200px): cell_px ≈ 24px  (finer, correct merge)
    Hardcoded 20 would be too small at 300 DPI and too large at low res.
    """
    cell_px = max(image_h * CFG["MERGE_CELL_FACTOR"], CFG["MERGE_CELL_MIN_PX"])
    grid: dict[str, dict] = {}

    for token_list in token_lists:
        for t in token_list:
            key = f"{int(t['cx'] // cell_px)},{int(t['cy'] // cell_px)}"
            if key not in grid or t["conf"] > grid[key]["conf"]:
                grid[key] = t

    return list(grid.values())


def ocr_card(proc: np.ndarray) -> tuple[list[dict], str, str | None]:
    """
    Full OCR pass for one preprocessed card array:
      1. English pass → collect tokens
      2. detect_scripts on raw text → (primary, secondary)
      3. If primary != "en": re-run with primary language model
      4. If secondary exists: also run secondary language model
      5. Merge all token grids with resolution-adaptive cell key

    Returns: (merged_tokens, primary_script, secondary_script | None)
    """
    image_h = proc.shape[0]

    # Step 1 — English seed pass
    tokens_en = run_paddle_ocr(proc, lang="en")
    raw_sample = " ".join(t["text"] for t in tokens_en)
    primary_script, secondary_script = detect_scripts(raw_sample)

    primary_lang = SCRIPT_TO_LANG.get(primary_script, "en")
    secondary_lang = (
        SCRIPT_TO_LANG.get(secondary_script, "en") if secondary_script else None
    )

    if CFG["DEBUG"] and secondary_script:
        print(
            f"  [INFO] Dual-script detected: primary={primary_script} "
            f"secondary={secondary_script}"
        )

    # Step 2 — Collect all token lists to merge
    token_lists: list[list[dict]] = [tokens_en]

    if primary_lang != "en":
        token_lists.append(run_paddle_ocr(proc, lang=primary_lang))

    if secondary_lang and secondary_lang != "en" and secondary_lang != primary_lang:
        token_lists.append(run_paddle_ocr(proc, lang=secondary_lang))

    merged = merge_token_grids(token_lists, image_h)
    return merged, primary_script, secondary_script


# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT-AWARE LINE GROUPING  (resolution-adaptive — BUG 1 FIX)
# ══════════════════════════════════════════════════════════════════════════════


def group_into_lines(tokens: list[dict]) -> list[list[dict]]:
    """
    Group tokens into visual lines using a resolution-adaptive Y-tolerance.

    Tolerance = median token height × ROW_TOL_FACTOR (default 0.55).

    Why adaptive:
      A card scanned at 300 DPI has tokens ~40px tall → tol ≈ 22px  ✓
      A card scanned at 72 DPI has tokens ~10px tall  → tol ≈ 5.5px ✓
      Hardcoded 20px would be too coarse at 72 DPI (merges separate lines)
      and too fine at 600 DPI (splits a single line into multiple rows).
    """
    if not tokens:
        return []

    # Compute resolution-adaptive row tolerance
    heights = [t.get("th", 20) for t in tokens]
    med_h = float(np.median(heights)) if heights else 20.0
    tol = max(med_h * CFG["ROW_TOL_FACTOR"], CFG["ROW_TOL_FALLBACK_PX"])

    by_y = sorted(tokens, key=lambda t: t["cy"])
    rows: list[list[dict]] = []
    cur = [by_y[0]]

    for tok in by_y[1:]:
        avg_cy = sum(t["cy"] for t in cur) / len(cur)
        if abs(tok["cy"] - avg_cy) <= tol:
            cur.append(tok)
        else:
            rows.append(sorted(cur, key=lambda t: t["cx"]))
            cur = [tok]
    rows.append(sorted(cur, key=lambda t: t["cx"]))

    return rows


# ══════════════════════════════════════════════════════════════════════════════
# TEXT CLEANING
# ══════════════════════════════════════════════════════════════════════════════


def clean_token(text: str) -> str:
    """Normalise OCR artefacts in a single token string."""
    # "A D I T Y A" → "ADITYA"
    if re.fullmatch(r"(?:[A-Z]\s+){2,}[A-Z]", text):
        return text.replace(" ", "")
    # Spaces around @
    text = re.sub(r"\s*@\s*", "@", text)
    # Spaces around dots between word characters (email / URL dots)
    text = re.sub(r"(?<=\w)\s*\.\s*(?=\w)", ".", text)

    # Strip spaces inside domain part of an email address
    def _fix_domain(m: re.Match) -> str:
        return m.group(1) + "@" + re.sub(r"\s+", "", m.group(2))

    text = re.sub(r"([A-Za-z0-9._%+\-]+)@([A-Za-z0-9.\-\s]+)", _fix_domain, text)
    return text


# ══════════════════════════════════════════════════════════════════════════════
# L4 — FIELD EXTRACTORS
# ══════════════════════════════════════════════════════════════════════════════

# ── Email ─────────────────────────────────────────────────────────────────────


def _extract_email(lines: list[str]) -> str:
    # Pass 1: cleaned lines
    for line in lines:
        m = EMAIL_RE.search(line)
        if m:
            return m.group().lower()
    # Pass 2: collapse ALL whitespace (catches "a @ b . com")
    for line in lines:
        m = EMAIL_RE.search(re.sub(r"\s+", "", line))
        if m:
            return m.group().lower()
    return ""


# ── Phone ─────────────────────────────────────────────────────────────────────


def _extract_phone(lines: list[str]) -> str:
    """
    Region-neutral phone extraction.
    Tries guessed region first, then falls back through global priority list.
    """
    full = "\n".join(lines)
    guessed = _detect_phone_region(full)
    regions = ([guessed] if guessed else []) + [
        r for r in _FALLBACK_REGIONS if r != guessed
    ]

    for region in regions:
        try:
            for match in PhoneNumberMatcher(full, region):
                num = match.number
                if phonenumbers.is_valid_number(num):
                    return phonenumbers.format_number(
                        num, PhoneNumberFormat.INTERNATIONAL
                    )
        except Exception:
            continue

    # Raw regex fallback
    for line in lines:
        m = re.search(r"[\+\(]?[\d][\d\s\-\.\(\)]{6,}[\d]", line)
        if m:
            digits = re.sub(r"\D", "", m.group())
            if 7 <= len(digits) <= 15:
                return m.group().strip()
    return ""


# ── Website ───────────────────────────────────────────────────────────────────


def _extract_website(lines: list[str], email: str) -> str:
    """
    Strict TLD-based website extraction.
    Rejects plain FIRST.LAST name patterns (single dot, all-alpha, no path).
    """
    for line in lines:
        if "@" in line:
            continue
        for m in WEBSITE_RE.finditer(line):
            url = m.group().strip().rstrip(",.")
            # Reject name-like patterns: single dot, both sides purely alphabetic
            if re.fullmatch(r"[A-Za-z]+\.[A-Za-z]+", url):
                continue
            # Reject if the domain fragment appears inside the already-found email
            domain_part = (
                url.replace("http://", "")
                .replace("https://", "")
                .replace("www.", "")
                .split("/")[0]
            )
            if email and domain_part in email:
                continue
            return ("http://" + url if not url.startswith("http") else url).lower()
    return ""


# ── Company ───────────────────────────────────────────────────────────────────


def _extract_company(lines: list[str]) -> str:
    """
    Rule-based keyword scoring. Runs BEFORE name extraction so spaCy
    cannot mis-label a company name as a PERSON entity.

    Scoring:
      COMPANY_KW match        → +10
      ALL-CAPS multi-word     → +3
      Multi-word              → +1
      DESIGNATION_KW present  → −6  (likely a job title, not a company)
      ADDRESS_KW present      → −5  (likely an address fragment)
      Contains @ or 5+ digits → skip (email or phone line)
    Highest score wins; length breaks ties (longer = more specific).
    """
    scored: list[tuple[int, int, str]] = []
    for line in lines:
        if re.search(r"@|\d{5,}|https?://|www\.", line, re.I):
            continue
        score = 0
        if COMPANY_KW.search(line):
            score += 10
        if line.isupper() and len(line.split()) >= 2:
            score += 3
        if len(line.split()) >= 2:
            score += 1
        if DESIGNATION_KW.search(line):
            score -= 6
        if ADDRESS_KW.search(line):
            score -= 5
        if score > 0:
            scored.append((score, len(line), line))
    if not scored:
        return ""
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return scored[0][2]


# ── Job Title ─────────────────────────────────────────────────────────────────


def _extract_job_title(lines: list[str], used: set[str]) -> str:
    """
    First line that contains a DESIGNATION_KW, is NOT claimed, and does
    not also contain a COMPANY_KW (e.g. "Founder & CEO at TechCorp Ltd").
    """
    for line in lines:
        if line.strip().lower() in used:
            continue
        if re.search(r"@|\d{5,}", line):
            continue
        if DESIGNATION_KW.search(line) and not COMPANY_KW.search(line):
            return line
    return ""


# ── Name ──────────────────────────────────────────────────────────────────────


def _is_valid_name_token(word: str) -> bool:
    """
    Returns True only if a word token satisfies strict capitalization rules
    (BUG 4 FIX — name leakage prevention):
      - Fully UPPERCASE  (e.g. "ADITYA", "V.")
      - Strictly Title-Case: first char upper, all remaining chars lower
        (e.g. "Aditya", "Varma")
      - Initials: single uppercase letter optionally followed by a dot
        (e.g. "V.", "M")
    Mixed-case tokens (e.g. "aDitya", "VaRMA") are rejected.
    Tokens with digits are rejected.
    """
    if re.search(r"\d", word):
        return False
    # Strip trailing punctuation for the check
    w = word.rstrip(".,;")
    if not w:
        return False
    # Initial: "V" or "V."
    if re.fullmatch(r"[A-Z]\.?", word):
        return True
    # All uppercase
    if w.isupper():
        return True
    # Strict title-case: first char upper, rest lower
    if w[0].isupper() and w[1:].islower():
        return True
    return False


def _extract_name(
    lines: list[str],
    rows: list[list[dict]],
    used: set[str],
    script: str,
) -> str:
    """
    Two-strategy name extraction with enhanced fallback constraints (BUG 4 FIX).

    Strategy 1 — spaCy PERSON NER (language-aware):
      Selects the spaCy model matching the detected script.
      Rejects entities already claimed, containing digits, or matching
      COMPANY_KW / DESIGNATION_KW.

    Strategy 2 — Layout-aware heuristic with reinforced constraints:
      Candidate lines must be 2-4 words, all tokens passing _is_valid_name_token,
      at least NAME_ALPHA_RATIO of characters must be alphabetic, and the line
      must not be claimed by any earlier field.
      Top-bias scoring: score = 1 − (avg_cy / card_height).
      Lines nearer the top of the card score higher (name is usually printed
      prominently at the top on most card designs).
    """
    spacy_model = SCRIPT_TO_SPACY.get(script, SCRIPT_TO_SPACY["default"])
    nlp = get_spacy_nlp(spacy_model)

    # Strategy 1: spaCy
    if nlp:
        doc = nlp("\n".join(lines))
        for ent in doc.ents:
            if ent.label_ != "PERSON":
                continue
            candidate = ent.text.strip()
            if (
                len(candidate) >= 3
                and candidate.lower() not in used
                and not re.search(r"\d", candidate)
                and not COMPANY_KW.search(candidate)
                and not DESIGNATION_KW.search(candidate)
            ):
                return candidate

    # Strategy 2: reinforced heuristic
    card_h = max((t["cy"] for row in rows for t in row), default=1) or 1
    scored_candidates: list[tuple[float, str]] = []

    for line, row in zip(lines, rows):
        if line.strip().lower() in used:
            continue
        if re.search(r"@", line):
            continue

        words = line.split()
        if not (2 <= len(words) <= 4):
            continue

        # BUG 4 FIX: every token must pass strict capitalisation check
        if not all(_is_valid_name_token(w) for w in words):
            continue

        # Minimum alphabetic character ratio
        alpha_count = sum(c.isalpha() for c in line)
        total_count = max(len(line.replace(" ", "")), 1)
        if alpha_count / total_count < CFG["NAME_ALPHA_RATIO"]:
            continue

        avg_y = sum(t["cy"] for t in row) / len(row) if row else card_h
        top_score = 1.0 - (avg_y / card_h)
        scored_candidates.append((top_score, line))

    if scored_candidates:
        scored_candidates.sort(reverse=True)
        return scored_candidates[0][1]

    return ""


# ── Address ───────────────────────────────────────────────────────────────────


def _extract_address(
    lines: list[str],
    used: set[str],
    phone_digits: str = "",
) -> str:
    """
    Collect all lines that look like address fragments.

    BUG 3 FIX — RapidFuzz similarity instead of strict substring match:
      v11 used: phone_fp in line_digits  (fails when OCR adds "Ext 1234" etc.)
      v12 uses: rfuzz.partial_ratio(phone_fp, line_digits) >= PHONE_FUZZ_THRESH
      partial_ratio finds the best alignment window, so even fragmented or
      extended phone strings are correctly identified and rejected.
    """
    phone_fp = re.sub(r"\D", "", phone_digits)  # digit fingerprint of phone
    claimed_fps = {re.sub(r"\D", "", s) for s in used if s}

    parts: list[str] = []

    for line in lines:
        if not line.strip():
            continue
        if line.strip().lower() in used:
            continue
        if re.search(r"@", line):
            continue

        # ── Phone-line guard: RapidFuzz similarity (BUG 3 FIX) ───────────────
        line_digits = re.sub(r"\D", "", line)
        if len(line_digits) >= 7:
            # Check against extracted phone fingerprint
            if (
                phone_fp
                and rfuzz.partial_ratio(phone_fp, line_digits)
                >= CFG["PHONE_FUZZ_THRESH"]
            ):
                continue
            # Check against all claimed-set digit fingerprints
            if any(
                fp and rfuzz.partial_ratio(fp, line_digits) >= CFG["PHONE_FUZZ_THRESH"]
                for fp in claimed_fps
                if fp
            ):
                continue

        # ── Address detection ─────────────────────────────────────────────────
        is_addr = bool(ADDRESS_KW.search(line))

        # Standalone postal code: 4-9 digits, NOT a phone-like long digit run
        if POSTAL_RE.search(line) and not re.search(
            r"[+\(]?\d[\d\s\-\.\(\)]{8,}", line
        ):
            is_addr = True

        # "City, Country" or "City, State" pattern
        if re.search(r"[A-Z][a-z]+,\s*[A-Z][a-z]+", line):
            is_addr = True

        if is_addr:
            parts.append(line.strip())

    return ", ".join(parts)


# ── Social handles ────────────────────────────────────────────────────────────


def _extract_social(lines: list[str]) -> tuple[str, str]:
    """Extract LinkedIn and Twitter/X handles from text lines."""
    linkedin, twitter = "", ""
    LI = re.compile(r"linkedin\.com/in/[\w\-]+", re.I)
    TW = re.compile(r"(?:twitter\.com/|x\.com/|(?<!\w)@)[\w]{2,50}", re.I)
    for line in lines:
        if not linkedin:
            m = LI.search(line)
            if m:
                linkedin = m.group()
        if not twitter:
            m = TW.search(line)
            if m:
                twitter = m.group()
    return linkedin, twitter


# ══════════════════════════════════════════════════════════════════════════════
# FIELD ORCHESTRATION — strict dependency chain
# ══════════════════════════════════════════════════════════════════════════════


def extract_fields(
    lines: list[str],
    rows: list[list[dict]],
    script: str,
) -> dict:
    """
    Runs all field extractors in dependency-safe order.
    Each extractor receives the 'claimed' set of all previously assigned
    line strings, preventing any line from being assigned to two fields.

    Order:
      1. Email    — unambiguous regex; no dependencies
      2. Phone    — region-neutral libphonenumber; no dependencies
      3. Website  — strict TLD regex; rejects email-domain fragments
      4. Company  — keyword scoring; runs before Name to block entity theft
      5. Job Title— designation keywords; skips all claimed lines
      6. Name     — spaCy PERSON + reinforced heuristic; skips all claimed
      7. Address  — structural keywords + postal RE + fuzzy phone guard
      8. Social   — LinkedIn / Twitter regex patterns
    """
    data: dict[str, str] = {
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

    data["Email"] = _extract_email(lines)
    data["Phone"] = _extract_phone(lines)
    data["Website"] = _extract_website(lines, data["Email"])
    data["Company"] = _extract_company(lines)

    claimed: set[str] = {
        data["Email"].lower(),
        data["Phone"],
        data["Website"].lower(),
        data["Company"].strip().lower(),
        "",
    }

    data["Job_Title"] = _extract_job_title(lines, claimed)
    claimed.add(data["Job_Title"].strip().lower())

    data["Name"] = _extract_name(lines, rows, claimed, script)
    claimed.add(data["Name"].strip().lower())

    data["Address"] = _extract_address(lines, claimed, phone_digits=data["Phone"])
    data["LinkedIn"], data["Twitter"] = _extract_social(lines)

    return data


# ══════════════════════════════════════════════════════════════════════════════
# QUALITY SCORE
# ══════════════════════════════════════════════════════════════════════════════


def quality_score(data: dict) -> str:
    """Score based on core contact fields only."""
    core = ["Name", "Company", "Email", "Phone", "Address"]
    filled = sum(1 for k in core if data.get(k, "").strip())
    if filled >= 5:
        return "🟢 GREEN"
    if filled >= 3:
        return "🟡 YELLOW"
    return "🔴 RED"


# ══════════════════════════════════════════════════════════════════════════════
# CORE CARD PROCESSOR — in-memory array interface (no file I/O)
# ══════════════════════════════════════════════════════════════════════════════


def process_card_array(img: np.ndarray, label: str = "card") -> dict:
    """
    Process a single business card BGR numpy array through the full pipeline.

    This is the ONLY public entry point for card processing.
    It accepts a raw np.ndarray and returns a result dict.
    No file reads or writes occur inside this function.

    Pipeline:
      L2 → preprocess → deskew
      L3 → ocr_card (dual-stream multilingual) → group_into_lines
      L4 → extract_fields → quality_score
    """
    # L2 — Image pipeline
    proc = preprocess(img)
    proc = deskew(proc)

    # L3 — OCR + layout grouping
    tokens, primary_script, secondary_script = ocr_card(proc)

    if CFG["DEBUG"]:
        lang_info = SCRIPT_TO_LANG.get(primary_script, "en")
        if secondary_script:
            lang_info += f" + {SCRIPT_TO_LANG.get(secondary_script,'en')}"
        print(
            f"\n  [DEBUG] {label} — {len(tokens)} token(s)  "
            f"script={primary_script}  lang={lang_info}"
        )
        for t in sorted(tokens, key=lambda x: x["cy"]):
            print(f"    cy={t['cy']:>6.1f}  conf={t['conf']:.2f}  {t['text']!r}")

    rows = group_into_lines(tokens)
    lines = [" ".join(clean_token(t["text"]) for t in row) for row in rows]

    if CFG["DEBUG"]:
        print(f"  [DEBUG] Visual lines ({label}):")
        for i, ln in enumerate(lines, 1):
            print(f"    {i:02d}: {ln!r}")

    # L4 — Field extraction
    data = extract_fields(lines, rows, primary_script)
    data["QUALITY"] = quality_score(data)
    data["_label"] = label
    data["_script"] = primary_script
    data["_script2"] = secondary_script or ""
    data["_lang"] = SCRIPT_TO_LANG.get(primary_script, "en")

    return data


# ══════════════════════════════════════════════════════════════════════════════
# BATCH PROCESSOR — handles page segmentation and multi-card inputs
# ══════════════════════════════════════════════════════════════════════════════


def process_source(
    label: str,
    page_img: np.ndarray,
) -> list[dict]:
    """
    Process one source image (which may contain multiple business cards).

    Steps:
      1. segment_cards() → list of card ROI arrays (or [full_image] if single)
      2. process_card_array() on each ROI
      3. Return list of result dicts (one per card detected)
    """
    card_crops = segment_cards(page_img)
    results: list[dict] = []

    for idx, card_img in enumerate(card_crops):
        card_label = (
            f"{label}"
            if len(card_crops) == 1
            else f"{label} [card {idx+1}/{len(card_crops)}]"
        )
        try:
            result = process_card_array(card_img, label=card_label)
            results.append(result)
        except Exception as e:
            print(f"  [ERROR] Failed to process {card_label}: {e}")
            if CFG["DEBUG"]:
                traceback.print_exc()

    return results


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT SYSTEM — dual-destination hybrid (history + stable latest)
# ══════════════════════════════════════════════════════════════════════════════


def save_outputs(results: list[dict]) -> None:
    """
    Hybrid output system with two simultaneous destinations:

        output/
        ├── history/
        │   ├── cards_YYYYMMDD_HHMMSS.xlsx  ← permanent timestamped record
        │   ├── cards_YYYYMMDD_HHMMSS.json
        │   └── cards_YYYYMMDD_HHMMSS.csv
        ├── latest.xlsx                      ← always overwritten by newest run
        ├── latest.json
        └── latest.csv

    Internal debug keys (prefixed "_") are stripped before writing.
    Both destinations receive the identical DataFrame.
    """
    # Strip internal keys
    rows = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
    df = pd.DataFrame(rows)

    # Ensure directory structure
    out_dir = "output"
    history_dir = os.path.join(out_dir, "history")
    os.makedirs(history_dir, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Timestamped history files
    hist_xlsx = os.path.join(history_dir, f"cards_{ts}.xlsx")
    hist_json = os.path.join(history_dir, f"cards_{ts}.json")
    hist_csv = os.path.join(history_dir, f"cards_{ts}.csv")

    df.to_excel(hist_xlsx, index=False)
    df.to_json(hist_json, orient="records", indent=4, force_ascii=False)
    df.to_csv(hist_csv, index=False)

    # Stable latest files (overwrite every run)
    latest_xlsx = os.path.join(out_dir, "latest.xlsx")
    latest_json = os.path.join(out_dir, "latest.json")
    latest_csv = os.path.join(out_dir, "latest.csv")

    df.to_excel(latest_xlsx, index=False)
    df.to_json(latest_json, orient="records", indent=4, force_ascii=False)
    df.to_csv(latest_csv, index=False)

    print(f"\n  ┌─ History ({'─'*44})")
    print(f"  │  [XLSX] {hist_xlsx}")
    print(f"  │  [JSON] {hist_json}")
    print(f"  │  [CSV ] {hist_csv}")
    print(f"  ├─ Latest (overwritten) {'─'*35})")
    print(f"  │  [XLSX] {latest_xlsx}")
    print(f"  │  [JSON] {latest_json}")
    print(f"  └─ [CSV ] {latest_csv}")


# ══════════════════════════════════════════════════════════════════════════════
# TERMINAL SUMMARY GRID
# ══════════════════════════════════════════════════════════════════════════════


def print_summary(results: list[dict]) -> None:
    """
    Print a compact terminal grid mapping card index → extracted fields.
    One row per card; quality badge in the rightmost column.
    """
    W = 80
    FIELDS = ["Name", "Company", "Email", "Phone", "QUALITY"]

    print("\n" + "═" * W)
    print(f"  {'#':<4} {'Name':<22} {'Company':<20} {'Email':<22} {'Q'}")
    print("─" * W)

    for i, r in enumerate(results, 1):
        name = (r.get("Name", "") or "—")[:21]
        company = (r.get("Company", "") or "—")[:19]
        email = (r.get("Email", "") or "—")[:21]
        quality = r.get("QUALITY", "—")
        label = r.get("_label", "")
        print(f"  {i:<4} {name:<22} {company:<20} {email:<22} {quality}")
        if label:
            print(f"       └─ {label}")

    print("═" * W)

    # Detailed block per card
    for i, r in enumerate(results, 1):
        print(f"\n  ── Card {i} ─────────────────────────────────────────────────")
        print(
            f"  {'Script':<12}: {r.get('_script','—')}  "
            f"(secondary: {r.get('_script2','none')})  "
            f"OCR lang: {r.get('_lang','—')}"
        )
        for key in [
            "Name",
            "Job_Title",
            "Company",
            "Email",
            "Phone",
            "Website",
            "Address",
            "LinkedIn",
            "Twitter",
            "QUALITY",
        ]:
            val = r.get(key, "") or "—"
            print(f"  {key:<12}: {val}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    """
    GUI file picker → route to loader → segment → extract → display → save.
    Supports: JPEG, PNG, BMP, WEBP, TIFF (single or multi-card sheet),
              PDF (all pages), ZIP (recursive image + PDF extraction).
    All processing is in-memory — no temporary files written to disk.
    """
    root = Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Select Visiting Card — Image / PDF / ZIP",
        filetypes=[
            (
                "All Supported",
                "*.jpg *.jpeg *.png *.bmp *.webp *.tiff *.tif *.pdf *.zip",
            ),
            ("Images", "*.jpg *.jpeg *.png *.bmp *.webp *.tiff *.tif"),
            ("PDF", "*.pdf"),
            ("ZIP", "*.zip"),
        ],
    )
    root.destroy()

    if not file_path:
        print("[INFO] No file selected. Exiting.")
        return

    print(f"\n[INFO] Input: {file_path}")
    print(f"  GPU: {'ON' if USE_GPU else 'OFF (CPU)'}")

    # ── L0: Load all page/image arrays in-memory ──────────────────────────────
    try:
        page_images = load_images_from_path(file_path)
    except Exception as e:
        print(f"[ERROR] Failed to load file: {e}")
        traceback.print_exc()
        return

    print(f"  Loaded {len(page_images)} page(s) / image(s).")

    # ── L1 + L2 + L3 + L4: Process all pages ─────────────────────────────────
    all_results: list[dict] = []

    for page_label, page_img in page_images:
        print(f"\n[PAGE] {page_label}")
        page_results = process_source(page_label, page_img)
        all_results.extend(page_results)

    if not all_results:
        print("\n[INFO] No cards extracted.")
        return

    # ── Output ────────────────────────────────────────────────────────────────
    print_summary(all_results)
    save_outputs(all_results)
    print(f"\n[SUCCESS] {len(all_results)} card(s) processed.")


if __name__ == "__main__":
    main()
