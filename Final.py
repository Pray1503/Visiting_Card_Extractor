#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           VISITING CARD OCR EXTRACTOR — ULTIMATE BUILD v5.0                ║
║                                                                              ║
║  Extracts : Name · Phone · Email · Company · Address · Job Title            ║
║  Input    : JPG · PNG · PDF · TIFF · BMP                                    ║
║  Output   : Colour-coded Excel sheet + full debug log                       ║
║                                                                              ║
║  CHANGELOG v4.0 → v5.0  (confirmed by diagnostic image analysis)           ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  ROOT CAUSE IDENTIFIED AND FIXED:                                           ║
║                                                                              ║
║  _is_dark_background() used np.mean() on the FULL image including the      ║
║  white border/background surrounding the card.  The white border raised     ║
║  the mean above 128, so the dark charcoal card was NEVER inverted.          ║
║  EasyOCR received white-text-on-dark which caused every downstream error:   ║
║    • Name split into "A D ITYA" [0.43 RED] + "V . VAR MA" [0.83]          ║
║    • Email fragmented into 5 separate tokens, never assembled               ║
║    • Job title read as "P RIN CIPAL ARCHITECT & FOUNDER"                   ║
║    • Company missing "INFRA" token                                          ║
║    • "Surt" misread for "Surat"                                             ║
║    • Quality cascaded to YELLOW                                             ║
║                                                                              ║
║  FIX-8  Centre-crop dark background detection                               ║
║         Samples middle 40%×40% of image (card body, not border)            ║
║         Two-condition check: mean AND dark-pixel-ratio must both agree      ║
║  FIX-9  Stronger contrast boost for inverted dark cards (α=2.0 vs 1.4)    ║
║                                                                              ║
║  All v4.0 fixes (FIX-1 through FIX-7) retained.                           ║
╚══════════════════════════════════════════════════════════════════════════════╝

INSTALL DEPENDENCIES:
    pip install opencv-python easyocr pandas numpy torch pdf2image
                phonenumbers pyspellchecker rapidfuzz openpyxl Pillow

POPPLER (required for PDF support only):
    Windows → https://github.com/oschwartz10612/poppler-windows/releases
              Set env var: POPPLER_PATH=C:\\poppler\\Library\\bin
    Linux   → sudo apt install poppler-utils
    macOS   → brew install poppler
"""

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — IMPORTS
# ══════════════════════════════════════════════════════════════════════════════
import os
import re
import sys
import logging
import traceback
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import easyocr
import phonenumbers
from phonenumbers import NumberParseException, PhoneNumberFormat
from rapidfuzz import process as fuzz_process, fuzz
from spellchecker import SpellChecker
from pdf2image import convert_from_path
from tkinter import Tk, filedialog
from openpyxl.styles import PatternFill, Font, Alignment
from openpyxl.utils import get_column_letter

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
CONFIG = {
    # ── Preprocessing ─────────────────────────────────────────────────────
    "RESIZE_SCALE": 0.85,
    "CONTRAST_ALPHA": 1.4,
    "CONTRAST_BETA": 0,
    "STD_DEV_HIGH": 60,
    "ADAPTIVE_BLOCK_SIZE": 15,
    "ADAPTIVE_C": 8,
    # ── Dark Background ───────────────────────────────────────────────────
    "DARK_BG_THRESHOLD": 128,
    # ── Card Segmentation ─────────────────────────────────────────────────
    "MIN_CARD_HEIGHT_PX": 200,
    "MIN_CARD_WIDTH_PX": 150,
    "SEGMENT_PADDING_PX": 15,
    "NOISE_THRESHOLD_RATIO": 0.01,
    # ── OCR ───────────────────────────────────────────────────────────────
    "OCR_WIDTH_THRESHOLD": 0.7,
    "OCR_MIN_CONFIDENCE": 0.45,
    # ── FIX-2: Cross-token join window ────────────────────────────────────
    # Tokens whose bounding boxes are within this many pixels vertically
    # are considered to be on the same line and are joined before field
    # extraction.  Tuned to ~1.5× typical character height on a 300-DPI scan.
    "SAME_LINE_Y_TOLERANCE_PX": 18,
    # ── FIX-3: Adjacent split-word gap ────────────────────────────────────
    # Two consecutive tokens that are horizontally adjacent within this many
    # pixels AND whose combined text forms a known keyword are merged.
    # Prevents "FOUND ER", "ARCHI TECT", "SOLU TIONS" etc.
    "SPLIT_WORD_X_GAP_PX": 22,
    # ── Name Detection ────────────────────────────────────────────────────
    "NAME_ZONE_RATIO": 0.65,  # Raised from 0.35 — centred names need more room
    "NAME_MIN_WORDS": 1,
    "NAME_MAX_WORDS": 6,  # Raised from 4 — "Aditya V. Varma" = 3 but
    # names like "Dr. Priya Anand Sharma" = 4
    # ── Spell Correction ──────────────────────────────────────────────────
    "SPELL_MAX_LEN_DIFF": 2,
    # ── Phone Parsing ─────────────────────────────────────────────────────
    "DEFAULT_REGION": "IN",
    # ── Fuzzy Matching ────────────────────────────────────────────────────
    "FUZZY_SCORE_CUTOFF": 82,
    # ── FIX-4: PIN↔city cross-validation map ─────────────────────────────
    # Maps 6-digit PIN code prefix → canonical city name.
    # When a token near a PIN code fuzzy-matches a city in this map,
    # the city name is corrected to the canonical form regardless of
    # what pyspellchecker says.  Extend freely.
    "PIN_PREFIX_TO_CITY": {
        "395": "Surat",
        "380": "Ahmedabad",
        "390": "Vadodara",
        "360": "Rajkot",
        "382": "Gandhinagar",
        "400": "Mumbai",
        "411": "Pune",
        "560": "Bangalore",
        "110": "Delhi",
        "500": "Hyderabad",
        "600": "Chennai",
        "700": "Kolkata",
        "302": "Jaipur",
        "226": "Lucknow",
        "452": "Indore",
        "440": "Nagpur",
        "160": "Chandigarh",
        "682": "Kochi",
        "641": "Coimbatore",
    },
    # ── Output ────────────────────────────────────────────────────────────
    "OUTPUT_FILENAME": "Visiting_Card_Data_ULTIMATE.xlsx",
    "LOG_FILENAME": "ocr_pipeline.log",
    # ── Poppler ───────────────────────────────────────────────────────────
    "POPPLER_PATH": os.environ.get("POPPLER_PATH", None),
    "PDF_DPI": 200,
}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — LOGGING
# ══════════════════════════════════════════════════════════════════════════════
def _setup_logging() -> logging.Logger:
    logger = logging.getLogger("card_ocr")
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(message)s", datefmt="%H:%M:%S"
    )
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    fh = logging.FileHandler(CONFIG["LOG_FILENAME"], mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


log = _setup_logging()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — DOMAIN CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
JOB_TITLE_KEYWORDS: frozenset = frozenset(
    {
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
        "representative",
        "agent",
        "broker",
        "dealer",
        "distributor",
    }
)

COMPANY_KEYWORDS: frozenset = frozenset(
    {
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
        "realty",
        "properties",
        "pharma",
        "healthcare",
        "clinic",
        "hospital",
        "school",
        "academy",
        "institute",
        "trading",
        "exports",
        "imports",
        "logistics",
        "finance",
        "investments",
    }
)

ADDRESS_KEYWORDS: frozenset = frozenset(
    {
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
        "near",
        "opp",
        "opposite",
        "highway",
        "junction",
        "bridge",
        "station",
        "airport",
        "cross",
        "marg",
        "chowk",
        "bazaar",
        "market",
        "ring road",
        "wing",
    }
)

KNOWN_CITIES: frozenset = frozenset(
    {
        "ahmedabad",
        "surat",
        "mumbai",
        "pune",
        "bangalore",
        "bengaluru",
        "delhi",
        "new delhi",
        "hyderabad",
        "chennai",
        "kolkata",
        "jaipur",
        "vadodara",
        "rajkot",
        "gandhinagar",
        "noida",
        "gurugram",
        "gurgaon",
        "indore",
        "bhopal",
        "nagpur",
        "lucknow",
        "chandigarh",
        "kochi",
        "coimbatore",
        "visakhapatnam",
        "patna",
        "agra",
        "nashik",
        "faridabad",
        "meerut",
        "thane",
        "navi mumbai",
        "aurangabad",
        "srinagar",
        "amritsar",
        "dhanbad",
    }
)

BUSINESS_CARD_VOCAB: set = {
    "ahmedabad",
    "surat",
    "vadodara",
    "rajkot",
    "gandhinagar",
    "mumbai",
    "pune",
    "bangalore",
    "bengaluru",
    "delhi",
    "hyderabad",
    "chennai",
    "kolkata",
    "jaipur",
    "lucknow",
    "indore",
    "nagpur",
    "gurugram",
    "ltd",
    "llp",
    "pvt",
    "inc",
    "corp",
    "co",
    "llc",
    "limited",
    "solutions",
    "consulting",
    "technologies",
    "infotech",
    "services",
    "enterprises",
    "associates",
    "group",
    "studio",
    "labs",
    "tech",
    "digital",
    "systems",
    "global",
    "ventures",
    "innovations",
    "infosys",
    "ceo",
    "cto",
    "cfo",
    "coo",
    "md",
    "vp",
    "avp",
    "gm",
    "agm",
    "director",
    "manager",
    "engineer",
    "developer",
    "consultant",
    "analyst",
    "executive",
    "officer",
    "partner",
    "founder",
    "specialist",
    "coordinator",
    "architect",
    "designer",
    "tower",
    "floor",
    "building",
    "complex",
    "nagar",
    "road",
    "street",
    "avenue",
    "sector",
    "phase",
    "block",
    "wing",
    "plaza",
    "centre",
    "center",
    "park",
    "square",
    "chowk",
    "infra",
    "infrastructure",
    "skytower",
    "skytoiver",  # common OCR variant of SkyTower
}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — SPELL CORRECTION
# ══════════════════════════════════════════════════════════════════════════════
spell = SpellChecker()
spell.word_frequency.load_words(BUSINESS_CARD_VOCAB)


def smart_correct(word: str) -> str:
    """
    Corrects a single word only when safe to do so.
    See v3.0 for full rationale — logic unchanged here.
    Contextual city correction (FIX-4) is handled separately in
    correct_city_near_pin() because it requires the surrounding PIN code.
    """
    if len(word) <= 2:
        return word
    if re.search(r"\d", word):
        return word
    if "@" in word or "." in word:
        return word
    if word.isupper() and len(word) <= 5:
        return word

    lower = word.lower()
    if lower in spell:
        return word

    correction = spell.correction(lower)
    if correction is None or correction == lower:
        return word
    if abs(len(correction) - len(lower)) > CONFIG["SPELL_MAX_LEN_DIFF"]:
        return word

    if word.isupper():
        return correction.upper()
    if word[0].isupper():
        return correction.capitalize()
    return correction


def correct_text(text: str) -> str:
    """Applies smart_correct to every word, rebuilding the string."""
    return " ".join(smart_correct(w) for w in text.split())


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — FIX-1 + FIX-3: TOKEN-LEVEL RECONSTRUCTION
#
# Two problems addressed here:
#
# FIX-1 — Spaced-letter reconstruction
# ────────────────────────────────────
# Some professional card fonts (wide-tracked, spaced caps) cause EasyOCR to
# read each letter as an individual token:
#     ["A", "D", "I", "T", "Y", "A", "V", ".", "V", "A", "R", "M", "A"]
# or as single-letter tokens on the same Y line:
#     ["A D I T Y A", "V .", "V A R M A"]
#
# Detection: tokens that are ALL single characters (or single char + dot)
# on the same baseline are collapsed into one word.
#
# FIX-3 — Split-word merging
# ──────────────────────────
# Kerning artefacts or tight bounding-box clipping cause OCR to split words:
#     "FOUND" + "ER" → should be "FOUNDER"
#     "ARCHI" + "TECT" → should be "ARCHITECT"
#     "SOLU"  + "TIONS" → should be "SOLUTIONS"
#
# Detection: two consecutive tokens on the same line within SPLIT_WORD_X_GAP_PX
# whose concatenation (no space) matches a word in our vocab OR a known keyword.
# ══════════════════════════════════════════════════════════════════════════════


def _bbox_top_y(bbox) -> float:
    """Y coordinate of the top edge of an EasyOCR bounding box."""
    return float(bbox[0][1])


def _bbox_bottom_y(bbox) -> float:
    return float(bbox[2][1])


def _bbox_mid_y(bbox) -> float:
    return (_bbox_top_y(bbox) + _bbox_bottom_y(bbox)) / 2.0


def _bbox_left_x(bbox) -> float:
    return float(bbox[0][0])


def _bbox_right_x(bbox) -> float:
    return float(bbox[1][0])


def _on_same_line(bbox_a, bbox_b) -> bool:
    """
    Returns True when two bounding boxes share the same text baseline,
    i.e. their vertical midpoints are within SAME_LINE_Y_TOLERANCE_PX.
    """
    return (
        abs(_bbox_mid_y(bbox_a) - _bbox_mid_y(bbox_b))
        <= CONFIG["SAME_LINE_Y_TOLERANCE_PX"]
    )


def _is_spaced_letter_token(text: str) -> bool:
    """
    Returns True if every space-separated segment in `text` is a single
    character (or a single char followed by a dot, e.g. "V.").
    Identifies spaced-letter OCR artefacts like "A D I T Y A" or "V .".
    """
    segments = text.strip().split()
    if len(segments) < 2:
        return False
    return all(re.fullmatch(r"[A-Za-z]\.?", seg) for seg in segments)


def _collapse_spaced_letters(text: str) -> str:
    """
    "A D I T Y A" → "ADITYA"
    "V ."         → "V."
    "V A R M A"   → "VARMA"
    Dot after a letter is kept (middle initial like "V.").
    """
    segments = text.strip().split()
    return "".join(segments)


def _concat_forms_keyword(a: str, b: str) -> bool:
    """
    Returns True if concatenating token `a` and token `b` (no space) produces
    a string that is in our BUSINESS_CARD_VOCAB, JOB_TITLE_KEYWORDS, or
    COMPANY_KEYWORDS — indicating a split word.
    Case-insensitive check.
    """
    combined = (a + b).lower()
    all_keywords = BUSINESS_CARD_VOCAB | JOB_TITLE_KEYWORDS | COMPANY_KEYWORDS
    return combined in all_keywords


def reconstruct_tokens(ocr_results: list) -> list:
    """
    Two-pass token reconstruction applied BEFORE any field extractor runs.

    Pass A — Spaced-letter reconstruction (FIX-1)
    ─────────────────────────────────────────────
    Groups consecutive tokens on the same baseline that are individually
    spaced-letter patterns.  Merges them into a single token joined by spaces
    at word boundaries.

    Example (3 tokens on same line):
        ("A D I T Y A", conf=0.95)  ("V .", conf=0.97)  ("V A R M A", conf=0.94)
        → ("ADITYA V. VARMA", conf=0.95)   (avg confidence)

    Pass B — Split-word merging (FIX-3)
    ────────────────────────────────────
    Scans consecutive token pairs on the same line.  If their horizontal gap
    is ≤ SPLIT_WORD_X_GAP_PX AND their concatenation forms a known keyword,
    they are merged.

    Example:
        ("FOUND", conf=0.88)  ("ER", conf=0.82)
        → ("FOUNDER", conf=0.85)

    Both passes preserve average confidence so downstream thresholds still work.
    """

    # ── Pass A: Spaced-letter reconstruction ─────────────────────────────
    # Sort by (mid_y, left_x) so same-line tokens appear consecutively.
    sorted_results = sorted(
        ocr_results, key=lambda r: (_bbox_mid_y(r[0]), _bbox_left_x(r[0]))
    )

    merged_a: list = []
    skip: set = set()

    for i, (bbox_i, text_i, conf_i) in enumerate(sorted_results):
        if i in skip:
            continue

        if not _is_spaced_letter_token(text_i):
            merged_a.append((bbox_i, text_i, conf_i))
            continue

        # This token looks like "A D I T Y A" — collect adjacent same-line
        # spaced-letter tokens to build the full name
        group_bboxes = [bbox_i]
        group_texts = [_collapse_spaced_letters(text_i)]
        group_confs = [conf_i]

        j = i + 1
        while j < len(sorted_results) and j not in skip:
            bbox_j, text_j, conf_j = sorted_results[j]
            if not _on_same_line(bbox_i, bbox_j):
                break
            if _is_spaced_letter_token(text_j):
                group_texts.append(_collapse_spaced_letters(text_j))
                group_confs.append(conf_j)
                group_bboxes.append(bbox_j)
                skip.add(j)
            else:
                # Non-spaced-letter token on same line — stop collecting
                break
            j += 1

        # Build a merged bbox (top-left of first, bottom-right of last)
        tl = group_bboxes[0][0]
        tr = group_bboxes[-1][1]
        br = group_bboxes[-1][2]
        bl = group_bboxes[0][3]
        merged_bbox = [tl, tr, br, bl]

        # Rejoin collapsed letter-groups with spaces where word boundaries are
        # Heuristic: if collapsed segment has > 1 char it's a word; separate by space
        rejoined = " ".join(group_texts)
        avg_conf = float(np.mean(group_confs))

        log.debug(
            f"  [FIX-1] Spaced-letter merge: "
            f"{[t for t in group_texts]} → '{rejoined}' ({avg_conf:.2f})"
        )
        merged_a.append((merged_bbox, rejoined, avg_conf))

    # ── Pass B: Split-word merging ────────────────────────────────────────
    merged_b: list = []
    skip_b: set = set()

    for i, (bbox_i, text_i, conf_i) in enumerate(merged_a):
        if i in skip_b:
            continue

        if i + 1 < len(merged_a):
            bbox_j, text_j, conf_j = merged_a[i + 1]

            gap = _bbox_left_x(bbox_j) - _bbox_right_x(bbox_i)
            same_line = _on_same_line(bbox_i, bbox_j)
            close_enough = gap <= CONFIG["SPLIT_WORD_X_GAP_PX"]

            if same_line and close_enough and _concat_forms_keyword(text_i, text_j):
                merged_text = text_i + text_j  # No space — they ARE one word
                merged_conf = float(np.mean([conf_i, conf_j]))
                tl = bbox_i[0]
                tr = bbox_j[1]
                br = bbox_j[2]
                bl = bbox_i[3]
                merged_bbox = [tl, tr, br, bl]
                log.debug(
                    f"  [FIX-3] Split-word merge: "
                    f"'{text_i}' + '{text_j}' → '{merged_text}' ({merged_conf:.2f})"
                )
                merged_b.append((merged_bbox, merged_text, merged_conf))
                skip_b.add(i + 1)
                continue

        merged_b.append((bbox_i, text_i, conf_i))

    return merged_b


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — FIX-2: LINE-LEVEL JOINING FOR EMAIL DETECTION
#
# EasyOCR can return a spaced email like "aditya.v @ urbanedgeinfra . co . in"
# as MULTIPLE tokens, e.g.:
#     "aditya.v"  "@"  "urbanedgeinfra"  "."  "co"  "."  "in"
# or as one token with internal spaces:
#     "aditya.v @ urbanedgeinfra . co . in"
#
# The v3.0 code only stripped spaces within a SINGLE token.  Multi-token
# emails were never assembled and always returned "N/A".
#
# Fix: group all tokens on the same line, join them without spaces, then
# run the email regex on the joined string.  This is safe because a real
# email address on a card is always on its own line.
# ══════════════════════════════════════════════════════════════════════════════


def build_line_strings(ocr_results: list) -> list:
    """
    Groups tokens into visual lines using Y-coordinate proximity, then
    returns a list of (joined_no_space_string, avg_conf, representative_bbox)
    for every detected line.

    This gives extract_email() a complete, space-stripped version of each
    line to match against, catching multi-token email addresses.
    """
    if not ocr_results:
        return []

    # Sort by mid-Y then left-X (reading order)
    sorted_r = sorted(
        ocr_results, key=lambda r: (_bbox_mid_y(r[0]), _bbox_left_x(r[0]))
    )

    lines: list = []  # list of [(bbox, text, conf), ...]
    current_line: list = [sorted_r[0]]

    for item in sorted_r[1:]:
        if _on_same_line(item[0], current_line[0][0]):
            current_line.append(item)
        else:
            lines.append(current_line)
            current_line = [item]
    lines.append(current_line)

    result = []
    for line_tokens in lines:
        joined_nospace = re.sub(r"\s+", "", "".join(t for _, t, _ in line_tokens))
        joined_space = " ".join(t for _, t, _ in line_tokens)
        avg_conf = float(np.mean([c for _, _, c in line_tokens]))
        rep_bbox = line_tokens[0][0]
        result.append(
            {
                "nospace": joined_nospace,
                "spaced": joined_space,
                "conf": avg_conf,
                "bbox": rep_bbox,
                "tokens": line_tokens,
            }
        )

    return result


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — FIX-4: PIN↔CITY CROSS-VALIDATION
#
# Problem: "Surt" and "Sure" both pass pyspellchecker unchanged because:
#   • "Surt" is 4 chars, ALL-CAPS → smart_correct() skips ALL-CAPS ≤ 5
#   • "Sure" is a valid English word → spellchecker returns "sure"
#
# Fix: when we find a 6-digit PIN code in the OCR output, look up the
# canonical city for that PIN prefix in CONFIG["PIN_PREFIX_TO_CITY"].
# Any token on the same line as the PIN that fuzzy-matches the expected
# city name is replaced with the canonical city name.
#
# This is purely contextual — we only correct a city when we have the
# independent evidence of a nearby PIN code to validate against.
# ══════════════════════════════════════════════════════════════════════════════


def correct_city_near_pin(ocr_results: list) -> list:
    """
    Scans all tokens for 6-digit PIN codes, derives the canonical city from
    CONFIG["PIN_PREFIX_TO_CITY"], then corrects any nearby city-like token.

    Returns a new list of (bbox, text, conf) tuples with city names fixed.
    """
    PIN_RE = re.compile(r"\b(\d{6})\b")
    corrected = list(ocr_results)  # shallow copy — we'll replace entries

    # Build a lookup: index → canonical_city for every token that contains a PIN
    pin_corrections: dict = {}  # token_index → canonical_city

    for idx, (bbox, text, conf) in enumerate(corrected):
        m = PIN_RE.search(text)
        if not m:
            continue
        pin = m.group(1)
        prefix = pin[:3]
        canonical = CONFIG["PIN_PREFIX_TO_CITY"].get(prefix)
        if canonical:
            log.debug(f"  [FIX-4] PIN {pin} → expected city '{canonical}'")
            pin_corrections[idx] = canonical

    if not pin_corrections:
        return corrected

    # For every identified PIN token, check its same-line neighbours
    for pin_idx, canonical_city in pin_corrections.items():
        pin_bbox = corrected[pin_idx][0]

        for idx, (bbox, text, conf) in enumerate(corrected):
            if idx == pin_idx:
                continue
            if not _on_same_line(pin_bbox, bbox):
                continue

            # Does this token look like a garbled city name?
            score = fuzz.partial_ratio(text.lower(), canonical_city.lower())
            if score >= CONFIG["FUZZY_SCORE_CUTOFF"]:
                log.debug(
                    f"  [FIX-4] City correction: '{text}' → '{canonical_city}' "
                    f"(score={score}, PIN={corrected[pin_idx][1]})"
                )
                corrected[idx] = (bbox, canonical_city, conf)

    return corrected


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — IMAGE PREPROCESSING
#
# ROOT CAUSE FIX (confirmed by diagnostic images):
# ─────────────────────────────────────────────────
# The original _is_dark_background() called np.mean() on the ENTIRE image.
# This card image has a large white border/background surrounding the dark
# card.  The white border pixels raised the mean well above 128, so the
# function returned False and the card was NEVER inverted — even though the
# card itself is dark charcoal with white text.
#
# Result: EasyOCR received white-text-on-dark-background, which caused:
#   • "A D ITYA" split + confidence 0.43 (dropped as RED)
#   • "V . VAR MA" partial name read
#   • Email split across 5+ tokens
#   • "Surt" misread for "Surat"
#   • Overall quality YELLOW instead of GREEN
#
# Fix — Three-layer dark detection:
#   Layer 1: Centre-crop test  — sample the middle 40% of the image where
#            the card body always falls, ignoring surrounding white space.
#   Layer 2: Dark-pixel-ratio  — count pixels below threshold; if >50% of
#            the centre crop is dark, it's a dark card.
#   Layer 3: Dual-pass OCR     — for dark cards, run OCR on BOTH the
#            inverted image AND the original, then keep whichever pass
#            yields more high-confidence tokens.  This handles edge cases
#            where only part of the card is dark.
# ══════════════════════════════════════════════════════════════════════════════


def _is_dark_background(gray: np.ndarray) -> bool:
    """
    Detects dark-background cards by sampling the CENTRE of the image,
    not the full image mean.

    Why centre crop:
        Cards photographed or scanned against a white surface have large
        bright borders.  np.mean() over the full image is dominated by those
        borders and returns a value well above 128 even for a jet-black card.

        Sampling the central 40% × 40% region captures the card body and
        ignores the surrounding background entirely.

    Two-condition check (both must agree to call it dark):
        1. Centre-crop mean < DARK_BG_THRESHOLD  (average darkness)
        2. >50% of centre-crop pixels are below threshold (majority is dark)

    This prevents a card with a single large dark logo from being falsely
    classified as dark-background.
    """
    h, w = gray.shape
    # Sample centre 40% in each dimension
    y1, y2 = int(h * 0.30), int(h * 0.70)
    x1, x2 = int(w * 0.30), int(w * 0.70)
    centre = gray[y1:y2, x1:x2]

    centre_mean = float(np.mean(centre))
    dark_pixel_ratio = float(np.sum(centre < CONFIG["DARK_BG_THRESHOLD"])) / centre.size

    is_dark = (centre_mean < CONFIG["DARK_BG_THRESHOLD"]) and (dark_pixel_ratio > 0.50)
    log.debug(
        f"  Dark-bg check: centre_mean={centre_mean:.1f} "
        f"dark_pixel_ratio={dark_pixel_ratio:.2f} → {'DARK' if is_dark else 'LIGHT'}"
    )
    return is_dark


def preprocess_image(img: np.ndarray) -> np.ndarray:
    """
    Converts a BGR card image into a preprocessed single-channel image.

    v5.0 change:
        Uses the fixed _is_dark_background() (centre-crop sampling).
        For dark cards, applies inversion BEFORE thresholding so EasyOCR
        always receives dark-text-on-light-background.

        Additionally: for dark cards, the contrast boost alpha is raised
        from 1.4 → 2.0 because inverted dark cards have lower native
        contrast and need a stronger boost to make white-on-dark text
        fully black after inversion.
    """
    scale = CONFIG["RESIZE_SCALE"]
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    dark = _is_dark_background(gray)
    if dark:
        gray = cv2.bitwise_not(gray)
        log.debug("  Dark background confirmed — image inverted.")
        # Stronger contrast for inverted dark cards
        alpha = 2.0
    else:
        alpha = CONFIG["CONTRAST_ALPHA"]

    std_dev = float(np.std(gray))
    if std_dev > CONFIG["STD_DEV_HIGH"]:
        log.debug(f"  Preprocessing: global contrast boost α={alpha} (σ={std_dev:.1f})")
        return cv2.convertScaleAbs(gray, alpha=alpha, beta=CONFIG["CONTRAST_BETA"])
    else:
        log.debug(f"  Preprocessing: adaptive threshold (σ={std_dev:.1f})")
        return cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=CONFIG["ADAPTIVE_BLOCK_SIZE"],
            C=CONFIG["ADAPTIVE_C"],
        )


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — CARD SEGMENTATION  (unchanged from v3.0)
# ══════════════════════════════════════════════════════════════════════════════


def _find_valleys(projection, span, min_size, axis="h"):
    noise_threshold = span * CONFIG["NOISE_THRESHOLD_RATIO"]
    bounds = []
    start = None
    for i, val in enumerate(projection):
        if val > noise_threshold and start is None:
            start = i
        elif val <= noise_threshold and start is not None:
            if i - start >= min_size:
                bounds.append((start, i))
            start = None
    if start is not None and (len(projection) - start) >= min_size:
        bounds.append((start, len(projection)))
    return bounds


def segment_cards(image: np.ndarray) -> list:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    height, width = gray.shape
    pad = CONFIG["SEGMENT_PADDING_PX"]
    all_crops = []

    h_proj = np.sum(binary, axis=1)
    h_bounds = _find_valleys(h_proj, width, CONFIG["MIN_CARD_HEIGHT_PX"], axis="h")
    h_slices = h_bounds if h_bounds else [(0, height)]

    for y1, y2 in h_slices:
        h_crop = image[max(0, y1 - pad) : min(height, y2 + pad), :]
        gray_s = cv2.cvtColor(h_crop, cv2.COLOR_BGR2GRAY)
        _, bin_s = cv2.threshold(
            gray_s, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        v_proj = np.sum(bin_s, axis=0)
        v_bounds = _find_valleys(
            v_proj, h_crop.shape[0], CONFIG["MIN_CARD_WIDTH_PX"], axis="v"
        )

        if len(v_bounds) > 1:
            for x1, x2 in v_bounds:
                v_crop = h_crop[:, max(0, x1 - pad) : min(h_crop.shape[1], x2 + pad)]
                all_crops.append(v_crop)
        else:
            all_crops.append(h_crop)

    result = all_crops if all_crops else [image]
    log.info(f"  Segmented into {len(result)} card(s).")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — CONFIDENCE FILTER  (unchanged from v3.0)
# ══════════════════════════════════════════════════════════════════════════════


def filter_by_confidence(ocr_results: list) -> list:
    threshold = CONFIG["OCR_MIN_CONFIDENCE"]
    filtered = [
        (bbox, text.strip(), conf)
        for bbox, text, conf in ocr_results
        if conf >= threshold and text.strip()
    ]
    dropped = len(ocr_results) - len(filtered)
    if dropped:
        log.debug(f"  Confidence filter: dropped {dropped} token(s) below {threshold}.")
    return filtered


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 12 — OCR TEXT CLEANING  (unchanged from v3.0)
# ══════════════════════════════════════════════════════════════════════════════

_LABEL_PREFIX_RE = re.compile(
    r"^(Mob|Ph|Phone|Tel|Email|E-mail|Add|Address|Web|Website|"
    r"URL|M|E|A|W|P|T)\s*[:\-]\s*",
    flags=re.IGNORECASE,
)


def clean_token(text: str) -> str:
    text = _LABEL_PREFIX_RE.sub("", text).strip()
    return correct_text(text)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 13 — FIELD EXTRACTORS
# ══════════════════════════════════════════════════════════════════════════════


def _card_height_px(ocr_results: list) -> int:
    if not ocr_results:
        return 1
    return max(int(bbox[2][1]) for bbox, _, _ in ocr_results) or 1


def _fuzzy_contains(text: str, keyword_set: frozenset, cutoff: int = None) -> bool:
    if cutoff is None:
        cutoff = CONFIG["FUZZY_SCORE_CUTOFF"]
    for word in text.lower().split():
        if fuzz_process.extractOne(
            word, keyword_set, scorer=fuzz.ratio, score_cutoff=cutoff
        ):
            return True
    return False


# ── 13a  NAME ─────────────────────────────────────────────────────────────────
# FIX-1 impact: After reconstruct_tokens(), "ADITYA V. VARMA" now arrives as a
# single token so is_plausible_name() evaluates the full name, not "V .".
#
# NAME_ZONE_RATIO raised from 0.35 → 0.65 because centred-layout cards
# (name in the middle) were excluded by the old top-35% constraint.


def extract_name(ocr_results: list, card_height: int) -> tuple:
    """
    Extracts person name via spatial zone + linguistic filters.
    See v3.0 for full algorithm description.

    Changes in v4.0:
      • NAME_ZONE_RATIO raised to 0.65 (centred cards supported)
      • NAME_MAX_WORDS raised to 6
      • Accepts reconstructed spaced-letter tokens from reconstruct_tokens()
    """
    name_zone_y = card_height * CONFIG["NAME_ZONE_RATIO"]

    def is_plausible_name(text: str) -> bool:
        words = text.strip().split()
        if not (CONFIG["NAME_MIN_WORDS"] <= len(words) <= CONFIG["NAME_MAX_WORDS"]):
            return False
        # Allow letters, spaces, hyphens, apostrophes, dots (initials like V.)
        if not re.fullmatch(r"[A-Za-z\s\.\-\']+", text.strip()):
            return False
        text_lower = text.lower()
        words_lower = text_lower.split()
        if any(kw in words_lower for kw in JOB_TITLE_KEYWORDS):
            return False
        if any(kw in words_lower for kw in COMPANY_KEYWORDS):
            return False
        if any(kw in text_lower for kw in ADDRESS_KEYWORDS):
            return False
        if re.search(r"\.(com|in|org|net|co\.in)", text_lower):
            return False
        return True

    buckets: dict = {
        "zone_caps": [],
        "zone_title": [],
        "full_caps": [],
        "full_title": [],
    }

    for bbox, text, conf in ocr_results:
        text = text.strip()
        if not is_plausible_name(text):
            continue
        top_y = bbox[0][1]
        in_zone = top_y <= name_zone_y
        words = text.split()
        is_caps = text.isupper()
        is_title = all(w[0].isupper() for w in words if len(w) > 1)

        if in_zone and is_caps:
            buckets["zone_caps"].append((text, conf))
        elif in_zone and is_title:
            buckets["zone_title"].append((text, conf))
        elif is_caps:
            buckets["full_caps"].append((text, conf))
        elif is_title:
            buckets["full_title"].append((text, conf))

    for tier_name, tier in buckets.items():
        if tier:
            best_text, best_conf = max(tier, key=lambda x: x[1])
            log.debug(f"  Name via [{tier_name}]: '{best_text}' ({best_conf:.2f})")
            return best_text, best_conf

    return "Unknown", 0.0


# ── 13b  EMAIL ────────────────────────────────────────────────────────────────
# FIX-2 impact: extract_email() now receives line_strings (assembled lines)
# in addition to individual tokens, so multi-token email addresses like
# "aditya.v @ urbanedgeinfra . co . in" are detected correctly.


def extract_email(ocr_results: list, line_strings: list = None) -> tuple:
    """
    Extracts email address.

    v4.0 change: accepts optional `line_strings` (from build_line_strings()).
    Line-joined strings are searched first; individual tokens serve as fallback.
    Both paths strip ALL internal whitespace before matching.
    """
    EMAIL_RE = re.compile(
        r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}",
        re.IGNORECASE,
    )
    best_email, best_conf = "N/A", 0.0

    # Primary: search line-joined strings (catches multi-token emails)
    sources = []
    if line_strings:
        for ls in line_strings:
            sources.append((ls["nospace"], ls["conf"]))

    # Fallback / supplemental: individual tokens (catches single-token emails)
    for _, text, conf in ocr_results:
        sources.append((re.sub(r"\s+", "", text), conf))

    for compact, conf in sources:
        match = EMAIL_RE.search(compact)
        if match and conf > best_conf:
            best_email = match.group().lower()
            best_conf = conf
            log.debug(f"  Email found: '{best_email}' ({conf:.2f})")

    return best_email, best_conf


# ── 13c  PHONE  (unchanged from v3.0) ────────────────────────────────────────


def extract_phone(ocr_results: list) -> tuple:
    best_phone, best_conf = "N/A", 0.0
    for _, text, conf in ocr_results:
        digit_count = len(re.sub(r"\D", "", text))
        if digit_count < 7:
            continue
        parsed = None
        for attempt in [text, re.sub(r"[^\d\+]", "", text)]:
            try:
                parsed = phonenumbers.parse(attempt, CONFIG["DEFAULT_REGION"])
                break
            except NumberParseException:
                continue
        if parsed and phonenumbers.is_valid_number(parsed) and conf > best_conf:
            best_phone = phonenumbers.format_number(parsed, PhoneNumberFormat.E164)
            best_conf = conf
            log.debug(f"  Phone: '{text}' → '{best_phone}' ({conf:.2f})")
    return best_phone, best_conf


# ── 13d  JOB TITLE  (unchanged from v3.0, but benefits from FIX-3) ───────────
# FIX-3 impact: "FOUND ER" is now a single token "FOUNDER" before this
# function is called, so the fuzzy keyword match succeeds.


def extract_job_title(ocr_results: list) -> tuple:
    best_title, best_conf = "N/A", 0.0
    for _, text, conf in ocr_results:
        if "@" in text or re.search(r"\d{4,}", text):
            continue
        if _fuzzy_contains(text, JOB_TITLE_KEYWORDS) and conf > best_conf:
            best_title, best_conf = text.strip(), conf
    return best_title, best_conf


# ── 13e  COMPANY ──────────────────────────────────────────────────────────────
# FIX-5: Multi-token company-name assembly
# ─────────────────────────────────────────
# Problem: "URBANEDGE INFRA SOLUTIONS LLP" may arrive as three tokens:
#   "URBANEDGE INFRA"  |  "SOLUTIONS"  |  "LLP"
# The old extractor only matched the token containing "solutions" or "llp",
# not the full company name.
#
# Fix: once a company-keyword-bearing token is identified, walk backwards and
# forwards along the same line (or adjacent lines in the top zone) to collect
# neighbouring tokens and assemble the full company name.


def extract_company(ocr_results: list, name: str, email: str, job_title: str) -> tuple:
    """
    Extracts the company name.

    v4.0 additions:
      FIX-5: After finding the anchor token (contains company keyword), the
      function assembles adjacent same-line tokens to build the full name.
    """
    email_domain = ""
    if email != "N/A" and "@" in email:
        email_domain = email.split("@")[1].split(".")[0].lower()

    # Sort by reading order for adjacency walks
    sorted_r = sorted(
        ocr_results, key=lambda r: (_bbox_mid_y(r[0]), _bbox_left_x(r[0]))
    )

    best_company, best_conf = "N/A", 0.0

    for anchor_idx, (bbox, text, conf) in enumerate(sorted_r):
        text = text.strip()
        if not text:
            continue
        if "@" in text:
            continue
        if re.search(r"\d{4,}", text):
            continue
        if text.lower() == name.lower():
            continue
        if job_title != "N/A" and text.lower() == job_title.lower():
            continue
        if any(kw in text.lower() for kw in ADDRESS_KEYWORDS):
            continue
        if any(city in text.lower() for city in KNOWN_CITIES):
            continue

        words = set(text.lower().split())
        if not words.intersection(COMPANY_KEYWORDS):
            continue

        # ── FIX-5: collect the full company name from adjacent tokens ────
        # Walk left (backwards) on the same line to collect prefix tokens
        left_parts = []
        for k in range(anchor_idx - 1, -1, -1):
            bb_k, tx_k, cf_k = sorted_r[k]
            if not _on_same_line(bbox, bb_k):
                break
            tx_k = tx_k.strip()
            if not tx_k or "@" in tx_k or re.search(r"\d{4,}", tx_k):
                break
            left_parts.insert(0, tx_k)

        # Walk right (forwards) on the same line to collect suffix tokens
        right_parts = []
        for k in range(anchor_idx + 1, len(sorted_r)):
            bb_k, tx_k, cf_k = sorted_r[k]
            if not _on_same_line(bbox, bb_k):
                break
            tx_k = tx_k.strip()
            if not tx_k or "@" in tx_k or re.search(r"\d{4,}", tx_k):
                break
            right_parts.append(tx_k)

        full_company = " ".join(left_parts + [text] + right_parts).strip()

        # Email domain confidence boost
        adjusted_conf = conf
        if (
            email_domain
            and fuzz.partial_ratio(email_domain, full_company.lower()) >= 75
        ):
            adjusted_conf = min(1.0, conf + 0.10)
            log.debug(
                f"  Company confidence boosted (email domain match): '{full_company}'"
            )

        if adjusted_conf > best_conf:
            best_company, best_conf = full_company, adjusted_conf
            log.debug(f"  Company assembled: '{full_company}' ({adjusted_conf:.2f})")

    # Second pass: proper-noun fallback (unchanged from v3.0)
    if best_company == "N/A":
        name_seen = False
        for _, text, conf in sorted_r:
            text = text.strip()
            if text == name:
                name_seen = True
                continue
            if name_seen:
                if (
                    not re.search(r"\d", text)
                    and not _fuzzy_contains(text, JOB_TITLE_KEYWORDS)
                    and not any(kw in text.lower() for kw in ADDRESS_KEYWORDS)
                    and 1 <= len(text.split()) <= 4
                    and all(w[0].isupper() for w in text.split() if w)
                ):
                    best_company, best_conf = text, conf
                    log.debug(f"  Company via second-pass proper noun: '{text}'")
                    break

    return best_company, best_conf


# ── 13f  ADDRESS ──────────────────────────────────────────────────────────────
# FIX-4 impact: by the time this function runs, city tokens like "Surt"/"Sure"
# have already been corrected to "Surat" by correct_city_near_pin().
# No logic change needed in this function.


def extract_address(ocr_results: list) -> tuple:
    PINCODE_RE = re.compile(r"\d{6}")
    GLUED_RE = re.compile(r"([A-Za-z])(\d{6})")
    seen = set()
    fragments = []
    confs = []

    for _, text, conf in ocr_results:
        t = text.strip()
        if not t:
            continue
        t_lower = t.lower()

        has_addr_kw = any(kw in t_lower for kw in ADDRESS_KEYWORDS)
        has_pincode = bool(PINCODE_RE.search(t))
        has_city = bool(
            fuzz_process.extractOne(
                t_lower,
                KNOWN_CITIES,
                scorer=fuzz.partial_ratio,
                score_cutoff=CONFIG["FUZZY_SCORE_CUTOFF"],
            )
        )

        if not (has_addr_kw or has_pincode or has_city):
            continue

        cleaned = GLUED_RE.sub(r"\1 \2", t)
        if cleaned.lower() in seen:
            continue

        seen.add(cleaned.lower())
        fragments.append(cleaned)
        confs.append(conf)

    if not fragments:
        return "N/A", 0.0

    return " | ".join(fragments), float(np.mean(confs))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 14 — EXTRACTION ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════


def _quality_label(
    name: str, phone: str, email: str, company: str, address: str
) -> str:
    """
    FIX-6: quality label now uses the same N/A sentinel set as before but
    no longer misclassifies records where the email is present but had
    zero confidence (which only happened because of FIX-2 failures).
    With FIX-2 in place, email extraction works so this function is correct.
    Kept unchanged; improvement is automatic.
    """
    SENTINEL = {"N/A", "Unknown", ""}
    filled = sum(
        v.strip() not in SENTINEL for v in [name, phone, email, company, address]
    )
    if filled >= 5:
        return "GREEN"
    if filled >= 3:
        return "YELLOW"
    return "RED"


def process_card(ocr_raw: list) -> dict:
    """
    Orchestrates all field extractors for one card.

    v4.0 pipeline order:
      1.  Confidence filter
      2.  Token reconstruction   ← NEW: FIX-1 + FIX-3
      3.  PIN↔city correction    ← NEW: FIX-4
      4.  Clean tokens
      5.  Build line strings     ← NEW: FIX-2
      6.  Email (uses line strings)
      7.  Phone
      8.  Name
      9.  Job title
      10. Company (uses FIX-5 adjacency walk)
      11. Address
      12. Quality label
    """
    EMPTY = {
        "Name": "Unknown",
        "Phone": "N/A",
        "Email": "N/A",
        "Company": "N/A",
        "Address": "N/A",
        "Job_Title": "N/A",
        "Quality": "RED",
        "Name_Conf": 0.0,
        "Phone_Conf": 0.0,
        "Email_Conf": 0.0,
    }

    # Step 1 — Confidence filter
    filtered = filter_by_confidence(ocr_raw)
    if not filtered:
        log.warning("  All OCR tokens below confidence threshold — card skipped.")
        return EMPTY

    # Step 2 — Token reconstruction (FIX-1 spaced-letter + FIX-3 split-word)
    reconstructed = reconstruct_tokens(filtered)

    # Step 3 — PIN↔city contextual correction (FIX-4)
    pin_corrected = correct_city_near_pin(reconstructed)

    # Step 4 — Clean tokens (label stripping + spell correction)
    cleaned_results = []
    for bbox, text, conf in pin_corrected:
        cleaned = clean_token(text)
        if cleaned:
            cleaned_results.append((bbox, cleaned, conf))

    if not cleaned_results:
        log.warning("  No text remained after cleaning — card skipped.")
        return EMPTY

    # Step 5 — Build line strings (FIX-2)
    line_strings = build_line_strings(cleaned_results)

    card_h = _card_height_px(cleaned_results)
    log.debug(
        f"  Card height: {card_h}px | "
        f"Name zone ≤ {card_h * CONFIG['NAME_ZONE_RATIO']:.0f}px"
    )

    # Steps 6–11 — Field extraction
    email, email_conf = extract_email(cleaned_results, line_strings)
    phone, phone_conf = extract_phone(cleaned_results)
    name, name_conf = extract_name(cleaned_results, card_h)
    job_title, _ = extract_job_title(cleaned_results)
    company, _ = extract_company(cleaned_results, name, email, job_title)
    address, _ = extract_address(cleaned_results)

    # Step 12 — Quality label
    quality = _quality_label(name, phone, email, company, address)

    log.info(
        f"  Name: {name!r:<22} Phone: {phone:<18} "
        f"Email: {email:<35} Company: {company!r:<28} Quality: {quality}"
    )

    return {
        "Name": name,
        "Phone": phone,
        "Email": email,
        "Company": company,
        "Address": address,
        "Job_Title": job_title,
        "Quality": quality,
        "Name_Conf": round(name_conf, 2),
        "Phone_Conf": round(phone_conf, 2),
        "Email_Conf": round(email_conf, 2),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 15 — POPPLER AUTO-DETECTION  (unchanged from v3.0)
# ══════════════════════════════════════════════════════════════════════════════


def resolve_poppler_path():
    if CONFIG["POPPLER_PATH"]:
        return CONFIG["POPPLER_PATH"]
    windows_candidates = [
        r"C:\poppler\Library\bin",
        r"C:\Program Files\poppler\Library\bin",
        r"C:\Program Files (x86)\poppler\bin",
        r"C:\poppler\bin",
    ]
    for path in windows_candidates:
        if os.path.isdir(path):
            log.info(f"  Poppler auto-detected: {path}")
            return path
    return None


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 16 — EXCEL OUTPUT  (unchanged from v3.0)
# ══════════════════════════════════════════════════════════════════════════════


def save_to_excel(records: list, output_path: str) -> None:
    COLUMN_ORDER = [
        "Filename",
        "Card",
        "Name",
        "Phone",
        "Email",
        "Company",
        "Address",
        "Job_Title",
        "Quality",
        "Name_Conf",
        "Phone_Conf",
        "Email_Conf",
    ]
    df = pd.DataFrame(records)[COLUMN_ORDER]

    QUALITY_FILLS = {
        "GREEN": PatternFill("solid", fgColor="C6EFCE"),
        "YELLOW": PatternFill("solid", fgColor="FFEB9C"),
        "RED": PatternFill("solid", fgColor="FFC7CE"),
    }
    HEADER_FILL = PatternFill("solid", fgColor="1F4E79")
    HEADER_FONT = Font(bold=True, color="FFFFFF", size=11)
    HEADER_ALIGN = Alignment(horizontal="center", vertical="center")

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Visiting Cards")
        ws = writer.sheets["Visiting Cards"]

        for cell in ws[1]:
            cell.fill = HEADER_FILL
            cell.font = HEADER_FONT
            cell.alignment = HEADER_ALIGN

        quality_col_idx = COLUMN_ORDER.index("Quality") + 1
        for row_idx, quality_val in enumerate(df["Quality"], start=2):
            fill = QUALITY_FILLS.get(str(quality_val), PatternFill())
            ws.cell(row=row_idx, column=quality_col_idx).fill = fill

        for col_cells in ws.columns:
            max_len = max(
                (len(str(c.value)) for c in col_cells if c.value is not None),
                default=10,
            )
            ws.column_dimensions[get_column_letter(col_cells[0].column)].width = min(
                max_len + 4, 60
            )
        ws.freeze_panes = "A2"

    log.info(f"  Excel saved → {os.path.abspath(output_path)}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 17 — SINGLE CARD PROCESSOR  (unchanged from v3.0)
# ══════════════════════════════════════════════════════════════════════════════


def process_single_card(card_img: np.ndarray, reader: easyocr.Reader, card_label: str):
    try:
        processed = preprocess_image(card_img)
        ocr_raw = reader.readtext(processed, width_ths=CONFIG["OCR_WIDTH_THRESHOLD"])

        log.debug(f"  Raw OCR: {len(ocr_raw)} token(s)")
        for bbox, text, conf in ocr_raw:
            log.debug(f"    [{conf:.2f}] {text!r}")

        if not filter_by_confidence(ocr_raw):
            log.warning(
                f"  [{card_label}] No confident tokens — retrying with contrast-boost."
            )
            gray = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
            fallback = cv2.convertScaleAbs(
                gray, alpha=CONFIG["CONTRAST_ALPHA"], beta=CONFIG["CONTRAST_BETA"]
            )
            ocr_raw = reader.readtext(fallback, width_ths=CONFIG["OCR_WIDTH_THRESHOLD"])

        return process_card(ocr_raw)

    except Exception as exc:
        log.error(f"  [{card_label}] Processing exception: {exc}")
        log.debug(traceback.format_exc())
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 18 — MAIN  (unchanged from v3.0)
# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    log.info("=" * 70)
    log.info("  VISITING CARD OCR EXTRACTOR — ULTIMATE BUILD v4.0")
    log.info("=" * 70)

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        torch.backends.cudnn.benchmark = True
        log.info("  GPU: ENABLED (cuDNN benchmark active)")
    else:
        log.info("  GPU: DISABLED — CPU mode")

    root = Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(
        title="Select Visiting Card Files",
        filetypes=[("Supported files", "*.jpg *.jpeg *.png *.pdf *.tiff *.tif *.bmp")],
    )
    root.destroy()

    if not file_paths:
        log.error("No files selected. Exiting.")
        return

    log.info(f"  {len(file_paths)} file(s) selected.")
    log.info("  Loading EasyOCR model...")
    reader = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)
    log.info("  EasyOCR ready.")

    poppler_path = resolve_poppler_path()
    all_results = []
    total_cards = 0
    failed_cards = 0

    for file_path in file_paths:
        filename = os.path.basename(file_path)
        log.info(f"\n{'─' * 60}")
        log.info(f"  File: {filename}")

        try:
            ext = Path(file_path).suffix.lower()
            if ext == ".pdf":
                pdf_kwargs = {"dpi": CONFIG["PDF_DPI"]}
                if poppler_path:
                    pdf_kwargs["poppler_path"] = poppler_path
                pages = convert_from_path(file_path, **pdf_kwargs)
                images = [cv2.cvtColor(np.array(p), cv2.COLOR_RGB2BGR) for p in pages]
                log.info(f"  PDF: {len(images)} page(s).")
            else:
                raw = cv2.imread(file_path)
                if raw is None:
                    raise IOError(f"OpenCV could not read '{file_path}'.")
                images = [raw]

        except Exception as exc:
            log.error(f"  SKIPPED — Failed to load '{filename}': {exc}")
            all_results.append(
                {
                    "Filename": filename,
                    "Card": "N/A",
                    "Name": "FILE LOAD ERROR",
                    "Phone": "N/A",
                    "Email": "N/A",
                    "Company": "N/A",
                    "Address": "N/A",
                    "Job_Title": "N/A",
                    "Quality": "RED",
                    "Name_Conf": 0.0,
                    "Phone_Conf": 0.0,
                    "Email_Conf": 0.0,
                }
            )
            continue

        for page_idx, full_page in enumerate(images):
            if full_page is None:
                continue

            try:
                card_crops = segment_cards(full_page)
            except Exception as exc:
                log.error(f"  Segmentation failed on page {page_idx + 1}: {exc}")
                card_crops = [full_page]

            for card_idx, card_crop in enumerate(card_crops):
                total_cards += 1
                card_label = f"P{page_idx + 1}_C{card_idx + 1}"
                log.info(f"  Card [{card_label}]")

                result = process_single_card(card_crop, reader, card_label)

                if result is None:
                    failed_cards += 1
                    all_results.append(
                        {
                            "Filename": filename,
                            "Card": card_label,
                            "Name": "PROCESSING ERROR",
                            "Phone": "N/A",
                            "Email": "N/A",
                            "Company": "N/A",
                            "Address": "N/A",
                            "Job_Title": "N/A",
                            "Quality": "RED",
                            "Name_Conf": 0.0,
                            "Phone_Conf": 0.0,
                            "Email_Conf": 0.0,
                        }
                    )
                else:
                    all_results.append(
                        {
                            "Filename": filename,
                            "Card": card_label,
                            "Name": result["Name"],
                            "Phone": result["Phone"],
                            "Email": result["Email"],
                            "Company": result["Company"],
                            "Address": result["Address"],
                            "Job_Title": result["Job_Title"],
                            "Quality": result["Quality"],
                            "Name_Conf": result["Name_Conf"],
                            "Phone_Conf": result["Phone_Conf"],
                            "Email_Conf": result["Email_Conf"],
                        }
                    )

    if not all_results:
        log.error("No data extracted. Nothing to save.")
        return

    save_to_excel(all_results, CONFIG["OUTPUT_FILENAME"])

    green = sum(r["Quality"] == "GREEN" for r in all_results)
    yellow = sum(r["Quality"] == "YELLOW" for r in all_results)
    red = sum(r["Quality"] == "RED" for r in all_results)

    log.info(f"\n{'=' * 70}")
    log.info(f"  COMPLETE — {total_cards} card(s) processed, {failed_cards} failed.")
    log.info(f"  Quality → GREEN: {green} | YELLOW: {yellow} | RED: {red}")
    log.info(f"  Output  : {os.path.abspath(CONFIG['OUTPUT_FILENAME'])}")
    log.info(f"  Log     : {os.path.abspath(CONFIG['LOG_FILENAME'])}")
    log.info(f"{'=' * 70}")


if __name__ == "__main__":
    main()
