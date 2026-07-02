import re

CFG = {
    # OCR
    "OCR_CONF_THRESH": 0.18,
    "PDF_DPI": 250,
    # Segmentation
    "SEG_MIN_AREA_FRAC": 0.03,
    "SEG_MAX_AREA_FRAC": 0.95,
    "SEG_PAD_PX": 12,
    "SEG_CARD_ASPECT_MIN": 0.5,
    "SEG_CARD_ASPECT_MAX": 5.0,
    # Token Reconstruction Graph (TRG)
    "TRG_GAP_SIGMA_MULTIPLIER": 2.2,
    "TRG_OVERLAP_MERGE_THRESH": 0.15,
    "TRG_MAX_SKEW_DEG": 3.0,
    # Row grouping
    "ROW_TOL_FACTOR": 0.50,
    "ROW_TOL_FALLBACK_PX": 8,
    # Vertical block proximity
    "BLOCK_PROX_FACTOR": 1.80,
    # Layout Role Classifier (LRC)
    "LRC_TOP_FRAC": 0.35,
    "LRC_BOTTOM_FRAC": 0.60,
    "LRC_TALL_SCALE_RATIO": 1.40,
    "LRC_NAME_MAX_TOKENS": 5,
    "LRC_COMPANY_MAX_TOKENS": 8,
    "LRC_TITLE_MAX_TOKENS": 8,
    # Anchor proximity
    "ANCHOR_PROX_LINES": 3,
    # Phone
    "PHONE_MIN_DIGITS": 7,
    "PHONE_MAX_DIGITS": 15,
    "MAX_PHONES": 5,
    # Fuzzy dedup
    "FUZZY_SIM_THRESH": 80,
    # Output
    "EXCEL_DB_FILE": "master_contacts.xlsx",
    "DEBUG": False,
    # Script detection
    "SECONDARY_SCRIPT_THRESH": 0.15,
    # IOU dedup
    "IOU_DEDUP_CELL_FACTOR": 0.40,
    "IOU_DEDUP_CELL_MIN_PX": 12,
    # Multi-pass OCR
    "OCR_BRIGHTNESS_ALPHA": 1.35,
    "OCR_BRIGHTNESS_BETA": 25,
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
EXCEL_COLS = [
    "Timestamp",
    "Name",
    "Job Title",
    "Company",
    "Email",
    "Phone_1",
    "Phone_2",
    "Phone_3",
    "Website",
    "Address",
    "LinkedIn",
    "Twitter",
    "Instagram",
    "GitHub",
    "Quality",
    "Confidence",
    "Source",
]

_QUALITY_COLOUR = {"🟢 GREEN": "C6EFCE", "🟡 YELLOW": "FFEB9C", "🔴 RED": "FFC7CE"}

from openpyxl.styles import Border, Side

_THIN_SIDE = Side(style="thin", color="BDD7EE")
_THIN_BORDER = Border(
    left=_THIN_SIDE, right=_THIN_SIDE, top=_THIN_SIDE, bottom=_THIN_SIDE
)

_EMAIL_RE = re.compile(
    r"[A-Za-z0-9._%+\-]{1,64}@[A-Za-z0-9.\-]{1,253}\.[A-Za-z]{2,12}", re.I
)
_WEB_RE = re.compile(
    r"(?:https?://|www\.)[A-Za-z0-9.\-/_%?=&#@]+|[A-Za-z0-9][\w\-]*\.[A-Za-z]{2,12}(?:/[^\s]*)?",
    re.I,
)
_PHONE_DIGITS_RE = re.compile(r"[\+\(]?\d[\d\s\-\.\(\)]{5,16}\d")
_SOCIAL_PREFIX_RE = re.compile(
    r"(?:^|[\s,|•·])(@[\w.]{2,32}|/in/[\w\-]{3,100}|linkedin\.com/in/[\w\-]+|github\.com/[\w\-]+|twitter\.com/[\w]+|x\.com/[\w]+|instagram\.com/[\w.]+)",
    re.I,
)
_AT_HANDLE_RE = re.compile(r"(?<![A-Za-z0-9])@([\w.]{2,32})")
_POSTAL_SHAPE_RE = re.compile(
    r"\b\d{4,7}\b"
    r"|\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b"
    r"|\b\d{3}-\d{4}\b"
    r"|\b[A-Z]\d[A-Z]\s*\d[A-Z]\d\b",
    re.I,
)
_ADDR_SEPARATOR_RE = re.compile(r"[,/\\|]{1}")
