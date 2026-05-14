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
