# import os
# from pathlib import Path
# import easyocr
# import cv2
# import pandas as pd

# # ===== CONFIG =====
# INPUT_FOLDER = "images"  # put images here
# OUTPUT_FOLDER = "output"

# os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# # ===== INIT OCR =====
# reader = easyocr.Reader(["en"], gpu=False)

# # ===== GET IMAGES =====
# VALID_EXT = (".jpg", ".jpeg", ".png")

# image_paths = []
# for file in os.listdir(INPUT_FOLDER):
#     if file.lower().endswith(VALID_EXT):
#         image_paths.append(os.path.join(INPUT_FOLDER, file))

# print(f"Found {len(image_paths)} images")

# # ===== PROCESS =====
# all_rows = []

# for img_path in image_paths:
#     print(f"Processing: {img_path}")

#     img = cv2.imread(img_path)
#     if img is None:
#         print("❌ Could not read image")
#         continue

#     results = reader.readtext(img)

#     for bbox, text, conf in results:
#         xs = [pt[0] for pt in bbox]
#         ys = [pt[1] for pt in bbox]

#         xmin, xmax = int(min(xs)), int(max(xs))
#         ymin, ymax = int(min(ys)), int(max(ys))

#         all_rows.append(
#             {
#                 "filename": os.path.basename(img_path),
#                 "text": text,
#                 "confidence": conf,
#                 "xmin": xmin,
#                 "ymin": ymin,
#                 "xmax": xmax,
#                 "ymax": ymax,
#             }
#         )

#         # draw box
#         cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
#         cv2.putText(
#             img, text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
#         )

#     # save annotated image
#     out_img_path = os.path.join(
#         OUTPUT_FOLDER, f"annotated_{os.path.basename(img_path)}"
#     )
#     cv2.imwrite(out_img_path, img)

# # ===== SAVE EXCEL =====
# df = pd.DataFrame(all_rows)
# excel_path = os.path.join(OUTPUT_FOLDER, "ocr_results.xlsx")
# df.to_excel(excel_path, index=False)

# print("✅ OCR Completed")
# print(f"Saved results to: {excel_path}")

# import os
# import re
# import cv2
# import easyocr
# import pandas as pd
# from tkinter import Tk, filedialog


# # ===== FILE PICKER =====
# def select_images():
#     root = Tk()
#     root.withdraw()
#     file_paths = filedialog.askopenfilenames(
#         title="Select Visiting Card Image(s)",
#         filetypes=[("Image files", "*.jpg *.jpeg *.png")],
#     )
#     return list(file_paths)


# # ===== EXTRACTION FUNCTION =====
# def extract_info(texts):
#     email = ""
#     phone = ""
#     name = ""
#     company = ""

#     for t in texts:
#         if "@" in t and "." in t:
#             email = t.strip()

#         digits = "".join(filter(str.isdigit, t))
#         if 10 <= len(digits) <= 13:
#             phone = digits

#     for t in texts:
#         if (
#             t != email
#             and len(t.split()) <= 3
#             and not any(char.isdigit() for char in t)
#             and "@" not in t
#         ):
#             name = t
#             break

#     for t in texts:
#         if t not in [name, email] and len(t) > 3:
#             company = t
#             break

#     return name, phone, email, company


# # ===== MAIN =====
# def main():
#     print("📂 Select image(s)...")

#     image_paths = select_images()

#     if not image_paths:
#         print("❌ No image selected")
#         return

#     if len(image_paths) == 1:
#         print("🟢 Single image selected")
#     else:
#         print(f"🟢 {len(image_paths)} images selected")

#     os.makedirs("output", exist_ok=True)

#     reader = easyocr.Reader(["en"], gpu=False)

#     all_rows = []

#     for image_path in image_paths:
#         print(f"\nProcessing: {image_path}")

#         img = cv2.imread(image_path)
#         if img is None:
#             continue

#         results = reader.readtext(img)

#         texts = []

#         for bbox, text, conf in results:
#             texts.append(text)

#             xs = [pt[0] for pt in bbox]
#             ys = [pt[1] for pt in bbox]

#             xmin, xmax = int(min(xs)), int(max(xs))
#             ymin, ymax = int(min(ys)), int(max(ys))

#             cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
#             cv2.putText(
#                 img,
#                 text,
#                 (xmin, ymin - 10),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 0.6,
#                 (0, 255, 0),
#                 2,
#             )

#         print("🔍 OCR Text:", texts)

#         name, phone, email, company = extract_info(texts)

#         all_rows.append(
#             {
#                 "filename": os.path.basename(image_path),
#                 "name": name,
#                 "phone": phone,
#                 "email": email,
#                 "company": company,
#             }
#         )

#         out_img = os.path.join("output", f"annotated_{os.path.basename(image_path)}")
#         cv2.imwrite(out_img, img)

#     df = pd.DataFrame(all_rows)
#     df.to_excel("output/structured_results.xlsx", index=False)

#     print("\n✅ Done!")


# if __name__ == "__main__":
#     main()


import os
import re
import cv2
import easyocr
import pandas as pd
import numpy as np
from tkinter import Tk, filedialog
from pdf2image import convert_from_path

print("🚀 UNIVERSAL OCR PIPELINE STARTING...")


# ===== DYNAMIC SEGMENTATION LOGIC =====
def segment_cards_dynamically(image):
    """
    Identifies the 'valleys' of white space between cards to split them.
    This logic adapts to any number of cards on a page.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Invert and threshold: making text/graphics white and background black
    _, binary = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)

    # Sum pixels across rows to find horizontal density
    horizontal_sum = np.sum(binary, axis=1)

    height, width = gray.shape
    # Threshold for noise (ignore rows with very few active pixels)
    noise_threshold = width * 0.02

    card_bounds = []
    start_y = None

    for y in range(height):
        if horizontal_sum[y] > noise_threshold and start_y is None:
            start_y = y  # Top of a card detected
        elif horizontal_sum[y] <= noise_threshold and start_y is not None:
            if y - start_y > 150:  # Minimum height for a valid card
                card_bounds.append((start_y, y))
            start_y = None

    # Catch last card if it touches the bottom
    if start_y is not None and (height - start_y > 150):
        card_bounds.append((start_y, height))

    # Split the original image based on these boundaries
    crops = []
    for y1, y2 in card_bounds:
        # Add 15px padding for better OCR context
        pad = 15
        crop = image[max(0, y1 - pad) : min(height, y2 + pad), :]
        crops.append(crop)

    return crops if crops else [image]


# ===== ROBUST DATA EXTRACTION =====
def extract_card_details(ocr_results):
    """
    Maps text to fields based on patterns rather than fixed line numbers.
    """
    # Sort OCR by Y-position (top to bottom)
    texts = [res[1].strip() for res in sorted(ocr_results, key=lambda x: x[0][0][1])]

    # Clean common labels
    clean_texts = [
        re.sub(r"^(Mob|Ph|Email|Add|Web|M|E|A|W):", "", t, flags=re.I).strip()
        for t in texts
    ]

    data = {
        "name": "Unknown",
        "phone": "N/A",
        "email": "N/A",
        "company": "N/A",
        "address": [],
    }

    for t in clean_texts:
        # 1. Email Check
        if "@" in t and "." in t:
            data["email"] = t.lower()
            continue

        # 2. Phone Check
        if re.search(r"\d{10,}", t.replace(" ", "").replace("-", "")):
            data["phone"] = t
            continue

        # 3. Company Keywords
        company_keywords = [
            "ltd",
            "inc",
            "solutions",
            "consulting",
            "studio",
            "collective",
            "tech",
            "group",
        ]
        if any(k in t.lower() for k in company_keywords):
            data["company"] = t
            continue

        # 4. Address Keywords or Pincodes
        addr_keywords = [
            "road",
            "st.",
            "street",
            "complex",
            "floor",
            "level",
            "nagar",
            "city",
            "colony",
            "block",
        ]
        if any(k in t.lower() for k in addr_keywords) or re.search(r"\d{6}", t):
            data["address"].append(t)
            continue

        # 5. Name Logic (First short line that isn't empty and has no digits)
        if (
            data["name"] == "Unknown"
            and 2 <= len(t.split()) <= 4
            and not re.search(r"\d", t)
        ):
            data["name"] = t

    return (
        data["name"],
        data["phone"],
        data["email"],
        data["company"],
        " | ".join(data["address"]),
    )


# ===== MAIN EXECUTION =====
def main():
    root = Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(title="Select Visiting Card PDF/Images")

    if not file_paths:
        print("❌ No files selected.")
        return

    os.makedirs("output_debug", exist_ok=True)
    reader = easyocr.Reader(["en"], gpu=False)
    results_list = []

    for path in file_paths:
        print(f"📁 Processing: {os.path.basename(path)}")

        # 1. Convert/Load Input
        if path.lower().endswith(".pdf"):
            # Path to poppler is required for Windows
            pages = convert_from_path(path, poppler_path=r"C:\poppler\Library\bin")
            images = [cv2.cvtColor(np.array(p), cv2.COLOR_RGB2BGR) for p in pages]
        else:
            images = [cv2.imread(path)]

        for p_idx, full_img in enumerate(images):
            # 2. SEGMENTATION (The fix for your error)
            # This cuts the page into 1, 2, or 3+ cards automatically
            card_images = segment_cards_dynamically(full_img)
            print(f"   Detected {len(card_images)} card(s) on Page {p_idx+1}")

            for c_idx, card_img in enumerate(card_images):
                # 3. OCR on isolated card
                ocr_output = reader.readtext(card_img)

                # 4. Logic-based extraction
                name, ph, mail, comp, addr = extract_card_details(ocr_output)

                results_list.append(
                    {
                        "Filename": os.path.basename(path),
                        "Card_Position": f"P{p_idx+1}_C{c_idx+1}",
                        "Name": name,
                        "Phone": ph,
                        "Email": mail,
                        "Company": comp,
                        "Address": addr,
                    }
                )

                # Save crop for debugging
                cv2.imwrite(f"output_debug/card_{p_idx}_{c_idx}.jpg", card_img)

    # 5. Save to Excel
    df = pd.DataFrame(results_list)
    df.to_excel("Final_Visiting_Card_Data.xlsx", index=False)
    print("\n✅ PROCESS COMPLETE. Data saved to 'Final_Visiting_Card_Data.xlsx'")


if __name__ == "__main__":
    main()
