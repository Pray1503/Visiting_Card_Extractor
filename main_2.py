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

import os
import re
import cv2
import easyocr
import pandas as pd
from tkinter import Tk, filedialog


# ===== FILE PICKER =====
def select_images():
    root = Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(
        title="Select Visiting Card Image(s)",
        filetypes=[("Image files", "*.jpg *.jpeg *.png")],
    )
    return list(file_paths)


# ===== EXTRACTION FUNCTION =====
def extract_info(texts):
    email = ""
    phone = ""
    name = ""
    company = ""

    for t in texts:
        if "@" in t and "." in t:
            email = t.strip()

        digits = "".join(filter(str.isdigit, t))
        if 10 <= len(digits) <= 13:
            phone = digits

    for t in texts:
        if (
            t != email
            and len(t.split()) <= 3
            and not any(char.isdigit() for char in t)
            and "@" not in t
        ):
            name = t
            break

    for t in texts:
        if t not in [name, email] and len(t) > 3:
            company = t
            break

    return name, phone, email, company


# ===== MAIN =====
def main():
    print("📂 Select image(s)...")

    image_paths = select_images()

    if not image_paths:
        print("❌ No image selected")
        return

    if len(image_paths) == 1:
        print("🟢 Single image selected")
    else:
        print(f"🟢 {len(image_paths)} images selected")

    os.makedirs("output", exist_ok=True)

    reader = easyocr.Reader(["en"], gpu=False)

    all_rows = []

    for image_path in image_paths:
        print(f"\nProcessing: {image_path}")

        img = cv2.imread(image_path)
        if img is None:
            continue

        results = reader.readtext(img)

        texts = []

        for bbox, text, conf in results:
            texts.append(text)

            xs = [pt[0] for pt in bbox]
            ys = [pt[1] for pt in bbox]

            xmin, xmax = int(min(xs)), int(max(xs))
            ymin, ymax = int(min(ys)), int(max(ys))

            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(
                img,
                text,
                (xmin, ymin - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        print("🔍 OCR Text:", texts)

        name, phone, email, company = extract_info(texts)

        all_rows.append(
            {
                "filename": os.path.basename(image_path),
                "name": name,
                "phone": phone,
                "email": email,
                "company": company,
            }
        )

        out_img = os.path.join("output", f"annotated_{os.path.basename(image_path)}")
        cv2.imwrite(out_img, img)

    df = pd.DataFrame(all_rows)
    df.to_excel("output/structured_results.xlsx", index=False)

    print("\n✅ Done!")


if __name__ == "__main__":
    main()
