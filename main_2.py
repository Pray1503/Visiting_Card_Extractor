import os
from pathlib import Path
import easyocr
import cv2
import pandas as pd

# ===== CONFIG =====
INPUT_FOLDER = "images"  # put images here
OUTPUT_FOLDER = "output"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ===== INIT OCR =====
reader = easyocr.Reader(["en"], gpu=False)

# ===== GET IMAGES =====
VALID_EXT = (".jpg", ".jpeg", ".png")

image_paths = []
for file in os.listdir(INPUT_FOLDER):
    if file.lower().endswith(VALID_EXT):
        image_paths.append(os.path.join(INPUT_FOLDER, file))

print(f"Found {len(image_paths)} images")

# ===== PROCESS =====
all_rows = []

for img_path in image_paths:
    print(f"Processing: {img_path}")

    img = cv2.imread(img_path)
    if img is None:
        print("❌ Could not read image")
        continue

    results = reader.readtext(img)

    for bbox, text, conf in results:
        xs = [pt[0] for pt in bbox]
        ys = [pt[1] for pt in bbox]

        xmin, xmax = int(min(xs)), int(max(xs))
        ymin, ymax = int(min(ys)), int(max(ys))

        all_rows.append(
            {
                "filename": os.path.basename(img_path),
                "text": text,
                "confidence": conf,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
            }
        )

        # draw box
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(
            img, text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )

    # save annotated image
    out_img_path = os.path.join(
        OUTPUT_FOLDER, f"annotated_{os.path.basename(img_path)}"
    )
    cv2.imwrite(out_img_path, img)

# ===== SAVE EXCEL =====
df = pd.DataFrame(all_rows)
excel_path = os.path.join(OUTPUT_FOLDER, "ocr_results.xlsx")
df.to_excel(excel_path, index=False)

print("✅ OCR Completed")
print(f"Saved results to: {excel_path}")
