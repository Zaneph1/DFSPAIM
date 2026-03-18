import os
import pandas as pd
from PIL import Image

# ================= Configuration =================
# Get the directory where the current script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Relative path: points to the image folder
ROOT_FOLDER = os.path.join(BASE_DIR, r"NewQuestionableCases\NewQuestionableCases\NewQuestionableCases_Small")

# Output CSV file name
OUTPUT_CSV = "output_nested_images.csv"

# ==================================================

def main():
    if not os.path.exists(ROOT_FOLDER):
        print(f"❌ Error: Folder not found '{ROOT_FOLDER}'")
        return

    data = []
    # Supported image formats
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')

    print(f"🔍 Scanning folder and subfolders recursively: {ROOT_FOLDER} ...")

    count = 0

    # os.walk automatically traverses all subdirectories
    # root: current directory path
    # dirs: list of subfolders in the current directory
    # files: list of files in the current directory
    for root, dirs, files in os.walk(ROOT_FOLDER):
        for filename in files:
            if filename.lower().endswith(valid_exts):
                file_path = os.path.join(root, filename)

                try:
                    # 1. Get filename without extension as image_name
                    image_name = os.path.splitext(filename)[0]

                    # 2. Open image to get width and height
                    with Image.open(file_path) as img:
                        width, height = img.size

                    # 3. Construct data row
                    row = {
                        "image_name": image_name,
                        "patient_id": 0,
                        "sex": 0,
                        "age_approx": 0,
                        "anatom_site_general_challenge": 0,
                        "diagnosis": "CD",
                        "benign_malignant": 0,
                        "target": 1,
                        "tfrecord": 4996,
                        "width": width,
                        "height": height
                    }

                    data.append(row)
                    count += 1

                    # Optional: print progress every 100 images
                    if count % 100 == 0:
                        print(f"   ...Processed {count} images")

                except Exception as e:
                    print(f"⚠️ Skipping corrupted file {filename} (in {root}): {e}")

    if count == 0:
        print("❌ No images found in the folder and its subfolders.")
        return

    # Create DataFrame
    df = pd.DataFrame(data)

    # Force column order
    columns_order = [
        "image_name", "patient_id", "sex", "age_approx",
        "anatom_site_general_challenge", "diagnosis",
        "benign_malignant", "target", "tfrecord",
        "width", "height"
    ]
    df = df[columns_order]

    # Save CSV
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

    print(f"\n✅ Successfully processed {count} images!")
    print(f"📄 File saved as: {OUTPUT_CSV}")
    print("\nFirst 5 rows preview:")
    print(df.head())


if __name__ == "__main__":
    main()