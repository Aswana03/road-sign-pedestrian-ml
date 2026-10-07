import pandas as pd
from pathlib import Path
from PIL import Image

# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

CSV_FILE = PROJECT_DIR / "train_selected.csv"
IMAGE_DIR = PROJECT_DIR / "images" / "train"

OUTPUT_DIR = PROJECT_DIR / "dataset" / "crops"
OUTPUT_CSV = PROJECT_DIR / "cropped_objects.csv"

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD SELECTED DATASET
# ============================================================

df = pd.read_csv(CSV_FILE)

print("=" * 60)
print("BDD100K OBJECT CROPPING")
print("=" * 60)

print(f"Selected objects : {len(df)}")
print(f"Image directory  : {IMAGE_DIR}")
print(f"Output directory : {OUTPUT_DIR}")

# ============================================================
# CROPPING
# ============================================================

cropped_records = []
successful = 0
failed = 0

for index, row in df.iterrows():

    image_name = row["image"]

    image_path = IMAGE_DIR / image_name

    # --------------------------------------------------------
    # Check image
    # --------------------------------------------------------

    if not image_path.exists():
        print(f"[ERROR] Image not found: {image_name}")
        failed += 1
        continue

    try:

        # ----------------------------------------------------
        # Open image
        # ----------------------------------------------------

        image = Image.open(image_path).convert("RGB")

        width, height = image.size

        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        x1 = int(row["x1"])
        y1 = int(row["y1"])
        x2 = int(row["x2"])
        y2 = int(row["y2"])

        # ----------------------------------------------------
        # Make sure coordinates stay inside image
        # ----------------------------------------------------

        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))

        # Invalid bounding box
        if x2 <= x1 or y2 <= y1:
            print(f"[ERROR] Invalid bounding box: {image_name}")
            failed += 1
            continue

        # ----------------------------------------------------
        # Crop object
        # ----------------------------------------------------

        crop = image.crop((x1, y1, x2, y2))

        # ----------------------------------------------------
        # Generate unique filename
        # ----------------------------------------------------

        category = str(row["category"]).replace(" ", "_")

        crop_name = f"{index:05d}_{category}_{image_name}"

        crop_path = OUTPUT_DIR / crop_name

        crop.save(crop_path, quality=95)

        # ----------------------------------------------------
        # Save metadata
        # ----------------------------------------------------

        record = row.to_dict()

        record["crop_filename"] = crop_name
        record["crop_path"] = str(crop_path.relative_to(PROJECT_DIR))
        record["crop_width"] = crop.width
        record["crop_height"] = crop.height

        cropped_records.append(record)

        successful += 1

        if successful % 100 == 0:
            print(
                f"Processed: {successful}/{len(df)}"
            )

    except Exception as e:

        print(f"[ERROR] {image_name}: {e}")
        failed += 1


# ============================================================
# SAVE CROPPED DATASET METADATA
# ============================================================

cropped_df = pd.DataFrame(cropped_records)

cropped_df.to_csv(
    OUTPUT_CSV,
    index=False
)

# ============================================================
# SUMMARY
# ============================================================

print("\n")
print("=" * 60)
print("CROPPING COMPLETE")
print("=" * 60)

print(f"Objects requested : {len(df)}")
print(f"Successfully cropped : {successful}")
print(f"Failed : {failed}")

print("\nClass distribution:")

if len(cropped_df) > 0:
    print(cropped_df["category"].value_counts())

print("\nWeather distribution:")

if len(cropped_df) > 0:
    print(cropped_df["weather"].value_counts())

print("\nOutput:")
print(OUTPUT_DIR)
print(OUTPUT_CSV)

print("=" * 60)