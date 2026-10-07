from pathlib import Path
import pandas as pd
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent

TRAIN_SELECTED = PROJECT_DIR / "train_selected.csv"
CROPPED_OBJECTS = PROJECT_DIR / "cropped_objects.csv"
CROP_DIR = PROJECT_DIR / "dataset" / "crops"

MISSING_IMAGE = "9ac69b54-61c8a3a1.jpg"


# ------------------------------------------------------------
# Load files
# ------------------------------------------------------------

selected_df = pd.read_csv(TRAIN_SELECTED)
cropped_df = pd.read_csv(CROPPED_OBJECTS)

print("Current cropped metadata rows:", len(cropped_df))


# ------------------------------------------------------------
# Check whether missing image is already present
# ------------------------------------------------------------

if MISSING_IMAGE in cropped_df["image"].values:

    print("Metadata for the image already exists.")
    print("No repair needed.")

else:

    # Find original selected row
    row = selected_df[
        selected_df["image"] == MISSING_IMAGE
    ].iloc[0]

    print("\nMissing metadata found for:")
    print(MISSING_IMAGE)

    # Same filename convention used by fix_missing_crop.py
    crop_filename = (
        f"{row.name:05d}_pedestrian_{row['image']}"
    )

    crop_path = CROP_DIR / crop_filename

    # Check crop exists
    if not crop_path.exists():

        raise FileNotFoundError(
            f"Crop not found:\n{crop_path}"
        )

    # Read crop dimensions
    with Image.open(crop_path) as image:

        crop_width, crop_height = image.size

    # Create metadata row
    new_row = {
        "image": row["image"],
        "video": row["video"],
        "frame_index": row["frame_index"],
        "category": row["category"],
        "x1": row["x1"],
        "y1": row["y1"],
        "x2": row["x2"],
        "y2": row["y2"],
        "weather": row["weather"],
        "timeofday": row["timeofday"],
        "scene": row["scene"],
        "occluded": row["occluded"],
        "truncated": row["truncated"],
        "crop_filename": crop_filename,
        "crop_path": str(crop_path),
        "crop_width": crop_width,
        "crop_height": crop_height
    }

    # Append row
    cropped_df = pd.concat(
        [
            cropped_df,
            pd.DataFrame([new_row])
        ],
        ignore_index=True
    )

    # Save
    cropped_df.to_csv(
        CROPPED_OBJECTS,
        index=False
    )

    print("\nMetadata repaired successfully!")


# ------------------------------------------------------------
# Final verification
# ------------------------------------------------------------

cropped_df = pd.read_csv(CROPPED_OBJECTS)

print("\n" + "=" * 50)
print("FINAL METADATA CHECK")
print("=" * 50)

print("Total rows:", len(cropped_df))

print("\nClass distribution:")
print(cropped_df["category"].value_counts())

print("\nWeather distribution:")
print(cropped_df["weather"].value_counts())

print("\nMissing image present:")
print(
    MISSING_IMAGE in cropped_df["image"].values
)