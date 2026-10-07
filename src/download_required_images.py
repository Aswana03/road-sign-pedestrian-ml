import os
import subprocess
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# Paths
# --------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent

CSV_FILE = PROJECT_DIR / "train_selected.csv"
OUTPUT_DIR = PROJECT_DIR / "images" / "train"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Read selected dataset
# --------------------------------------------------
df = pd.read_csv(CSV_FILE)

image_names = df["image"].drop_duplicates().tolist()

print("=" * 60)
print("BDD100K REQUIRED IMAGE DOWNLOADER")
print("=" * 60)
print(f"Unique images required : {len(image_names)}")
print(f"Output directory       : {OUTPUT_DIR}")
print("=" * 60)

# --------------------------------------------------
# Download images
# --------------------------------------------------
successful = 0
skipped = 0
failed = []

for i, image_name in enumerate(image_names, start=1):

    output_file = OUTPUT_DIR / image_name

    # Skip if already downloaded
    if output_file.exists() and output_file.stat().st_size > 0:
        skipped += 1
        print(f"[{i}/{len(image_names)}] Already exists: {image_name}")
        continue

    print(f"\n[{i}/{len(image_names)}] Downloading: {image_name}")

    kaggle_path = (
        f"bdd100k/bdd100k/images/100k/train/{image_name}"
    )

    # Temporary folder for Kaggle download
    temp_dir = PROJECT_DIR / "_kaggle_temp"
    temp_dir.mkdir(exist_ok=True)

    command = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        "awsaf49/bdd100k-dataset",
        "-f",
        kaggle_path,
        "-p",
        str(temp_dir)
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True
        )

        # Kaggle downloads a ZIP file
        zip_file = temp_dir / (image_name + ".zip")

        if zip_file.exists():

            # Extract ZIP
            subprocess.run(
                [
                    "powershell",
                    "-Command",
                    f"Expand-Archive -Path '{zip_file}' "
                    f"-DestinationPath '{temp_dir}' -Force"
                ],
                check=True
            )

            extracted_file = temp_dir / image_name

            if extracted_file.exists():
                extracted_file.replace(output_file)
                zip_file.unlink(missing_ok=True)

                successful += 1
                print("    ✓ Downloaded")

            else:
                failed.append(image_name)
                print("    ✗ Image not found after extraction")

        else:
            # Some Kaggle versions may download directly
            direct_file = temp_dir / image_name

            if direct_file.exists():
                direct_file.replace(output_file)
                successful += 1
                print("    ✓ Downloaded")
            else:
                failed.append(image_name)
                print("    ✗ Download failed")

    except Exception as e:
        failed.append(image_name)
        print(f"    ✗ Error: {e}")

# --------------------------------------------------
# Summary
# --------------------------------------------------
print("\n")
print("=" * 60)
print("DOWNLOAD COMPLETE")
print("=" * 60)

print(f"Required images : {len(image_names)}")
print(f"Downloaded      : {successful}")
print(f"Already existed : {skipped}")
print(f"Failed          : {len(failed)}")

if failed:
    print("\nFailed images:")
    for name in failed:
        print(name)

    # Save failed list for retry
    with open(PROJECT_DIR / "failed_images.txt", "w") as f:
        for name in failed:
            f.write(name + "\n")

print("=" * 60)