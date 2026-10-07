from pathlib import Path
from PIL import Image
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent

df = pd.read_csv(PROJECT_DIR / "train_selected.csv")

# Find the missing sample
row = df[df["image"] == "9ac69b54-61c8a3a1.jpg"].iloc[0]

image_path = PROJECT_DIR / "images" / "train" / row["image"]
output_dir = PROJECT_DIR / "dataset" / "crops"

# Open image
image = Image.open(image_path).convert("RGB")

# Bounding box
x1 = int(row["x1"])
y1 = int(row["y1"])
x2 = int(row["x2"])
y2 = int(row["y2"])

# Crop
crop = image.crop((x1, y1, x2, y2))

# Use the same row index as the original cropping script
crop_name = f"{row.name:05d}_pedestrian_{row['image']}"
crop_path = output_dir / crop_name

crop.save(crop_path, quality=95)

print("Missing crop created successfully!")
print("Class   :", row["category"])
print("Weather :", row["weather"])
print("Crop    :", crop_path)
print("Size    :", crop.size)