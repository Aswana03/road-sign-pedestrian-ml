from pathlib import Path
import random
from PIL import Image
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parent
CROP_DIR = PROJECT_DIR / "dataset" / "crops"

# Number of images to display
NUM_IMAGES = 12

# Find all cropped images
crop_files = list(CROP_DIR.glob("*.jpg"))

if len(crop_files) == 0:
    print("No crop images found!")
    exit()

print("Total crops found:", len(crop_files))

# Randomly select images
random.seed(42)
selected = random.sample(crop_files, min(NUM_IMAGES, len(crop_files)))

# Create figure
fig, axes = plt.subplots(3, 4, figsize=(12, 9))
axes = axes.ravel()

for ax, image_path in zip(axes, selected):

    try:
        image = Image.open(image_path).convert("RGB")

        ax.imshow(image)
        ax.axis("off")

        # Filename format:
        # 00001_pedestrian_xxxxx.jpg
        filename = image_path.stem

        # Extract class from filename
        parts = filename.split("_")

        if len(parts) >= 2:
            category = parts[1]
        else:
            category = "unknown"

        ax.set_title(category.capitalize())

    except Exception as e:
        ax.text(
            0.5,
            0.5,
            "Error loading image",
            ha="center",
            va="center"
        )
        ax.axis("off")

# Hide unused axes
for ax in axes[len(selected):]:
    ax.axis("off")

plt.suptitle(
    "BDD100K Object Crop Verification",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout()
plt.show()