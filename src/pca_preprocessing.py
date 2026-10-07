from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

CROP_DIR = PROJECT_DIR / "dataset" / "crops"
METADATA_FILE = PROJECT_DIR / "cropped_objects.csv"

OUTPUT_DIR = PROJECT_DIR / "dataset" / "pca"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = (32, 32)


# ============================================================
# LOAD METADATA
# ============================================================

print("=" * 60)
print("BDD100K PCA PREPROCESSING")
print("=" * 60)

df = pd.read_csv(METADATA_FILE)

print("\nMetadata loaded.")
print("Number of samples:", len(df))

print("\nClass distribution:")
print(df["category"].value_counts())

print("\nWeather distribution:")
print(df["weather"].value_counts())


# ============================================================
# LOAD AND PREPROCESS IMAGES
# ============================================================

print("\nLoading images...")

features = []
valid_rows = []

for index, row in df.iterrows():

    image_path = CROP_DIR / row["crop_filename"]

    try:

        # Open image
        image = Image.open(image_path).convert("L")

        # Resize to 32 x 32
        image = image.resize(IMAGE_SIZE)

        # Convert to NumPy array
        image_array = np.asarray(image, dtype=np.float32)

        # Normalize pixel values to 0-1
        image_array = image_array / 255.0

        # Flatten 32x32 -> 1024 features
        image_array = image_array.flatten()

        features.append(image_array)
        valid_rows.append(index)

    except Exception as e:

        print(f"Could not process {image_path}: {e}")


X = np.array(features)

df = df.iloc[valid_rows].reset_index(drop=True)


print("\nImage preprocessing completed.")

print("Feature matrix shape:", X.shape)

print("Expected shape: (5000, 1024)")


# ============================================================
# SAVE BASIC FEATURES
# ============================================================

np.save(
    OUTPUT_DIR / "image_features.npy",
    X
)

df.to_csv(
    OUTPUT_DIR / "metadata.csv",
    index=False
)

print("\nSaved:")
print(" - image_features.npy")
print(" - metadata.csv")


# ============================================================
# STANDARDIZATION
# ============================================================

print("\nStandardizing features...")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("Standardized feature shape:", X_scaled.shape)


# ============================================================
# PCA
# ============================================================

print("\nRunning PCA...")

pca = PCA()

X_pca = pca.fit_transform(X_scaled)


print("PCA completed.")

print("Original number of features:", X.shape[1])
print("Number of PCA components:", X_pca.shape[1])


# ============================================================
# EXPLAINED VARIANCE
# ============================================================

explained_variance = pca.explained_variance_ratio_

cumulative_variance = np.cumsum(explained_variance)


print("\nExplained variance:")

variance_levels = [0.80, 0.85, 0.90, 0.95, 0.99]

for level in variance_levels:

    components = np.argmax(
        cumulative_variance >= level
    ) + 1

    print(
        f"{int(level * 100)}% variance -> "
        f"{components} components"
    )


# ============================================================
# FIND 95% VARIANCE COMPONENTS
# ============================================================

n_components_95 = np.argmax(
    cumulative_variance >= 0.95
) + 1


print("\nRecommended PCA components:")
print(
    f"{n_components_95} components "
    f"retain at least 95% of the variance."
)


# ============================================================
# SCREE PLOT
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    range(1, len(explained_variance) + 1),
    explained_variance
)

plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")

plt.title(
    "PCA Scree Plot"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "scree_plot.png",
    dpi=300
)

plt.show()


# ============================================================
# CUMULATIVE EXPLAINED VARIANCE
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    range(1, len(cumulative_variance) + 1),
    cumulative_variance
)

plt.axhline(
    y=0.95,
    linestyle="--",
    label="95% variance"
)

plt.axvline(
    x=n_components_95,
    linestyle="--",
    label=f"{n_components_95} components"
)

plt.xlabel("Number of PCA Components")

plt.ylabel(
    "Cumulative Explained Variance"
)

plt.title(
    "Cumulative Explained Variance"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "cumulative_explained_variance.png",
    dpi=300
)

plt.show()


# ============================================================
# SAVE PCA RESULTS
# ============================================================

np.save(
    OUTPUT_DIR / "pca_features_all.npy",
    X_pca
)

np.save(
    OUTPUT_DIR / "explained_variance_ratio.npy",
    explained_variance
)

np.save(
    OUTPUT_DIR / "cumulative_variance.npy",
    cumulative_variance
)


# ============================================================
# PCA WITH 95% VARIANCE
# ============================================================

pca_95 = PCA(
    n_components=n_components_95
)

X_pca_95 = pca_95.fit_transform(X_scaled)


np.save(
    OUTPUT_DIR / "pca_features_95.npy",
    X_pca_95
)


# ============================================================
# SAVE FINAL METADATA
# ============================================================

df.to_csv(
    OUTPUT_DIR / "pca_metadata.csv",
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PCA PREPROCESSING COMPLETE")
print("=" * 60)

print("\nOriginal feature size:")
print("1024 features per image")

print("\nPCA feature size:")
print(X_pca_95.shape[1])

print("\nSamples:")
print(X_pca_95.shape[0])

print("\nVariance retained:")
print(
    f"{pca_95.explained_variance_ratio_.sum() * 100:.2f}%"
)

print("\nOutput directory:")
print(OUTPUT_DIR)

print("\nGenerated files:")

print(" - image_features.npy")
print(" - metadata.csv")
print(" - pca_features_all.npy")
print(" - pca_features_95.npy")
print(" - explained_variance_ratio.npy")
print(" - cumulative_variance.npy")
print(" - pca_metadata.csv")
print(" - scree_plot.png")
print(" - cumulative_explained_variance.png")

print("\nDone!")