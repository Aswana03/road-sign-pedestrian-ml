import pandas as pd


# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------

INPUT_FILE = "train_objects.csv"
OUTPUT_FILE = "train_selected.csv"

WEATHER_CLASSES = [
    "clear",
    "overcast",
    "partly cloudy",
    "rainy",
    "snowy"
]

TARGET_PER_CLASS_WEATHER = 500

RANDOM_STATE = 42


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

print("=" * 60)
print("CREATING BALANCED TRAINING DATASET")
print("=" * 60)

df = pd.read_csv(INPUT_FILE)

print("\nOriginal dataset:", len(df))


# ---------------------------------------------------------
# CLEAN DATA
# ---------------------------------------------------------

# Keep only required weather conditions
df = df[df["weather"].isin(WEATHER_CLASSES)]

# Keep clearly visible objects
df = df[
    (df["occluded"] == False) &
    (df["truncated"] == False)
]

print("After filtering:", len(df))


# ---------------------------------------------------------
# STRATIFIED SAMPLING
# ---------------------------------------------------------

selected_parts = []

for weather in WEATHER_CLASSES:

    for category in ["pedestrian", "traffic sign"]:

        subset = df[
            (df["weather"] == weather) &
            (df["category"] == category)
        ]

        print(
            f"{weather:15s} | "
            f"{category:15s} | "
            f"available: {len(subset)}"
        )

        if len(subset) < TARGET_PER_CLASS_WEATHER:
            raise ValueError(
                f"Not enough samples for {weather} - {category}"
            )

        sampled = subset.sample(
            n=TARGET_PER_CLASS_WEATHER,
            random_state=RANDOM_STATE
        )

        selected_parts.append(sampled)


# ---------------------------------------------------------
# COMBINE
# ---------------------------------------------------------

selected_df = pd.concat(
    selected_parts,
    ignore_index=True
)


# Shuffle the final dataset
selected_df = selected_df.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)


# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

selected_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ---------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("FINAL DATASET")
print("=" * 60)

print("\nTotal objects:", len(selected_df))

print("\nClass distribution")
print("-" * 40)
print(selected_df["category"].value_counts())

print("\nWeather distribution")
print("-" * 40)
print(selected_df["weather"].value_counts())

print("\nClass × Weather")
print("-" * 40)
print(
    pd.crosstab(
        selected_df["weather"],
        selected_df["category"]
    )
)

print("\nTime of day")
print("-" * 40)
print(selected_df["timeofday"].value_counts())

print("\nClass × Time of day")
print("-" * 40)
print(
    pd.crosstab(
        selected_df["timeofday"],
        selected_df["category"]
    )
)

print("\nSaved to:", OUTPUT_FILE)

print("\nDataset creation complete!")