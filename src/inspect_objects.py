import pandas as pd

CSV_FILE = "train_objects.csv"

df = pd.read_csv(CSV_FILE)

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("\nTotal objects:", len(df))

print("\nClass distribution")
print("-" * 40)
print(df["category"].value_counts())

print("\nWeather distribution")
print("-" * 40)
print(df["weather"].value_counts())

print("\nClass distribution by weather")
print("-" * 40)
print(pd.crosstab(df["weather"], df["category"]))

print("\nClass distribution by time of day")
print("-" * 40)
print(pd.crosstab(df["timeofday"], df["category"]))

print("\nClass distribution by scene")
print("-" * 40)
print(pd.crosstab(df["scene"], df["category"]))

print("\nClass distribution by weather and time")
print("-" * 40)
print(
    pd.crosstab(
        [df["weather"], df["timeofday"]],
        df["category"]
    )
)

print("\nOcclusion")
print("-" * 40)
print(pd.crosstab(df["category"], df["occluded"]))

print("\nTruncation")
print("-" * 40)
print(pd.crosstab(df["category"], df["truncated"]))