import json
from collections import Counter, defaultdict


# ---------------------------------------------------------
# FILE PATHS
# ---------------------------------------------------------

TRAIN_JSON = "det_v2_train_release.json"
VAL_JSON = "det_v2_val_release.json"


# ---------------------------------------------------------
# FUNCTION TO ANALYZE ONE JSON FILE
# ---------------------------------------------------------

def analyze_file(filename):

    print("\n" + "=" * 60)
    print("Analyzing:", filename)
    print("=" * 60)

    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("Total frames:", len(data))

    # Counters
    object_counts = Counter()
    weather_counts = Counter()
    time_counts = Counter()
    scene_counts = Counter()

    # More detailed counts
    weather_objects = defaultdict(Counter)
    time_objects = defaultdict(Counter)

    total_labels = 0

    # -----------------------------------------------------
    # READ EACH FRAME
    # -----------------------------------------------------

    for frame in data:

        # Frame-level attributes
        attributes = frame.get("attributes", {})

        weather = attributes.get("weather", "unknown")
        timeofday = attributes.get("timeofday", "unknown")
        scene = attributes.get("scene", "unknown")

        weather_counts[weather] += 1
        time_counts[timeofday] += 1
        scene_counts[scene] += 1

        # -------------------------------------------------
        # READ OBJECT LABELS
        # -------------------------------------------------

        labels = frame.get("labels") or []

        for label in labels:

            category = label.get("category", "unknown")

            object_counts[category] += 1
            total_labels += 1

            # Count objects according to weather
            weather_objects[weather][category] += 1

            # Count objects according to time of day
            time_objects[timeofday][category] += 1

    # -----------------------------------------------------
    # PRINT RESULTS
    # -----------------------------------------------------

    print("\nTOTAL OBJECTS")
    print("-" * 40)

    for category, count in object_counts.most_common():
        print(f"{category:20s}: {count}")

    print("\nWEATHER DISTRIBUTION")
    print("-" * 40)

    for weather, count in weather_counts.most_common():
        print(f"{weather:20s}: {count}")

    print("\nTIME OF DAY DISTRIBUTION")
    print("-" * 40)

    for time, count in time_counts.most_common():
        print(f"{time:20s}: {count}")

    print("\nSCENE DISTRIBUTION")
    print("-" * 40)

    for scene, count in scene_counts.most_common():
        print(f"{scene:25s}: {count}")

    # -----------------------------------------------------
    # PEDestrian + TRAFFIC SIGN
    # -----------------------------------------------------

    print("\nPEDestrian / TRAFFIC SIGN BY WEATHER")
    print("-" * 40)

    for weather in weather_counts:

        pedestrian = weather_objects[weather]["pedestrian"]
        sign = weather_objects[weather]["traffic sign"]

        print(
            f"{weather:15s} | "
            f"Pedestrian: {pedestrian:7d} | "
            f"Traffic Sign: {sign:7d}"
        )

    print("\nPEDestrian / TRAFFIC SIGN BY TIME")
    print("-" * 40)

    for time in time_counts:

        pedestrian = time_objects[time]["pedestrian"]
        sign = time_objects[time]["traffic sign"]

        print(
            f"{time:15s} | "
            f"Pedestrian: {pedestrian:7d} | "
            f"Traffic Sign: {sign:7d}"
        )

    print("\nTotal labels:", total_labels)

    return {
        "frames": len(data),
        "objects": object_counts,
        "weather": weather_counts,
        "timeofday": time_counts,
        "scene": scene_counts,
        "weather_objects": weather_objects,
        "time_objects": time_objects
    }


# ---------------------------------------------------------
# MAIN PROGRAM
# ---------------------------------------------------------

train_results = analyze_file(TRAIN_JSON)
val_results = analyze_file(VAL_JSON)


print("\n")
print("=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)