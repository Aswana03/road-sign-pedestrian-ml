import json
import csv


TRAIN_JSON = "det_v2_train_release.json"
VAL_JSON = "det_v2_val_release.json"

TRAIN_CSV = "train_objects.csv"
VAL_CSV = "val_objects.csv"


# ---------------------------------------------------------
# EXTRACT PEDESTRIAN AND TRAFFIC SIGN OBJECTS
# ---------------------------------------------------------

def extract_objects(json_file, csv_file):

    print("\nReading:", json_file)

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []

    for frame in data:

        frame_name = frame.get("name", "")
        video_name = frame.get("videoName", "")
        frame_index = frame.get("index", "")

        # Frame-level attributes
        attributes = frame.get("attributes") or {}

        weather = attributes.get("weather", "undefined")
        timeofday = attributes.get("timeofday", "undefined")
        scene = attributes.get("scene", "undefined")

        # Object labels
        labels = frame.get("labels") or []

        for label in labels:

            category = label.get("category", "")

            # We only want these two classes
            if category not in ["pedestrian", "traffic sign"]:
                continue

            box = label.get("box2d")

            # Skip objects without bounding boxes
            if not box:
                continue

            x1 = box.get("x1")
            y1 = box.get("y1")
            x2 = box.get("x2")
            y2 = box.get("y2")

            label_attributes = label.get("attributes") or {}

            occluded = label_attributes.get("occluded", "")
            truncated = label_attributes.get("truncated", "")

            rows.append([
                frame_name,
                video_name,
                frame_index,
                category,
                x1,
                y1,
                x2,
                y2,
                weather,
                timeofday,
                scene,
                occluded,
                truncated
            ])

    # -----------------------------------------------------
    # WRITE CSV
    # -----------------------------------------------------

    headers = [
        "image",
        "video",
        "frame_index",
        "category",
        "x1",
        "y1",
        "x2",
        "y2",
        "weather",
        "timeofday",
        "scene",
        "occluded",
        "truncated"
    ]

    with open(csv_file, "w", newline="", encoding="utf-8") as f:

        writer = csv.writer(f)

        writer.writerow(headers)
        writer.writerows(rows)

    print("Saved:", csv_file)
    print("Total selected objects:", len(rows))


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

extract_objects(TRAIN_JSON, TRAIN_CSV)

extract_objects(VAL_JSON, VAL_CSV)

print("\nExtraction complete!")