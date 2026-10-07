import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ------------------------------------------------
# Load data
# ------------------------------------------------

X = np.load("dataset/pca/image_features.npy")

metadata = pd.read_csv("dataset/pca/metadata.csv")

y = metadata["category"]
weather = metadata["weather"]


# ------------------------------------------------
# PCA + MLP Pipeline
# ------------------------------------------------

pipeline = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),

    (
        "pca",
        PCA(
            n_components=0.95,
            random_state=42
        )
    ),

    (
        "mlp",
        MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="sgd",
            learning_rate_init=0.01,
            momentum=0.95,
            max_iter=300,
            random_state=42
        )
    )
])


# ------------------------------------------------
# 5-Fold Cross Validation
# ------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


print("Generating out-of-fold predictions...")

y_pred = cross_val_predict(
    pipeline,
    X,
    y,
    cv=cv
)

print("Predictions generated.")


# ------------------------------------------------
# Weather-wise evaluation
# ------------------------------------------------

weather_types = [
    "clear",
    "overcast",
    "partly cloudy",
    "rainy",
    "snowy"
]

results = []

for condition in weather_types:

    mask = weather == condition

    y_true_weather = y[mask]
    y_pred_weather = y_pred[mask]

    accuracy = accuracy_score(
        y_true_weather,
        y_pred_weather
    )

    precision = precision_score(
        y_true_weather,
        y_pred_weather,
        average="macro"
    )

    recall = recall_score(
        y_true_weather,
        y_pred_weather,
        average="macro"
    )

    f1 = f1_score(
        y_true_weather,
        y_pred_weather,
        average="macro"
    )

    results.append({
        "weather": condition,
        "samples": mask.sum(),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    })


# ------------------------------------------------
# Display results
# ------------------------------------------------

results_df = pd.DataFrame(results)

print("\nWeather-wise Performance")
print("=" * 80)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ------------------------------------------------
# Save results
# ------------------------------------------------

results_df.to_csv(
    "results/evaluation/weather_performance.csv",
    index=False
)

print("\nResults saved to:")
print("results/evaluation/weather_performance.csv")