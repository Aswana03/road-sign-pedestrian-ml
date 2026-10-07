import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier


# ------------------------------------------------
# Load original image features
# ------------------------------------------------

X = np.load("dataset/pca/image_features.npy")

metadata = pd.read_csv("dataset/pca/metadata.csv")

y = metadata["category"]

print("Feature shape:", X.shape)
print("Number of samples:", len(y))

print("\nClass distribution:")
print(y.value_counts())


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
# Stratified 5-Fold Cross Validation
# ------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ------------------------------------------------
# Evaluation metrics
# ------------------------------------------------

scoring = {
    "accuracy": "accuracy",
    "precision": "precision_macro",
    "recall": "recall_macro",
    "f1": "f1_macro"
}


print("\nRunning 5-Fold Cross-Validation...")
print("-" * 50)

cv_results = cross_validate(
    pipeline,
    X,
    y,
    cv=cv,
    scoring=scoring,
    return_train_score=False
)


# ------------------------------------------------
# Display fold results
# ------------------------------------------------

print("\nFold-wise Results")
print("=" * 60)

for i in range(5):
    print(
        f"Fold {i + 1}: "
        f"Accuracy={cv_results['test_accuracy'][i]:.4f}, "
        f"Precision={cv_results['test_precision'][i]:.4f}, "
        f"Recall={cv_results['test_recall'][i]:.4f}, "
        f"F1={cv_results['test_f1'][i]:.4f}"
    )


# ------------------------------------------------
# Mean and Standard Deviation
# ------------------------------------------------

print("\nCross-Validation Summary")
print("=" * 60)

for metric in ["accuracy", "precision", "recall", "f1"]:

    scores = cv_results[f"test_{metric}"]

    print(
        f"{metric.capitalize():10s}: "
        f"{scores.mean():.4f} ± {scores.std():.4f}"
    )