import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)
import matplotlib.pyplot as plt

# Load PCA features
X = np.load("dataset/pca/pca_features_95.npy")

# Load metadata
metadata = pd.read_csv("dataset/pca/metadata.csv")

# Extract target labels
y = metadata["category"]

print("Feature shape:", X.shape)
print("Number of labels:", len(y))
print("\nClass distribution:")
print(y.value_counts())

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Validation samples:", len(X_val))

# Create MLP classifier using Stochastic Gradient Descent
mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="sgd",
    learning_rate_init=0.001,
    momentum=0.9,
    max_iter=100,
    random_state=42
)

print("\nTraining MLP...")
mlp.fit(X_train, y_train)

print("Training completed!")

# Make predictions
y_pred = mlp.predict(X_val)

# Calculate accuracy
accuracy = accuracy_score(y_val, y_pred)

print("\nValidation Accuracy:", accuracy)

# ------------------------------------------------
# SGD Hyperparameter Optimization
# ------------------------------------------------

learning_rates = [0.0001, 0.001, 0.01]
momentums = [0.0, 0.9, 0.95]

results = []

print("\nSGD Hyperparameter Optimization")
print("-" * 50)

for lr in learning_rates:
    for momentum in momentums:

        print(f"\nTesting learning_rate={lr}, momentum={momentum}")

        model = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="sgd",
            learning_rate_init=lr,
            momentum=momentum,
            max_iter=300,
            random_state=42
        )

        model.fit(X_train, y_train)

        val_pred = model.predict(X_val)
        val_accuracy = accuracy_score(y_val, val_pred)

        results.append({
            "learning_rate": lr,
            "momentum": momentum,
            "accuracy": val_accuracy,
            "iterations": model.n_iter_
        })

        print(f"Accuracy: {val_accuracy:.4f}")
        print(f"Iterations: {model.n_iter_}")

results_df = pd.DataFrame(results)

print("\n\nFinal Results")
print("=" * 60)
print(results_df)

best_result = results_df.loc[results_df["accuracy"].idxmax()]

print("\nBest Configuration:")
print(best_result)

# ------------------------------------------------
# Final Optimized MLP
# ------------------------------------------------

final_mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="sgd",
    learning_rate_init=0.01,
    momentum=0.95,
    max_iter=300,
    random_state=42
)

print("\nTraining Final Optimized MLP...")

final_mlp.fit(X_train, y_train)

# Predictions
y_pred = final_mlp.predict(X_val)

print("Final model trained!")
print("Iterations:", final_mlp.n_iter_)



# ------------------------------------------------
# Classification Metrics
# ------------------------------------------------

accuracy = accuracy_score(y_val, y_pred)

precision = precision_score(
    y_val,
    y_pred,
    pos_label="pedestrian"
)

recall = recall_score(
    y_val,
    y_pred,
    pos_label="pedestrian"
)

f1 = f1_score(
    y_val,
    y_pred,
    pos_label="pedestrian"
)

print("\nFinal Model Performance")
print("=" * 50)
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_val, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_val, y_pred))

# ------------------------------------------------
# ROC Curve
# ------------------------------------------------

# Probability of pedestrian class
pedestrian_index = list(final_mlp.classes_).index("pedestrian")

y_prob = final_mlp.predict_proba(X_val)[:, pedestrian_index]

# Convert labels to binary
y_binary = (y_val == "pedestrian").astype(int)

# ROC
fpr, tpr, thresholds = roc_curve(y_binary, y_prob)
roc_auc = roc_auc_score(y_binary, y_prob)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, label=f"MLP-SGD (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - MLP with SGD")
plt.legend()
plt.grid(True)

plt.savefig(
    "results/evaluation/roc_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(f"\nROC-AUC: {roc_auc:.4f}")


# ------------------------------------------------
# Precision-Recall Curve
# ------------------------------------------------

precision_values, recall_values, pr_thresholds = precision_recall_curve(
    y_binary,
    y_prob
)

pr_auc = average_precision_score(y_binary, y_prob)

plt.figure(figsize=(7, 6))
plt.plot(
    recall_values,
    precision_values,
    label=f"MLP-SGD (AP = {pr_auc:.3f})"
)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve - MLP with SGD")
plt.legend()
plt.grid(True)

plt.savefig(
    "results/evaluation/precision_recall_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(f"PR-AUC / Average Precision: {pr_auc:.4f}")