import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from tensorflow import keras
import matplotlib.pyplot as plt

from predict import (
    CATEGORICAL_COLS,
    NUMERIC_COLS,
    CROP_YIELD_RANGES,
)

CROPS = ["rice", "coconut", "arecanut", "banana", "black pepper", "cashew", "cocoa", "sweet potato"]
MODEL_NAMES = ["LSTM", "BiLSTM", "GRU", "CNN-LSTM", "Transformer", "Autoencoder"]

FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATASET_PATH = PROJECT_ROOT / "data" / "dakshina_kannada_crop_data.csv"
MODELS_DIR = BASE_DIR / "models"
MODEL_EXT = ".keras"

sns.set_theme(style="whitegrid")


def load_pickles(crop):
    import pickle

    with open(MODELS_DIR / f"{crop}_scaler_X.pkl", "rb") as f:
        scaler_x = pickle.load(f)
    with open(MODELS_DIR / f"{crop}_scaler_y.pkl", "rb") as f:
        scaler_y = pickle.load(f)
    with open(MODELS_DIR / f"{crop}_label_encoders.pkl", "rb") as f:
        encoders = pickle.load(f)
    return scaler_x, scaler_y, encoders


def encode_categoricals(df, encoders):
    df = df.copy()
    for col in CATEGORICAL_COLS:
        le = encoders[col]
        known = list(le.classes_)
        df[col] = df[col].astype(str).map(lambda v: v if v in known else known[0])
        df[col] = le.transform(df[col])
    return df


def train_val_test_split(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, shuffle=True
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, shuffle=True
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def yield_category(crop, values):
    y_min, y_max = CROP_YIELD_RANGES[crop]
    span = y_max - y_min
    thresholds = [y_min + 0.25 * span, y_min + 0.5 * span, y_min + 0.75 * span]
    labels = ["Poor", "Average", "Good", "Excellent"]
    cats = []
    for val in values:
        if val < thresholds[0]:
            cats.append(labels[0])
        elif val < thresholds[1]:
            cats.append(labels[1])
        elif val < thresholds[2]:
            cats.append(labels[2])
        else:
            cats.append(labels[3])
    return np.array(cats), labels


def compute_class_scores(crop, y_pred, labels):
    y_min, y_max = CROP_YIELD_RANGES[crop]
    span = y_max - y_min
    centers = np.array([
        y_min + 0.125 * span,
        y_min + 0.375 * span,
        y_min + 0.625 * span,
        y_min + 0.875 * span,
    ])
    scale = span / 8.0 if span > 0 else 1.0
    distances = np.abs(y_pred.reshape(-1, 1) - centers.reshape(1, -1))
    raw_scores = np.exp(-distances / scale)
    normalized = raw_scores / np.maximum(raw_scores.sum(axis=1, keepdims=True), 1e-8)
    return normalized


def model_predict(model, model_name, X_scaled):
    if model_name in ("LSTM", "BiLSTM", "GRU", "CNN-LSTM", "Transformer"):
        X_input = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
    else:
        X_input = X_scaled
    return model.predict(X_input, verbose=0).reshape(-1)


def compute_regression_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8))) * 100
    accuracy_pct = max(0, (1 - mape / 100) * 100)
    return {
        "r2": round(float(r2), 4),
        "rmse": round(float(rmse), 2),
        "mae": round(float(mae), 2),
        "mape": round(float(mape), 2),
        "accuracy_percent": round(float(accuracy_pct), 2),
    }

def save_plot(fig, out_dir, name, title):
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{name}.png"
    svg_path = out_dir / f"{name}.svg"
    fig.suptitle(title, fontsize=14, fontweight="600")
    fig.tight_layout()
    fig.savefig(png_path, dpi=200)
    fig.savefig(svg_path)
    plt.close(fig)
    return png_path.name


def plot_confusion_matrix(y_true_cat, y_pred_cat, labels, out_dir):
    cm = confusion_matrix(y_true_cat, y_pred_cat, labels=labels)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", cbar=False,
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted Category")
    ax.set_ylabel("Actual Category")
    return save_plot(fig, out_dir, "confusion_matrix", "Yield Category Confusion Matrix")


def plot_classification_metrics(report, out_dir):
    labels = [k for k in report.keys() if k not in ("accuracy", "macro avg", "weighted avg")]
    metrics = ["precision", "recall", "f1-score"]
    data = {m: [report[l][m] for l in labels] for m in metrics}

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for idx, metric in enumerate(metrics):
        ax.bar(x + idx * width, data[metric], width, label=metric.title())

    ax.set_xticks(x + width)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.legend()
    return save_plot(fig, out_dir, "precision_recall_f1", "Precision, Recall, and F1 by Category")


def plot_roc_curve(y_true_cat, y_pred_scores, labels, out_dir):
    y_true_bin = label_binarize(y_true_cat, classes=labels)
    if y_true_bin.shape[1] < 2:
        return None

    fig, ax = plt.subplots(figsize=(7, 5.5))
    plotted = 0
    for idx, label in enumerate(labels):
        positives = y_true_bin[:, idx].sum()
        if positives == 0 or positives == len(y_true_bin):
            continue
        fpr, tpr, _ = roc_curve(y_true_bin[:, idx], y_pred_scores[:, idx])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{label} (AUC {roc_auc:.2f})")
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return None

    ax.plot([0, 1], [0, 1], "--", color="#888")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    return save_plot(fig, out_dir, "roc_curve", "ROC Curve (One-vs-Rest)")


def plot_predicted_vs_actual(y_true, y_pred, out_dir):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolor="none")
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "--", color="#444")
    ax.set_xlabel("Actual Yield (kg/ha)")
    ax.set_ylabel("Predicted Yield (kg/ha)")
    return save_plot(fig, out_dir, "predicted_vs_actual", "Predicted vs Actual")


def plot_residuals(y_true, y_pred, out_dir):
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].scatter(y_pred, residuals, alpha=0.6, edgecolor="none")
    axes[0].axhline(0, color="#444", linestyle="--")
    axes[0].set_xlabel("Predicted Yield")
    axes[0].set_ylabel("Residuals")

    sns.histplot(residuals, bins=30, kde=True, ax=axes[1], color="#2D6A4F")
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Count")
    return save_plot(fig, out_dir, "residuals", "Residual Analysis")


def plot_feature_importance(model, model_name, X_test, y_true, feature_names, out_dir, repeats=3):
    base_pred = model_predict(model, model_name, X_test)
    base_r2 = r2_score(y_true, base_pred)
    importances = []
    rng = np.random.default_rng(42)

    for idx in range(X_test.shape[1]):
        drops = []
        for _ in range(repeats):
            X_shuffled = X_test.copy()
            rng.shuffle(X_shuffled[:, idx])
            pred = model_predict(model, model_name, X_shuffled)
            drops.append(base_r2 - r2_score(y_true, pred))
        importances.append(np.mean(drops))

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.barh(importance_df["feature"], importance_df["importance"], color="#40916C")
    ax.set_xlabel("R2 Drop (Permutation Importance)")
    return save_plot(fig, out_dir, "feature_importance", "Permutation Feature Importance")


def plot_training_curves(history, out_dir):
    if not history or "loss" not in history:
        return None

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.plot(history["loss"], label="Train Loss")
    if "val_loss" in history:
        ax.plot(history["val_loss"], label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    return save_plot(fig, out_dir, "training_curves", "Training Loss Curves")


def build_manifest(crop, model, out_dir, metrics, plots, notes):
    manifest = {
        "crop": crop,
        "model": model,
        "metrics": metrics,
        "plots": plots,
        "notes": notes,
    }
    with open(out_dir / "plots.json", "w") as f:
        json.dump(manifest, f, indent=2)


def generate_for_pair(crop, model_name, args):
    crop = crop.lower().strip()
    model_name = model_name.strip()
    if crop not in CROP_YIELD_RANGES:
        raise SystemExit(f"Unknown crop: {crop}")

    out_dir = Path(args.output_dir) if args.output_dir else (
        PROJECT_ROOT / "frontend" / "assets" / "evaluation" / f"{crop}_{model_name}"
    )

    df = pd.read_csv(DATASET_PATH)
    df = df[df["crop"].str.lower() == crop].copy()
    if df.empty:
        raise SystemExit(f"No rows found for crop: {crop}")

    scaler_x, scaler_y, encoders = load_pickles(crop)
    df_encoded = encode_categoricals(df, encoders)

    X = df_encoded[FEATURE_COLS].values.astype(np.float32)
    y = df_encoded["yield_kg_ha"].values.astype(np.float32)

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)
    X_test_scaled = scaler_x.transform(X_test)

    model_path = MODELS_DIR / f"{crop}_{model_name}{MODEL_EXT}"
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")
    model = keras.models.load_model(model_path)

    y_pred_scaled = model_predict(model, model_name, X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).reshape(-1)

    metrics = compute_regression_metrics(y_test, y_pred)

    plots = []
    notes = []

    plots.append({
        "title": "Predicted vs Actual",
        "file": plot_predicted_vs_actual(y_test, y_pred, out_dir),
        "description": "Scatter plot with identity line.",
    })

    plots.append({
        "title": "Residual Analysis",
        "file": plot_residuals(y_test, y_pred, out_dir),
        "description": "Residuals vs predicted and residual distribution.",
    })

    y_true_cat, labels = yield_category(crop, y_test)
    y_pred_cat, _ = yield_category(crop, y_pred)

    plots.append({
        "title": "Yield Category Confusion Matrix",
        "file": plot_confusion_matrix(y_true_cat, y_pred_cat, labels, out_dir),
        "description": "Confusion matrix for yield categories.",
    })

    report = classification_report(y_true_cat, y_pred_cat, output_dict=True, zero_division=0)
    plots.append({
        "title": "Precision / Recall / F1",
        "file": plot_classification_metrics(report, out_dir),
        "description": "Classification metrics per category.",
    })

    if len(np.unique(y_true_cat)) > 1:
        y_scores = compute_class_scores(crop, y_pred, labels)
        roc_file = plot_roc_curve(y_true_cat, y_scores, labels, out_dir)
        if roc_file:
            plots.append({
                "title": "ROC Curve (One-vs-Rest)",
                "file": roc_file,
                "description": "ROC curves for each yield category.",
            })
    else:
        notes.append("ROC curve skipped due to a single class in the test set.")

    if args.sample_size and X_test_scaled.shape[0] > args.sample_size:
        idx = np.random.default_rng(42).choice(X_test_scaled.shape[0], args.sample_size, replace=False)
        X_import = X_test_scaled[idx]
        y_import = y_test[idx]
    else:
        X_import = X_test_scaled
        y_import = y_test

    plots.append({
        "title": "Permutation Feature Importance",
        "file": plot_feature_importance(
            model,
            model_name,
            X_import,
            y_import,
            FEATURE_COLS,
            out_dir,
            repeats=args.perm_repeats,
        ),
        "description": "Permutation importance based on R2 drop.",
    })

    if args.history:
        history_path = Path(args.history)
        if history_path.exists():
            with open(history_path, "r") as f:
                history = json.load(f)
            training_file = plot_training_curves(history, out_dir)
            if training_file:
                plots.append({
                    "title": "Training Loss Curves",
                    "file": training_file,
                    "description": "Training and validation loss curves.",
                })
        else:
            notes.append("Training history file not found; skipped loss curves.")
    else:
        notes.append("Training history not provided; skipped loss curves.")

    build_manifest(crop, model_name, out_dir, metrics, plots, notes)
    print(f"Saved {len(plots)} plots to {out_dir}")


def parse_list(value, default_values):
    if not value:
        return list(default_values)
    return [item.strip() for item in value.split(",") if item.strip()]


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation plots for crop yield models.")
    parser.add_argument("--crop", default="rice", help="Crop name (e.g., rice)")
    parser.add_argument("--model", default="LSTM", help="Model name (e.g., LSTM)")
    parser.add_argument("--crops", default=None, help="Comma-separated crops (overrides --crop)")
    parser.add_argument("--models", default=None, help="Comma-separated models (overrides --model)")
    parser.add_argument("--all", action="store_true", help="Generate plots for all crops and models")
    parser.add_argument("--output-dir", default=None, help="Output directory for plots")
    parser.add_argument("--history", default=None, help="Optional training history JSON file")
    parser.add_argument("--perm-repeats", type=int, default=3, help="Permutation repeats")
    parser.add_argument("--sample-size", type=int, default=600, help="Sample size for heavy plots")
    args = parser.parse_args()

    if not DATASET_PATH.exists():
        raise SystemExit("Dataset not found. Expected data/dakshina_kannada_crop_data.csv")

    if args.all:
        crops = list(CROPS)
        models = list(MODEL_NAMES)
    else:
        crops = parse_list(args.crops, [args.crop])
        models = parse_list(args.models, [args.model])

    total = len(crops) * len(models)
    count = 0
    for crop in crops:
        for model_name in models:
            count += 1
            print(f"[{count}/{total}] {crop} / {model_name}")
            generate_for_pair(crop, model_name, args)


if __name__ == "__main__":
    main()
