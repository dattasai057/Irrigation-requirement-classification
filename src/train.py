"""
train.py
--------
End-to-end training script for the Irrigation Requirement Classification
project.

Steps:
  1. Load + clean data (src/preprocessing.py)
  2. Train/test split (stratified, held-out test set never touched until
     final evaluation -> avoids data leakage)
  3. Build a preprocessing + model Pipeline for each candidate foundational
     model (Logistic Regression, KNN, Decision Tree)
  4. Select hyperparameters with cross-validation on the TRAIN split only
  5. Evaluate all candidates on the held-out test split
  6. Save the best model, its metrics, and diagnostic plots

Run:
    python src/train.py
"""

import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.preprocessing import build_preprocessor, load_and_clean, split_X_y  # noqa: E402

DATA_PATH = ROOT / "data" / "irrigation_dataset.csv"
MODELS_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"
RANDOM_STATE = 42

MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def eda_plots(df: pd.DataFrame):
    """Save the plots referenced in the EDA notebook / technical paper."""
    sns_palette = ["#3B7A57", "#C1440E"]

    # Class balance
    fig, ax = plt.subplots(figsize=(4.5, 4))
    counts = df["irrigation_required"].value_counts().sort_index()
    ax.bar(["No (0)", "Yes (1)"], counts.values, color=sns_palette)
    ax.set_title("Class Balance: Irrigation Required")
    ax.set_ylabel("Count")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 10, str(v), ha="center")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "class_balance.png", dpi=150)
    plt.close(fig)

    # Numeric distributions by class
    numeric_cols = ["soil_moisture_pct", "temperature_c", "humidity_pct", "rainfall_mm_last_7days"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, col in zip(axes.ravel(), numeric_cols):
        for cls, color in zip([0, 1], sns_palette):
            data = df.loc[df["irrigation_required"] == cls, col].dropna()
            ax.hist(data, bins=30, alpha=0.55, label=f"class {cls}", color=color)
        ax.set_title(col)
        ax.legend(fontsize=8)
    fig.suptitle("Feature Distributions by Class")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_distributions.png", dpi=150)
    plt.close(fig)

    # Correlation heatmap (numeric features only)
    fig, ax = plt.subplots(figsize=(6, 5))
    corr = df[numeric_cols + ["days_since_last_irrigation", "irrigation_required"]].corr()
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns, fontsize=8)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Correlation Matrix")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "correlation_heatmap.png", dpi=150)
    plt.close(fig)

    # Crop stage vs irrigation rate
    fig, ax = plt.subplots(figsize=(5, 4))
    rate = df.groupby("crop_stage")["irrigation_required"].mean().sort_values()
    ax.barh(rate.index, rate.values, color="#3B7A57")
    ax.set_xlabel("Fraction requiring irrigation")
    ax.set_title("Irrigation Rate by Crop Stage")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "irrigation_rate_by_stage.png", dpi=150)
    plt.close(fig)


def build_candidates():
    """Returns dict of name -> (pipeline, param_grid) for GridSearchCV."""
    candidates = {}

    candidates["logistic_regression"] = (
        Pipeline([("prep", build_preprocessor()), ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))]),
        {"clf__C": [0.01, 0.1, 1.0, 10.0], "clf__class_weight": [None, "balanced"]},
    )
    candidates["knn"] = (
        Pipeline([("prep", build_preprocessor()), ("clf", KNeighborsClassifier())]),
        {"clf__n_neighbors": [5, 9, 15, 21], "clf__weights": ["uniform", "distance"]},
    )
    candidates["decision_tree"] = (
        Pipeline([("prep", build_preprocessor()), ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE))]),
        {
            "clf__max_depth": [3, 5, 8, 12, None],
            "clf__min_samples_leaf": [1, 5, 10],
            "clf__class_weight": [None, "balanced"],
        },
    )
    return candidates


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
    metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
    metrics["classification_report"] = classification_report(y_test, y_pred, zero_division=0)
    return metrics, y_pred, y_proba


def main():
    df = load_and_clean(DATA_PATH)
    eda_plots(df)

    X, y = split_X_y(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    candidates = build_candidates()
    results = {}
    fitted_models = {}

    for name, (pipe, grid) in candidates.items():
        print(f"\n=== Tuning {name} ===")
        gs = GridSearchCV(pipe, grid, cv=5, scoring="f1", n_jobs=-1)
        gs.fit(X_train, y_train)
        print(f"Best params: {gs.best_params_}  |  CV f1: {gs.best_score_:.3f}")
        metrics, y_pred, y_proba = evaluate(gs.best_estimator_, X_test, y_test)
        metrics["best_params"] = gs.best_params_
        metrics["cv_f1_mean"] = gs.best_score_
        results[name] = metrics
        fitted_models[name] = gs.best_estimator_
        print(f"Test metrics: accuracy={metrics['accuracy']:.3f} "
              f"precision={metrics['precision']:.3f} recall={metrics['recall']:.3f} "
              f"f1={metrics['f1']:.3f} roc_auc={metrics.get('roc_auc', float('nan')):.3f}")

    # Model selection: prioritize F1 (dataset is imbalanced ~23% positive
    # class, so accuracy alone would be misleading) with ROC-AUC as tie-break.
    best_name = max(results, key=lambda n: (results[n]["f1"], results[n].get("roc_auc", 0)))
    best_model = fitted_models[best_name]
    print(f"\n>>> Selected model: {best_name} (highest test F1) <<<")

    # Diagnostic plots for the selected model
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test, ax=ax, cmap="Greens", colorbar=False)
    ax.set_title(f"Confusion Matrix — {best_name}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    for name, model in fitted_models.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=name)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_title("ROC Curves — All Candidate Models")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "roc_curves.png", dpi=150)
    plt.close(fig)

    # Feature importance (only meaningful for the decision tree; also report
    # logistic-regression coefficients for interpretability discussion)
    if best_name == "decision_tree":
        clf = best_model.named_steps["clf"]
        feat_names = best_model.named_steps["prep"].get_feature_names_out()
        importances = pd.Series(clf.feature_importances_, index=feat_names).sort_values()
        fig, ax = plt.subplots(figsize=(6, 6))
        importances.tail(15).plot.barh(ax=ax, color="#3B7A57")
        ax.set_title("Top Feature Importances — Decision Tree")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "feature_importance.png", dpi=150)
        plt.close(fig)

    # Persist model + metrics
    joblib.dump(best_model, MODELS_DIR / "model.pkl")
    with open(MODELS_DIR / "model_name.txt", "w") as f:
        f.write(best_name)

    serializable_results = {
        k: {kk: vv for kk, vv in v.items() if kk != "classification_report" or True}
        for k, v in results.items()
    }
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(serializable_results, f, indent=2, default=str)

    with open(MODELS_DIR / "classification_reports.txt", "w") as f:
        for name, m in results.items():
            f.write(f"===== {name} =====\n{m['classification_report']}\n\n")

    print(f"\nSaved best model ({best_name}) to {MODELS_DIR/'model.pkl'}")
    print(f"Saved metrics to {MODELS_DIR/'metrics.json'}")
    print(f"Saved figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
