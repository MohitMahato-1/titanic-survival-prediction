"""Reproducible Titanic baseline, model selection, held-out evaluation, and prediction.

Run ``python titanic.py --help``. Paths default to this file's repository directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
SEED = 42
INPUT_COLUMNS = ["PassengerId", "Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
NUMERIC = ["Age", "Fare", "SibSp", "Parch", "FamilySize", "IsAlone"]
CATEGORICAL = ["Pclass", "Sex", "Embarked", "Title"]


def read_data(path: Path, labeled: bool = True) -> pd.DataFrame:
    """Reject malformed inputs early instead of silently training on broken data."""
    frame = pd.read_csv(path)
    required = INPUT_COLUMNS + (["Survived"] if labeled else [])
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    if frame.empty or frame.PassengerId.isna().any() or frame.PassengerId.duplicated().any():
        raise ValueError("PassengerId must be present and unique in a nonempty dataset")
    if labeled and (frame.Survived.isna().any() or not set(frame.Survived).issubset({0, 1})):
        raise ValueError("Survived must contain only 0 or 1")
    return frame


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive only per-passenger features: no population statistics or target access."""
    features = frame[INPUT_COLUMNS].copy()
    features["FamilySize"] = features.SibSp + features.Parch + 1
    features["IsAlone"] = (features.FamilySize == 1).astype(int)
    title = features.Name.str.extract(r" ([A-Za-z]+)\.", expand=False)
    title = title.replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    features["Title"] = title.where(title.isin(["Mr", "Mrs", "Miss", "Master"]), "Rare")
    return features[NUMERIC + CATEGORICAL]


def make_pipeline(estimator) -> Pipeline:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric, NUMERIC), ("categorical", categorical, CATEGORICAL),
    ])
    return Pipeline([
        ("features", FunctionTransformer(add_features, validate=False)),
        ("preprocess", preprocess), ("model", estimator),
    ])


def split_data(frame: pd.DataFrame):
    """A fixed stratified 80/20 partition, shared by EDA and training."""
    return train_test_split(frame, test_size=0.2, stratify=frame.Survived, random_state=SEED)


def data_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def scores(truth, predicted, probabilities) -> dict:
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "precision": float(precision_score(truth, predicted, zero_division=0)),
        "recall": float(recall_score(truth, predicted, zero_division=0)),
        "f1": float(f1_score(truth, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(truth, probabilities)),
        "confusion_matrix": confusion_matrix(truth, predicted, labels=[0, 1]).tolist(),
    }


def train(data: Path, output: Path):
    """Select candidates using development-fold CV only; never evaluate holdout here."""
    frame = read_data(data)
    development, holdout = split_data(frame)
    candidates = {
        "dummy_majority": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(max_iter=2000, C=1.0, random_state=SEED),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=5, min_samples_leaf=3,
            random_state=SEED, n_jobs=1,
        ),
    }
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows = []
    for name, estimator in candidates.items():
        result = cross_validate(
            make_pipeline(estimator), development, development.Survived, cv=folds,
            scoring={"accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy", "roc_auc": "roc_auc"},
            error_score="raise",
        )
        for fold in range(5):
            rows.append({"model": name, "fold": fold + 1, **{
                metric: float(result[f"test_{metric}"][fold])
                for metric in ("accuracy", "balanced_accuracy", "roc_auc")
            }})
    cv = pd.DataFrame(rows)
    means = cv.groupby("model").balanced_accuracy.mean()
    chosen = str(means.idxmax())
    fitted = make_pipeline(candidates[chosen]).fit(development, development.Survived)
    metadata = {
        "selected_model": chosen,
        "selection_metric": "mean development CV balanced accuracy",
        "seed": SEED, "folds": 5,
        "development_rows": len(development), "holdout_rows": len(holdout),
        "training_data_sha256": data_digest(data),
        "development_passenger_ids": sorted(development.PassengerId.tolist()),
        "holdout_passenger_ids": sorted(holdout.PassengerId.tolist()),
        "python": platform.python_version(), "sklearn": sklearn.__version__,
        "pandas": pd.__version__, "numpy": np.__version__,
    }
    output.mkdir(parents=True, exist_ok=True)
    cv.to_csv(output / "cv_results.csv", index=False, float_format="%.8f")
    write_json(output / "model_selection.json", metadata)
    joblib.dump({"pipeline": fitted, "metadata": metadata}, output / "development-model.joblib")
    print(cv.groupby("model")[["accuracy", "balanced_accuracy", "roc_auc"]].agg(["mean", "std"]).round(4))
    print(f"Selected {chosen}; the {len(holdout)} holdout labels were not used in model selection.")
    return metadata


def load_run(data: Path, output: Path):
    """Only load the local model generated by train; joblib is unsafe for untrusted files."""
    bundle = joblib.load(output / "development-model.joblib")
    if data_digest(data) != bundle["metadata"]["training_data_sha256"]:
        raise ValueError("Training data changed; rerun train before evaluating or predicting")
    return bundle, read_data(data)


def evaluate(data: Path, output: Path):
    bundle, frame = load_run(data, output)
    meta, model = bundle["metadata"], bundle["pipeline"]
    holdout = frame[frame.PassengerId.isin(meta["holdout_passenger_ids"])]
    development = frame[frame.PassengerId.isin(meta["development_passenger_ids"])]
    predicted = model.predict(holdout)
    probabilities = model.predict_proba(holdout)[:, 1]
    baseline = DummyClassifier(strategy="most_frequent").fit(development, development.Survived)
    report = {
        "selected_model": meta["selected_model"], "holdout_rows": len(holdout),
        "threshold": 0.5, "positive_class": "Survived=1",
        "model": scores(holdout.Survived, predicted, probabilities),
        "dummy_majority": scores(holdout.Survived, baseline.predict(holdout), baseline.predict_proba(holdout)[:, 1]),
        "note": "One fixed holdout. No threshold tuning on holdout; no Kaggle test labels are available.",
    }
    details = holdout[["PassengerId", "Pclass", "Sex", "Age", "Survived"]].copy()
    details["Predicted"] = predicted
    details["SurvivalProbability"] = probabilities
    details["Correct"] = predicted == holdout.Survived
    details["ErrorType"] = np.where(details.Correct, "correct", np.where(predicted == 1, "false_positive", "false_negative"))
    details.to_csv(output / "holdout_predictions.csv", index=False, float_format="%.8f")
    groups = []
    for column in ("Sex", "Pclass"):
        for value, group in details.groupby(column):
            groups.append({"feature": column, "group": str(value), "n": len(group), "errors": int((~group.Correct).sum()), "accuracy": float(group.Correct.mean())})
    pd.DataFrame(groups).to_csv(output / "subgroup_errors.csv", index=False, float_format="%.8f")
    write_json(output / "metrics.json", report)
    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
    ConfusionMatrixDisplay.from_predictions(holdout.Survived, predicted, display_labels=["Did not survive", "Survived"], ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Held-out predictions | {len(holdout)} passengers")
    fig.savefig(output / "holdout_confusion_matrix.png", dpi=160)
    plt.close(fig)
    print(json.dumps(report, indent=2))
    return report


def predict(data: Path, test: Path, output: Path):
    """Refit the CV-selected specification on all labeled rows for an unlabeled CSV."""
    bundle, frame = load_run(data, output)
    test_frame = read_data(test, labeled=False)
    if set(test_frame.PassengerId) & set(frame.PassengerId):
        raise ValueError("Prediction and training PassengerIds overlap")
    final = clone(bundle["pipeline"]).fit(frame, frame.Survived)
    submission = pd.DataFrame({"PassengerId": test_frame.PassengerId, "Survived": final.predict(test_frame).astype(int)})
    submission.to_csv(output / "submission.csv", index=False)
    write_json(output / "submission_provenance.json", {
        "selected_model": bundle["metadata"]["selected_model"],
        "fit_rows": len(frame), "prediction_rows": len(test_frame),
        "test_data_sha256": data_digest(test),
        "evaluation": "Unlabeled predictions only. No Kaggle score has been obtained.",
    })
    print(f"Wrote {len(submission)} predictions; refit on {len(frame)} labeled passengers. No test accuracy is available.")
    return submission


def eda(data: Path, images: Path):
    """Describe only the development partition; keep modeling holdout out of EDA."""
    development, _ = split_data(read_data(data))
    images.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), layout="constrained")
    for ax, column, title in zip(axes, ["Sex", "Pclass"], ["Passenger sex", "Ticket class"]):
        grouped = development.groupby(column).Survived.agg(["mean", "size"])
        ax.bar(grouped.index.astype(str), grouped["mean"], color=["#2d6a9f", "#58a9a0", "#e4a04b"][:len(grouped)])
        for index, row in enumerate(grouped.itertuples()):
            ax.text(index, row.mean + .025, f"{row.mean:.1%}\nn={row.size}", ha="center", fontsize=10)
        ax.set(title=title, ylabel="Observed survival proportion", ylim=(0, 1))
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Titanic survival patterns | development data only (n={len(development)})", fontsize=13)
    fig.savefig(images / "survival_patterns.png", dpi=160)
    plt.close(fig)
    missing = development[INPUT_COLUMNS].isna().sum().sort_values(ascending=False)
    print(f"Development rows: {len(development)}; survival rate: {development.Survived.mean():.1%}")
    print("Missing values:\n", missing[missing > 0].to_string())
    return development


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["eda", "train", "evaluate", "predict"])
    parser.add_argument("--data", type=Path, default=ROOT / "data/train.csv")
    parser.add_argument("--test", type=Path, default=ROOT / "data/test.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts")
    parser.add_argument("--images", type=Path, default=ROOT / "images")
    args = parser.parse_args()
    if args.command == "eda":
        eda(args.data, args.images)
    elif args.command == "train":
        train(args.data, args.output)
    elif args.command == "evaluate":
        evaluate(args.data, args.output)
    else:
        predict(args.data, args.test, args.output)


if __name__ == "__main__":
    # Persist transform functions as titanic.add_features, not __main__.add_features,
    # so a model produced by the CLI can also be loaded from a notebook.
    from titanic import main as cli_main
    cli_main()
