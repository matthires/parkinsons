#!/usr/bin/env python3
"""
Evaluate a saved **binary** model (`best_model.pkl`) on either a CSV or **dataset directories**.

Dataset-dirs modes & expected layout
------------------------------------
The directory structure is assumed to be:
    <root_path>/
      └─ <subject_name>/
          └─ <task_name>/
              └─ <files>

- If you pass **ONE** root path: evaluation is **unlabeled** (no ground-truth labels present).
- If you pass **TWO** root paths: the **first** is treated as **class 0** and the **second** as **class 1**;
  labels are expected to be available in the extracted dataframe (via FeatureExtractor).

Optionally filter by specific **tasks** (e.g., `--tasks vowelA vowelE`) that match the `task` column.

Outputs:
- predictions_samples_eval.csv  (per-sample predictions)
- predictions_patients_eval.csv (per-patient aggregated predictions)
- If labels present:
    * 'true' and 'misclassified' columns
- feature_importance_eval.csv   (if supported by the model)

Usage examples
--------------
# From a CSV that already has columns: subject, task, (optional) labels, + features
python evaluate_model.py \
  --model-path results/xgboost/<uuid>/best_model.pkl \
  --csv csv/hepa_combined.csv \
  --tasks vowelA vowelE

# From **one** dataset root (unlabeled). Evaluates and saves predictions only.
python evaluate_model.py \
  --model-path results/xgboost/<uuid>/best_model.pkl \
  --dataset-dirs /data/ALL_SUBJECTS \
  --tasks vowelA

# From **two** dataset roots (class 0 then class 1). Labels expected to be inferred by the extractor.
python evaluate_model.py \
  --model-path results/xgboost/<uuid>/best_model.pkl \
  --dataset-dirs /data/CLASS0 /data/CLASS1 \
  --tasks vowelA
"""

from __future__ import annotations
import os
import argparse
import uuid
import numpy as np
import pandas as pd
import joblib

from feature_extraction import FeatureExtractor
from utils import ProbaAdapter, _save_feature_importance, features_name_from_roots


def _ensure_binary_optional(y: np.ndarray | None):
    if y is None:
        return None
    uniq = np.unique(y)
    if len(uniq) != 2:
        raise ValueError(f"Binary labels required for evaluation metrics; got {len(uniq)} unique labels: {uniq}")
    mapping = {uniq[0]: 0, uniq[1]: 1}
    return np.vectorize(mapping.get)(y)


def main():
    p = argparse.ArgumentParser(
        description=(
            "Evaluate a saved binary model on CSV or dataset dirs."
            "Dataset-dirs: pass ONE root (unlabeled) or TWO roots (class0 first, class1 second)."
        )
    )
    p.add_argument("--model-path", required=True, help="Path to best_model.pkl")
    p.add_argument(
        "--csv",
        default=None,
        help=(
            "Path to input CSV with columns: subject, task, (optional) labels, + features."
            "If provided, evaluation ignores --dataset-dirs."
        ),
    )
    p.add_argument(
        "--dataset-dirs",
        nargs="*",
        default=None,
        help=(
            "One or two dataset roots following layout: root/subject/task/files."
            "If ONE path is provided -> unlabeled evaluation."
            "If TWO paths are provided -> first is class 0, second is class 1 (labels expected)."
            "Example (two paths): --dataset-dirs /data/HC /data/PD"
        ),
    )
    p.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Optional list of task names to filter by (e.g., --tasks vowelA vowelE). Must match 'task' column.",
    )
    p.add_argument("--save-dir", default=None, help="Directory to write evaluation CSVs (default: ./eval_results)")

    args = p.parse_args()

    save_dir = args.save_dir or os.path.join(os.getcwd(), "eval_results", str(uuid.uuid4()))
    os.makedirs(save_dir, exist_ok=True)

    # Load data
    if args.csv is not None:
        data = pd.read_csv(args.csv)
    else:
        # dataset dirs path mode
        if not args.dataset_dirs or len(args.dataset_dirs) == 0:
            raise ValueError("Provide either --csv or --dataset-dirs (one or two paths).")
        if len(args.dataset_dirs) not in (1, 2):
            raise ValueError(f"--dataset-dirs expects 1 (unlabeled) or 2 (labeled) paths; got {len(args.dataset_dirs)}")

        # Build evaluation features filename from root names
        feat_name = features_name_from_roots(args.dataset_dirs, suffix="features", eval_mode=True)

        # Let FeatureExtractor handle reading + optional labeling by root position
        data = FeatureExtractor().extract_features(
            args.dataset_dirs,
            save=True,
            save_name=feat_name,
        )

    # Validate basic columns
    if "subject" not in data.columns:
        raise ValueError("Data must contain 'subject' column.")
    if "task" not in data.columns:
        raise ValueError("Data must contain 'task' column.")

    # Task filtering / validation
    available_tasks = sorted(data["task"].unique().tolist())
    if args.tasks is not None and len(args.tasks) > 0:
        missing = [t for t in args.tasks if t not in available_tasks]
        if missing:
            raise ValueError(
                f"Requested tasks {missing} not found in dataset. " f"Available tasks are: {available_tasks}"
            )
        data = data[data["task"].isin(args.tasks)].reset_index(drop=True)

    # labels optional
    label_col = None
    for cand in ["labels", "label", "y"]:
        if cand in data.columns:
            label_col = cand
            break

    y = data[label_col].values if label_col is not None else None

    # If passed ONE dataset root, we treat it as unlabeled regardless of any stray column
    if args.csv is None and args.dataset_dirs and len(args.dataset_dirs) == 1:
        y = None

    y = _ensure_binary_optional(y) if y is not None else None

    X = data.drop(columns=[c for c in ["subject", "labels", "task"] if c in data.columns])
    subj = data["subject"].values

    # load model
    model = joblib.load(args.model_path)
    est = getattr(model, "base_estimator", model)
    if not hasattr(est, "predict_proba") and hasattr(est, "decision_function"):
        model = ProbaAdapter(est, n_classes=2)

    # Save feature importance (if possible)
    _save_feature_importance(model, X.columns, save_dir)

    # Align evaluation features with training features
    expected_features = None

    if hasattr(model, "feature_names_in_"):
        expected_features = list(model.feature_names_in_)

    # If model is a wrapper around a pipeline, try unwrapping
    if expected_features is None and hasattr(model, "base_estimator"):
        base = model.base_estimator
        if hasattr(base, "feature_names_in_"):
            expected_features = list(base.feature_names_in_)

    if expected_features is not None:
        current_features = list(X.columns)

        missing = sorted(set(expected_features) - set(current_features))
        extra = sorted(set(current_features) - set(expected_features))

        if missing:
            raise ValueError(
                "Evaluation data is missing features the model was trained on.\n" f"Missing features: {missing}"
            )
        if extra:
            print(
                "Warning: evaluation data has extra features not used during training. "
                "They will be ignored.\n"
                f"Extra features: {extra}"
            )

        # Reindex to ensure correct order and drop any extras
        X = X.reindex(columns=expected_features)

    probs = model.predict_proba(X)[:, 1]
    pred_cls = (probs >= 0.5).astype(int)

    # per-sample CSV
    out_s = pd.DataFrame(
        {
            "subject": subj,
            "predicted": probs,
            "predicted_binary": pred_cls,
        }
    )
    if y is not None:
        out_s["true"] = y
        out_s["misclassified"] = (pred_cls != y).astype(int)
    out_s.to_csv(os.path.join(save_dir, "predictions_samples_eval.csv"), index=False)

    # per-patient aggregation
    df = out_s.copy()
    if y is not None:
        agg = df.groupby("subject").agg(
            predicted=("predicted", "mean"), predicted_binary=("predicted_binary", "mean"), true=("true", "first")
        )
        agg["predicted_binary"] = (agg["predicted"] >= 0.5).astype(int)
        agg["misclassified"] = (agg["predicted_binary"] != agg["true"]).astype(int)
    else:
        agg = df.groupby("subject").agg(predicted=("predicted", "mean"), predicted_binary=("predicted_binary", "mean"))
        agg["predicted_binary"] = (agg["predicted"] >= 0.5).astype(int)

    agg.reset_index().to_csv(os.path.join(save_dir, "predictions_patients_eval.csv"), index=False)
    print("Saved eval CSVs to:", save_dir)


if __name__ == "__main__":
    main()
