#!/usr/bin/env python3
"""
Train and cross-validate a **binary (0/1)** classifier on either a CSV or dataset directories.

Dataset-dirs modes & expected layout
------------------------------------
The directory structure is assumed to be:
    <root_path>/
      └─ <subject_name>/
          └─ <task_name>/
              └─ <files>

- If you pass **TWO** root paths: the **first** is treated as **class 0** and the **second** as **class 1**;
  labels are expected to be created by `FeatureExtractor` into a `label` (or `labels`) column.
- If you pass **ONE** root path: this is considered **unlabeled** evaluation data and is **NOT** allowed here
  (use the evaluator script instead).

Optionally filter by specific **tasks** (e.g., `--tasks vowelA vowelE`) that match the `task` column.

Outputs (in results/<name>/<uuid>/):
- predictions_samples.csv   (per-sample predictions, all folds)
- predictions_patients.csv  (per-patient aggregated predictions, all folds)
- metrics_samples.csv       (per-fold sample metrics + mean/std rows)
- metrics_patients.csv      (per-fold patient metrics + mean/std rows)
- best_model.pkl            (sklearn Pipeline: imputer -> [scaler] -> model)
- feature_importance.csv    (if supported by underlying model)
- config.json               (configuration, tuned parameters, feature names, etc.)

Usage examples
--------------
# From a CSV with: subject, task, label/labels/y, + features
python train_model.py \
  --name xgb_vowels \
  --model xgboost \
  --csv csv/hepa_combined.csv \
  --tasks vowelA vowelE \
  --num-folds 10 \
  --tune-params

# From TWO dataset roots (class 0 then class 1)
python train_model.py \
  --name xgb_vowelA \
  --model xgboost \
  --dataset-dirs /data/HC /data/PD \
  --tasks vowelA \
  --num-folds 10 \
  --tune-params
"""

from __future__ import annotations

import os
import uuid
import json
import argparse
from copy import deepcopy
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    roc_auc_score,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.utils.class_weight import compute_class_weight

import xgboost as xgb
import joblib

from feature_extraction import FeatureExtractor
from utils import (
    _save_feature_importance,
    create_dir_if_not_exist,
    features_name_from_roots,
    json_safe,
    write_results_to_csv,
    calculate_results_per_patient,
    RegressorAsClassifier,
)

SEED = 42


class UnifiedCV:
    """
    Cross-validation runner for **binary 0/1 classification**.

    - Uses a Pipeline: [imputer] -> [optional scaler] -> [model]
    - No calibration wrapper.
    - Saves:
        * metrics_samples.csv   (per-fold + mean/std, sample-level)
        * metrics_patients.csv  (per-fold + mean/std, patient-level)
        * predictions_samples.csv
        * predictions_patients.csv
        * best_model.pkl
        * feature_importance.csv (if available)
        * config.json
    """

    def __init__(self, name: str = "experiment"):
        self.extractor = FeatureExtractor()
        self.uuid = str(uuid.uuid4())
        self.name = name
        self.results_dir = os.path.join(os.getcwd(), "results", self.name, self.uuid)
        create_dir_if_not_exist(self.results_dir)
        self.config: Dict[str, Any] = {"params": [], "model": None, "random_seed": SEED}
        self.num_class = 2  # binary only

    # ---------- Model registry & tuner ----------

    def _base_model_and_grid(self, model: str):
        """
        Return (base_estimator, param_grid, default_params) for the *model step*
        (no pipeline prefixes).
        """
        m = model.lower()

        if m == "svm":
            base = SVC(probability=True, class_weight="balanced", random_state=SEED)
            grid = {
                "kernel": ["rbf"],
                "C": [0.01, 0.1, 1, 10, 100],
                "gamma": ["scale", 1, 0.1, 0.01, 0.001, 0.0001],
            }
            default = {"kernel": "rbf", "C": 1.0, "gamma": "scale"}

        elif m == "random_forest":
            base = RandomForestClassifier(
                n_estimators=300,
                n_jobs=-1,
                random_state=SEED,
                class_weight="balanced",
            )
            grid = {
                "n_estimators": [200, 300, 500],
                "max_depth": [None, 10, 20, 40],
                "min_samples_split": [2, 5, 10],
            }
            default = {"n_estimators": 300, "max_depth": None, "min_samples_split": 2}

        elif m == "naive_bayes":
            base = GaussianNB()
            grid = {"var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6]}
            default = {"var_smoothing": 1e-9}

        elif m == "logreg":
            base = LogisticRegression(
                max_iter=1000,
                solver="liblinear",
                class_weight="balanced",
            )
            grid = {"C": [0.01, 0.1, 1, 10, 100], "penalty": ["l1", "l2"]}
            default = {"C": 1.0, "penalty": "l2"}

        elif m == "linreg":  # Linear Regression treated as classifier via logistic mapping
            base = RegressorAsClassifier(LinearRegression())
            grid = {}
            default = {}

        elif m == "adaboost":
            base = AdaBoostClassifier(
                n_estimators=300,
                learning_rate=0.5,
                random_state=SEED,
            )
            grid = {
                "n_estimators": [100, 200, 300, 500],
                "learning_rate": [0.05, 0.1, 0.5, 1.0],
            }
            default = {"n_estimators": 300, "learning_rate": 0.5}

        elif m == "xgboost":
            base = xgb.XGBClassifier(
                n_estimators=400,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                n_jobs=-1,
                random_state=SEED,
                scale_pos_weight=1.0,
            )
            grid = {
                "n_estimators": [300, 400, 600],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.03, 0.05, 0.1],
                "subsample": [0.7, 0.8, 1.0],
                "colsample_bytree": [0.7, 0.8, 1.0],
            }
            default = {
                "n_estimators": 400,
                "max_depth": 5,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            }

        else:
            raise ValueError(
                "Unsupported model. Use 'svm', 'random_forest', 'naive_bayes', "
                "'logreg', 'linreg', 'adaboost', or 'xgboost'."
            )

        return base, grid, default

    def _build_pipeline_and_grid(self, model: str, scale: bool) -> tuple[Pipeline, Dict[str, list], Dict[str, Any]]:
        """
        Construct Pipeline([imputer, scaler?, model]) and grid with 'model__' prefixes.
        """
        base, grid, default = self._base_model_and_grid(model)

        steps = [("imputer", SimpleImputer(strategy="mean"))]
        if scale:
            steps.append(("scaler", StandardScaler()))
        steps.append(("model", base))

        pipe = Pipeline(steps)

        # Prefix grid & defaults with model__
        pref_grid = {f"model__{k}": v for k, v in grid.items()}
        pref_default = {f"model__{k}": v for k, v in default.items()}

        return pipe, pref_grid, pref_default

    def _tune(
        self,
        pipe: Pipeline,
        grid: Dict[str, list],
        X_tr: pd.DataFrame,
        y_tr: np.ndarray,
        grp_tr: pd.Series,
        n_splits: int = 5,
    ):
        """
        Hyperparameter tuning on the *pipeline* using ROC AUC.
        Returns (best_estimator (fitted pipeline), best_params_dict).
        If grid empty → returns (pipe fitted with defaults, empty dict).
        """
        if not grid or sum(len(v) for v in grid.values()) == 0:
            pipe.fit(X_tr, y_tr)
            return pipe, {}

        def _auc_callable(est, X, y):
            proba = est.predict_proba(X)[:, 1]
            return roc_auc_score(y, proba)

        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        gs = GridSearchCV(pipe, grid, scoring=_auc_callable, refit=True, cv=cv, n_jobs=-1, verbose=0)
        gs.fit(X_tr, y_tr, groups=grp_tr)

        best_est = gs.best_estimator_
        best_params = json_safe(gs.best_params_)

        return best_est, best_params

    # ---------- Main CV ----------

    def k_fold_cv(
        self,
        *,
        model: str = "svm",
        data_path: Optional[str] = None,
        extract_features: bool = False,
        dataset_paths: Optional[List[str]] = None,
        save_features: bool = True,
        save_name: str = "features.csv",
        tasks: Optional[List[str]] = None,
        num_folds: int = 10,
        scale: bool = True,
        tune_params: bool = False,
        save_best_model: bool = True,
    ):
        # ---- load / build data ----
        if data_path:
            data = pd.read_csv(data_path)
        elif extract_features:
            if not dataset_paths:
                raise ValueError("dataset_paths must be provided when extract_features=True")

            if len(dataset_paths) == 1:
                raise ValueError(
                    "Trainer requires labeled data. "
                    "You passed ONE dataset root (unlabeled). "
                    "Use the evaluator script or pass TWO dataset roots (class 0 and class 1)."
                )

            if len(dataset_paths) != 2:
                raise ValueError(
                    f"--dataset-dirs expects exactly 2 paths (class 0 and class 1); got {len(dataset_paths)}."
                )

            # build a default features filename from roots if user didn't specify
            if save_features:
                # if caller passed a custom save_name, keep it; otherwise build from roots
                if save_name is None or save_name == "":
                    save_name_local = features_name_from_roots(dataset_paths, suffix="features", eval_mode=False)
                else:
                    save_name_local = save_name
            else:
                save_name_local = "features_tmp.csv"

            data = self.extractor.extract_features(
                dataset_paths,
                save=save_features,
                save_name=save_name_local,
            )
        else:
            raise ValueError("Either --csv or --dataset-dirs (with extract_features=True) must be provided.")

        # ---- basic validation ----

        if "subject" not in data.columns:
            raise ValueError("Data must contain 'subject' column.")
        if "task" not in data.columns:
            raise ValueError("Data must contain 'task' column.")

        # label column: allow 'labels' or 'label' or 'y'
        label_col = None
        for cand in ["labels", "label", "y"]:
            if cand in data.columns:
                label_col = cand
                break
        if label_col is None:
            raise ValueError("No label column found (expected one of: 'labels', 'label', 'y').")

        # ---------------------------------------------------------
        # Validate and filter by task (if provided)
        # ---------------------------------------------------------
        available_tasks = sorted(data["task"].unique().tolist())

        if tasks is not None and len(tasks) > 0:
            missing = [t for t in tasks if t not in available_tasks]
            if missing:
                raise ValueError(
                    f"Requested tasks {missing} not found in dataset. " f"Available tasks are: {available_tasks}"
                )
            data = data[data["task"].isin(tasks)].reset_index(drop=True)
            selected_tasks = tasks
        else:
            selected_tasks = available_tasks

        self.config["available_tasks"] = available_tasks
        self.config["selected_tasks"] = selected_tasks
        self.config["model"] = model
        self.config["label_column"] = label_col

        # extract X/y/groups
        X = data.drop(columns=[c for c in ["subject", "task", label_col] if c in data.columns])
        y = data[label_col].values
        groups = data["subject"]

        # validate binary labels 0/1
        uniq = np.unique(y)
        if len(uniq) != 2 or not np.array_equal(np.sort(uniq), np.array([0, 1])):
            raise ValueError(
                f"Trainer supports only binary labels 0/1. Found labels: {uniq}. " f"Please map your labels to 0 and 1."
            )

        # Class weighting from label distribution
        classes = np.array([0, 1])
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
        class_weights = {int(c): float(w) for c, w in zip(classes, weights)}
        # XGBoost-style ratio (neg:pos)
        n_neg = np.sum(y == 0)
        n_pos = np.sum(y == 1)
        scale_pos_weight = n_neg / n_pos

        # ---- per-fold collectors ----
        per_fold_metrics_samples = []  # sample-level metrics
        all_preds_rows = []  # per-sample predictions (all folds)
        best_model = None
        best_auc = -np.inf
        best_model_params = None

        # build pipeline and its grid
        pipe_template, grid, default_params = self._build_pipeline_and_grid(model, scale=scale)

        splits = StratifiedGroupKFold(num_folds, shuffle=True, random_state=SEED).split(X, y, groups)
        for fold_idx, (train_index, val_index) in enumerate(splits):
            print(f"\nFold {fold_idx + 1}/{num_folds}")

            xtr, xvl = X.iloc[train_index], X.iloc[val_index]
            ytr, yvl = y[train_index], y[val_index]
            grp_tr, grp_vl = groups.iloc[train_index], groups.iloc[val_index]

            # fresh pipeline for this fold
            pipe_fold = deepcopy(pipe_template)

            model_step = pipe_fold.named_steps.get("model", None)

            if model_step is not None:
                # Generic sklearn class_weight
                if hasattr(model_step, "class_weight"):
                    model_step.set_params(class_weight=class_weights)

                # XGBoost-specific weighting
                if isinstance(model_step, xgb.XGBClassifier):
                    model_step.set_params(scale_pos_weight=scale_pos_weight)

            # hyperparameter tuning
            if tune_params:
                est_for_fit, tuned = self._tune(pipe_fold, grid, xtr, ytr, grp_tr)
            else:
                est_for_fit = pipe_fold
                est_for_fit.fit(xtr, ytr)
                tuned = default_params

            # record tuned params in config (one entry per fold)
            self.config["params"].append({"fold": fold_idx, **tuned})

            # predictions
            probs = est_for_fit.predict_proba(xvl)[:, 1]
            pred_cls = (probs >= 0.5).astype(int)

            # --- metrics (per-fold, sample-level) ---
            fold_acc = accuracy_score(yvl, pred_cls)
            fold_sens = recall_score(yvl, pred_cls)
            fold_spec = recall_score(1 - yvl, 1 - pred_cls)
            try:
                fold_auc = roc_auc_score(yvl, probs)
            except ValueError:
                fold_auc = np.nan

            metrics_row = {
                "fold": str(fold_idx),
                "accuracy": fold_acc,
                "sensitivity": fold_sens,
                "specificity": fold_spec,
                "auc": fold_auc,
            }
            per_fold_metrics_samples.append(metrics_row)

            # --- store per-sample preds ---
            for subj, p, b, t in zip(grp_vl, probs, pred_cls, yvl):
                all_preds_rows.append(
                    {
                        "fold": str(fold_idx),
                        "subject": subj,
                        "predicted": float(p),
                        "predicted_binary": int(b),
                        "true": int(t),
                        "misclassified": int(b != t),
                    }
                )

            # --- best model selection (by AUC) ---
            auc_for_selection = fold_auc if np.isfinite(fold_auc) else -np.inf
            if auc_for_selection > best_auc:
                best_auc = auc_for_selection
                best_model = deepcopy(est_for_fit)
                best_model_params = tuned

        # ----- after all folds: metrics & predictions -----

        # sample-level metrics (+ mean/std)
        metrics_df_samples = pd.DataFrame(per_fold_metrics_samples)
        num_cols = [c for c in metrics_df_samples.columns if c != "fold"]
        means = metrics_df_samples[num_cols].mean(axis=0)
        stds = metrics_df_samples[num_cols].std(axis=0)

        summary_samples = pd.DataFrame(
            [
                {"fold": "mean", **means.to_dict()},
                {"fold": "std", **stds.to_dict()},
            ]
        )
        metrics_samples_all = pd.concat([metrics_df_samples, summary_samples], axis=0, ignore_index=True)
        write_results_to_csv(self.results_dir, metrics_samples_all, "metrics_samples")

        # all per-sample predictions
        all_preds = pd.DataFrame(all_preds_rows)
        all_preds["fold"] = all_preds["fold"].astype(str)
        all_preds.to_csv(os.path.join(self.results_dir, "predictions_samples.csv"), index=False)

        # per-patient metrics & predictions
        # 1) per-patient per-fold + mean/std
        val_final_perp_summary, val_final_perp_all_folds = self.prepare_per_fold_summary_per_patient(all_preds)
        metrics_patients_all = pd.concat([val_final_perp_all_folds, val_final_perp_summary], axis=0, ignore_index=True)
        write_results_to_csv(self.results_dir, metrics_patients_all, "metrics_patients")

        # 2) per-patient aggregated over ALL folds (one row per subject)
        preds_per_patient, _final_per_patient = calculate_results_per_patient(all_preds, self.num_class)
        preds_per_patient.to_csv(os.path.join(self.results_dir, "predictions_patients.csv"), index=False)

        # ----- best model & config & importance -----

        if save_best_model and best_model is not None:
            joblib.dump(best_model, os.path.join(self.results_dir, f"{model}_best_model.pkl"))

        # config dump
        self.config["num_features"] = X.shape[1]
        self.config["features"] = list(X.columns)
        self.config["best_fold_auc"] = float(best_auc)
        self.config["best_model_params"] = best_model_params
        with open(os.path.join(self.results_dir, "config.json"), "w") as f:
            json.dump(self.config, f, indent=2)

        # save feature importance (if available)
        _save_feature_importance(best_model, X.columns, self.results_dir)

        return metrics_df_samples  # per-fold sample-level metrics (without mean/std)

    def prepare_per_fold_summary_per_patient(self, all_preds: pd.DataFrame):
        """
        For each fold:
            - aggregate per-patient metrics via calculate_results_per_patient
        Then compute mean/std across folds.
        """
        perp_metrics_rows = []

        for f in sorted(all_preds["fold"].unique(), key=lambda x: int(x)):
            fold_df = all_preds[all_preds["fold"] == f].copy()
            preds_per_patient_f, final_per_patient_f = calculate_results_per_patient(fold_df, self.num_class)

            metrics_row_f = {"fold": f, **final_per_patient_f.iloc[0].to_dict()}
            perp_metrics_rows.append(metrics_row_f)

        val_final_perp_all_folds = pd.DataFrame(perp_metrics_rows)

        num_cols_perp = val_final_perp_all_folds.select_dtypes(include=[np.number]).columns
        means_perp = val_final_perp_all_folds[num_cols_perp].mean(axis=0)
        stds_perp = val_final_perp_all_folds[num_cols_perp].std(axis=0)

        val_final_perp_summary = pd.DataFrame(
            [
                {"fold": "mean", **means_perp.to_dict()},
                {"fold": "std", **stds_perp.to_dict()},
            ]
        )

        cols_perp = ["fold"] + [c for c in val_final_perp_summary.columns if c != "fold"]
        val_final_perp_summary = val_final_perp_summary[cols_perp]

        return val_final_perp_summary, val_final_perp_all_folds


# ---- CLI ----


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and cross-validate a binary (0/1) classifier on CSV or dataset directories."
    )

    parser.add_argument(
        "--name",
        type=str,
        default="experiment",
        help="Name of the experiment (used in results/<name>/<uuid>).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="xgboost",
        choices=["svm", "random_forest", "naive_bayes", "logreg", "linreg", "adaboost", "xgboost"],
        help="Model to train.",
    )

    # data sources
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to input CSV with 'subject', 'task', label/labels/y, and feature columns.",
    )
    parser.add_argument(
        "--dataset-dirs",
        nargs="*",
        default=None,
        help=(
            "Two dataset roots following layout root/subject/task/files. "
            "First path = class 0, second path = class 1. "
            "One path (unlabeled) is NOT allowed here; use evaluator instead."
        ),
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help="Optional list of task names to filter by (e.g., --tasks vowelA vowelE). Must match 'task' column.",
    )

    parser.add_argument(
        "--num-folds",
        type=int,
        default=10,
        help="Number of StratifiedGroupKFold folds.",
    )
    parser.add_argument(
        "--no-scale",
        action="store_true",
        help="Disable StandardScaler in the pipeline (default is to scale).",
    )
    parser.add_argument(
        "--tune-params",
        action="store_true",
        help="Enable hyperparameter tuning with GridSearchCV.",
    )
    parser.add_argument(
        "--no-save-best-model",
        action="store_true",
        help="Do not save best_model.pkl.",
    )
    parser.add_argument(
        "--features-save-name",
        type=str,
        default="features.csv",
        help="If extracting features from dataset-dirs, save them under this CSV name.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if (args.csv is None) == (args.dataset_dirs is None):
        raise ValueError("You must provide exactly one of --csv or --dataset-dirs.")

    runner = UnifiedCV(name=f"{args.model}_{'_'.join(args.task)}" if args.tasks is not None else f"{args.model}")

    if args.csv is not None:
        runner.k_fold_cv(
            model=args.model,
            data_path=args.csv,
            extract_features=False,
            dataset_paths=None,
            save_features=False,
            save_name=args.features_save_name,
            tasks=args.tasks,
            num_folds=args.num_folds,
            scale=not args.no_scale,
            tune_params=args.tune_params,
            save_best_model=not args.no_save_best_model,
        )
    else:
        runner.k_fold_cv(
            model=args.model,
            data_path=None,
            extract_features=True,
            dataset_paths=args.dataset_dirs,
            save_features=True,
            save_name=args.features_save_name,
            tasks=args.tasks,
            num_folds=args.num_folds,
            scale=not args.no_scale,
            tune_params=args.tune_params,
            save_best_model=not args.no_save_best_model,
        )


if __name__ == "__main__":
    main()
