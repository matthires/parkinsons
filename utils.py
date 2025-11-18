import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    recall_score,
    precision_score,
    roc_auc_score,
)

import os


def fpr(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp / (fp + tn) if (fp + tn) > 0 else 0.0


def fnr(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fn / (fn + tp) if (fn + tp) > 0 else 0.0


def npv(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fn) if (tn + fn) > 0 else 0.0


def calculate_results_per_patient(results: pd.DataFrame, num_class: int = 2):
    """
    Aggregate probabilities per patient and compute binary metrics.

    Parameters
    ----------
    results : pd.DataFrame
        Must contain columns:
            - 'subject'
            - 'predicted'  (per-sample probability for class 1)
            - 'true'       (ground truth label, 0 or 1)
            - (optional) 'predicted_binary', 'misclassified' are ignored and recomputed
    num_class : int, optional
        Kept for backwards compatibility; must be 2 (binary).

    Returns
    -------
    results_per_patient : pd.DataFrame
        One row per subject with:
            - subject
            - predicted          (mean probability for class 1, or new_preds)
            - predicted_binary   (thresholded at 0.5)
            - true               (first label seen for that subject)
            - misclassified      (predicted_binary != true)
    results_df : pd.DataFrame
        Single-row DataFrame with summary metrics:
            - accuracy
            - specificity
            - sensitivity
            - auc                 (using probabilities)
            - FPR, FNR, PPV, NPV
            - conf_matrix         (stringified confusion matrix)
    """
    if num_class != 2:
        raise ValueError(f"calculate_results_per_patient now supports only binary (num_class=2). Got {num_class}.")

    required_cols = {"subject", "predicted", "true"}
    missing = required_cols - set(results.columns)
    if missing:
        raise ValueError(f"Missing required columns in results: {missing}")

    # Aggregate per subject: mean probability and first true label
    results_per_patient = (
        results.groupby("subject")
        .agg(
            predicted=("predicted", "mean"),
            true=("true", "first"),
        )
        .reset_index()
    )

    # Derive binary label from probability
    results_per_patient["predicted_binary"] = (results_per_patient["predicted"] >= 0.5).astype(int)
    results_per_patient["misclassified"] = (
        results_per_patient["predicted_binary"] != results_per_patient["true"]
    ).astype(int)

    # Convenience aliases
    y_true = results_per_patient["true"].values
    y_pred = results_per_patient["predicted_binary"].values
    y_score = results_per_patient["predicted"].values

    # Confusion matrix & basic metrics
    conf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    accuracy = accuracy_score(y_true, y_pred)
    specificity = recall_score(y_true, y_pred, pos_label=0)
    sensitivity = recall_score(y_true, y_pred, pos_label=1)

    try:
        roc_auc = roc_auc_score(y_true, y_score)
    except ValueError:
        # e.g. only one class present in y_true
        roc_auc = np.nan

    # Summary metrics
    results_df = pd.DataFrame(
        {
            "accuracy": [accuracy],
            "specificity": [specificity],
            "sensitivity": [sensitivity],
            "auc": [roc_auc],
            "FPR": [fpr(y_true, y_pred)],
            "FNR": [fnr(y_true, y_pred)],
            "PPV": [precision_score(y_true, y_pred, pos_label=1, zero_division=0)],
            "NPV": [npv(y_true, y_pred)],
            "conf_matrix": [np.array_str(conf_matrix)],
        }
    )

    return results_per_patient, results_df


def write_results_to_csv(path, results, label):
    """
    Writes evaluation results to a CSV file.

    Args:
    path (str): The directory path where the CSV file will be saved.
    results (pd.DataFrame or dict): The evaluation results. If a DataFrame,
                                    it is directly written to CSV. If a dictionary,
                                    it is converted to a DataFrame and then written.
    label (str): A label to include in the CSV file name.

    Returns:
    None
    """
    if isinstance(results, pd.DataFrame):
        results = results.reset_index(drop=True)
        results = results.map(round_percentage)
        results.to_csv(os.path.join(path, f"{label}_results.csv"), index=False)
    else:
        df = pd.DataFrame()
        df["metric"] = results.keys()
        df["value"] = results.values()
        df.to_csv(os.path.join(path, label + f"{label}_results.csv"), index=False)


def create_dir_if_not_exist(path):
    """
    Creates directory if it doesn't exist

    Args:
        path (str): Path of directory to create
    """
    try:
        os.makedirs(path)
    except FileExistsError:
        pass


def round_percentage(value):
    """
    Rounds a numerical value to a percentage with four decimal places.

    Args:
    value (int, float): The numerical value to be converted to a percentage.

    Returns:
    float: The value as a percentage rounded to four decimal places.
    If the input is not a numerical type, it is returned unchanged.
    """

    if isinstance(value, (int, float)):
        return round(value * 100, 4)
    else:
        return value


def json_safe(o):
    """Convert sklearn parameters into something JSON serializable."""
    if o is None:
        return None
    if isinstance(o, (str, int, float, bool)):
        return o
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (list, tuple)):
        return [json_safe(x) for x in o]
    if isinstance(o, dict):
        return {json_safe(k): json_safe(v) for k, v in o.items()}
    # sklearn objects, classes, functions → convert to string
    return str(o)


def features_name_from_roots(roots, suffix: str = "features", eval_mode: bool = False) -> str:
    """
    Build a feature CSV name from one or more dataset root paths.

    examples:
        roots = ["/data/HC", "/data/PD"] -> "HC_PD_features.csv"
        roots = ["/data/ALL_SUBJECTS"]   -> "ALL_SUBJECTS_features_eval.csv" (if eval_mode=True)
    """
    if isinstance(roots, str):
        roots = [roots]

    base_names = [os.path.basename(os.path.normpath(p)) for p in roots]
    root_tag = "_".join(base_names)

    if eval_mode:
        return f"{root_tag}_{suffix}_eval.csv"
    else:
        return f"{root_tag}_{suffix}.csv"


def _save_feature_importance(best_model, feature_names, results_dir):
    """
    Try to extract feature importances from the underlying classifier.
    Supports:
        - tree-based models with `feature_importances_`
        - linear models with `coef_`
    Works with a Pipeline.
    """
    if best_model is None:
        return

    base = best_model

    # base should be a Pipeline
    if not hasattr(base, "named_steps"):
        return

    model_step = base.named_steps.get("model", None)
    if model_step is None:
        return

    importances = None

    if hasattr(model_step, "feature_importances_"):
        importances = np.asarray(model_step.feature_importances_)
    elif hasattr(model_step, "coef_"):
        coef = np.asarray(model_step.coef_)
        if coef.ndim == 1:
            importances = np.abs(coef)
        else:
            importances = np.mean(np.abs(coef), axis=0)

    if importances is None:
        return

    # Prefer the model's own feature_names_in_ if available
    if hasattr(base, "feature_names_in_"):
        fn = list(base.feature_names_in_)
    else:
        fn = list(feature_names)

    # --- length sanity check & alignment ---
    n_imp = len(importances)
    n_feat = len(fn)

    if n_imp != n_feat:
        # align to the common prefix and warn in stdout
        n_common = min(n_imp, n_feat)
        print(
            f"[WARN] feature_importance: length mismatch "
            f"(importances={n_imp}, features={n_feat}). "
            f"Truncating to first {n_common} entries."
        )
        importances = importances[:n_common]
        fn = fn[:n_common]

    fi_df = pd.DataFrame({"feature": fn, "importance": importances})
    fi_df.sort_values("importance", ascending=False, inplace=True)
    fi_df.to_csv(os.path.join(results_dir, "feature_importance.csv"), index=False)


class RegressorAsClassifier:
    """Wrap a regressor (e.g., LinearRegression) to expose a classifier-like API with
    predict_proba via a logistic mapping. Intended only for binary labels {0,1}.
    """

    def __init__(self, regressor):
        self.regressor = regressor

    def fit(self, X, y):
        self.regressor.fit(X, y)
        return self

    def decision_function(self, X):
        # raw scores from regression
        return self.regressor.predict(X)

    def predict_proba(self, X):
        z = self.decision_function(X)
        # logistic mapping -> probability
        p = 1 / (1 + np.exp(-z))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.c_[1 - p, p]

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def get_params(self, deep=True):
        return {"regressor": self.regressor}

    def set_params(self, **params):
        if "regressor" in params:
            self.regressor = params["regressor"]
        return self


def _softmax(z):
    z = np.asarray(z)
    z = z - np.max(z, axis=1, keepdims=True)
    ez = np.exp(z)
    sm = ez / np.sum(ez, axis=1, keepdims=True)
    return sm


class ProbaAdapter:
    """Adapter that ensures predict_proba exists by falling back to decision_function.
    - Binary: sigmoid(decision_function)
    - Multiclass: softmax(decision_function)
    """

    def __init__(self, base_estimator, n_classes: int):
        self.base = base_estimator
        self.n_classes = n_classes

    def fit(self, X, y):
        self.base.fit(X, y)
        return self

    def predict(self, X):
        if hasattr(self.base, "predict"):
            return self.base.predict(X)
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def decision_function(self, X):
        if hasattr(self.base, "decision_function"):
            return self.base.decision_function(X)
        raise AttributeError("Base estimator has neither predict_proba nor decision_function")

    def predict_proba(self, X):
        if hasattr(self.base, "predict_proba"):
            return self.base.predict_proba(X)
        scores = self.decision_function(X)
        scores = np.asarray(scores)
        if scores.ndim == 1 or scores.shape[1] == 1:
            p = 1 / (1 + np.exp(-scores))
            p = np.clip(p, 1e-6, 1 - 1e-6)
            return np.c_[1 - p, p]
        return _softmax(scores)

    def get_params(self, deep=True):
        return {"base_estimator": self.base, "n_classes": self.n_classes}

    def set_params(self, **params):
        if "base_estimator" in params:
            self.base = params["base_estimator"]
        if "n_classes" in params:
            self.n_classes = params["n_classes"]
        if hasattr(self.base, "set_params"):
            supported = {k: v for k, v in params.items() if k in self.base.get_params()}
            if supported:
                self.base.set_params(**supported)
        return self
