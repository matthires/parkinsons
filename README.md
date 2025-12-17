# Binary Voice-based Parkinson's Disease Classification Pipeline

This repository provides a complete pipeline for **binary classification of PD voice data**, including:

- audio/voice **feature extraction**
- **cross-validated model training**
- **patient-level aggregation**
- **model evaluation on new data**

The pipeline is designed to avoid subject leakage by using **grouped cross-validation** and supports both **CSV-based features** and **raw audio datasets**.

---

## Repository Structure

```
.
├── train_model.py           # Main training & cross-validation entrypoint
├── evaluate_model.py        # Evaluate a trained model on new data
├── feature_extraction.py    # Audio feature extraction logic
├── utils.py                 # Helper utilities (metrics, aggregation, saving)
├── data/                    # Directory containing csv data files
├── models/                  # Directory containing model checkpoints
├── eval_results/            # Auto-generated testing experiment outputs
└── results/                 # Auto-generated training experiment outputs
```

---

## Core Scripts

### `train_model.py`

Trains a binary classifier using **StratifiedGroupKFold cross-validation** (grouped by subject).
It optionally performs **hyperparameter tuning**, aggregates predictions per subject, and exports the best model.

**Key features**
- Supports CSV input **or** raw dataset directories
- Subject-level grouping to prevent data leakage
- Optional task filtering
- Automatic metrics + artifact saving
- Model export as a single `sklearn` pipeline

---

### `evaluate_model.py`

Loads a previously trained model (`best_model.pkl`) and evaluates it on:

- a CSV with features, or
- raw audio dataset directories (features extracted automatically)

Supports both **labeled** and **unlabeled** evaluation.

---

## Data Input Options

### Option 1: CSV Input

Your CSV must contain:

| Column | Description |
|------|-------------|
| `subject` | Subject / patient ID (used for grouping) |
| `task` | Task name (optional filtering) |
| `label` / `labels` / `y` | Binary label (0 or 1) |
| feature columns | All remaining numeric columns |

---

### Option 2: Dataset Directories

Expected structure:

```
<root_path>/
└── <subject_id>/
    └── <task_name>/
        └── audio_files.wav
```

#### Training
You must pass **two dataset roots**:
1. Class 0 samples
2. Class 1 samples

#### Evaluation
- **Two roots** → labeled evaluation
- **One root** → unlabeled evaluation

---

## Training the Model

### Train from a CSV

```bash
python train_model.py   --name xgb_vowels   --model xgboost   --csv data/features.csv   --tasks vowelA vowelE   --num-folds 10   --tune-params
```

---

### Train from Dataset Directories

```bash
python train_model.py   --name xgb_vowelA   --model xgboost   --dataset-dirs /data/CLASS0 /data/CLASS1   --tasks vowelA   --num-folds 10
```

---

### Important Training Arguments

| Argument | Description |
|--------|-------------|
| `--name` | Experiment name (used in `results/`) |
| `--model` | `svm`, `random_forest`, `naive_bayes`, `logreg`, `linreg`, `adaboost`, `xgboost` |
| `--csv` | Path to labeled feature CSV |
| `--dataset-dirs` | Two dataset roots (class 0 then class 1) |
| `--tasks` | Optional task filter |
| `--num-folds` | Number of CV folds (default: 10) |
| `--tune-params` | Enable GridSearchCV |
| `--no-scale` | Disable feature scaling |
| `--no-save-best-model` | Skip saving `best_model.pkl` |

---

## Training Outputs

Each run creates a unique directory:

```
results/<experiment_name>/<uuid>/
```

### Saved artifacts

| File | Description |
|----|-------------|
| `best_model.pkl` | Trained sklearn pipeline |
| `predictions_samples.csv` | Per-sample CV predictions |
| `predictions_patients.csv` | Aggregated per-subject predictions |
| `metrics_samples.csv` | Sample-level metrics (mean & std) |
| `metrics_patients.csv` | Patient-level metrics (mean & std) |
| `feature_importance.csv` | Feature importance (if supported) |
| `config.json` | Full experiment configuration |

---

## Evaluating a Trained Model

### Evaluate on a CSV

```bash
python evaluate_model.py   --model-path results/xgb_vowels/<uuid>/best_model.pkl   --csv data/eval_features.csv   --tasks vowelA   --save-dir eval_results
```

---

### Evaluate on Dataset Directories

**Labeled evaluation**
```bash
python evaluate_model.py   --model-path results/xgb_vowels/<uuid>/best_model.pkl   --dataset-dirs /data/CLASS0 /data/CLASS1   --tasks vowelA
```

**Unlabeled evaluation**
```bash
python evaluate_model.py   --model-path results/xgb_vowels/<uuid>/best_model.pkl   --dataset-dirs /data/UNLABELED   --tasks vowelA
```

---

### Evaluation Arguments

| Argument | Description |
|--------|-------------|
| `--model-path` | Path to `best_model.pkl` |
| `--csv` | CSV file with features |
| `--dataset-dirs` | One (unlabeled) or two (labeled) dataset roots |
| `--tasks` | Optional task filter |
| `--save-dir` | Output directory (default: `./eval_results`) |

---

## Notes

- Cross-validation is **grouped by subject** to avoid leakage
- Patient-level predictions are created by **averaging probabilities**
- Task filtering must match CSV values or folder names exactly

---

## Requirements

- Python **3.9.18**

The codebase was developed and tested with Python 3.9.18.  
Other Python versions may work but are not guaranteed.

## Installation of requirements

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -U pip
pip install -r requirements.txt