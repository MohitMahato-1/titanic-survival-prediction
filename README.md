# Titanic Survival Analysis

A Jupyter notebook project exploring survival patterns in the Kaggle Titanic dataset. The current notebook covers exploratory data analysis and feature engineering; model training and evaluation are still to be completed.

## What the notebook covers

- Checks dataset structure and missing values.
- Explores survival patterns by passenger sex and ticket class.
- Creates `FamilySize`, `IsAlone`, and `Title` features for further analysis.

## Dataset

The project uses the Kaggle Titanic files in `data/`:

- `train.csv` — training data with survival labels.
- `test.csv` — test data for predictions.
- `gender_submission.csv` — sample submission format.

## Project structure

```text
Titanic-survival-prediction/
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── gender_submission.csv
├── notebook/
│   └── titanic_analysis.ipynb
└── README.md
```

## Setup and run

Create an environment, install the notebook dependencies, and open Jupyter from the repository root:

```bash
python -m venv .venv
```

On Windows, activate it with:

```powershell
.venv\Scripts\Activate.ps1
```

On macOS or Linux, activate it with:

```bash
source .venv/bin/activate
```

Install dependencies and start the notebook:

```bash
python -m pip install numpy pandas matplotlib seaborn scikit-learn jupyter
jupyter notebook notebook/titanic_analysis.ipynb
```

> **Path note:** The notebook currently uses absolute Windows paths for the training CSV and a saved plot. Replace those paths with paths that match your local checkout before running it.

## Current status

**In progress.** Exploratory analysis and initial feature engineering are present in the notebook. A trained classifier, evaluation metrics, and prediction workflow have not been added yet.
