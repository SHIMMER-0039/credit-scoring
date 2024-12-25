# credit-scoring
Adaptive Voting Stacking with Attention Mechanism for the Enhanced Credit Scoring

# Adaptive Voting Stacking with Attention Mechanism

## Project Overview

This project implements an advanced credit scoring model using **Adaptive Bayesian Stacking**, a method that combines multiple base classifiers and assigns dynamic weights using a Bayesian Ridge model. The project also includes automated feature selection techniques to optimize the model's performance. This approach is designed to enhance the prediction accuracy and robustness of credit risk evaluation models.

---

## Directory Structure

```plaintext
Adaptive Voting Stacking with Attention Mechanism
│
├── compare
│   ├── herter/                 # Supporting experiment files
│   ├── HETER/                  # Experiment data folder
│   ├── rador/                  # Experiment-related data
│   ├── ranked/                 # Supporting ranked features data
│   ├── bankfear.py             # Bankfear dataset experiment
│   ├── diff_pict.py            # Comparison of different graphs
│   ├── fannie.py               # Experiment using Fannie Mae dataset
│   ├── feature_othermethod.py  # Other feature selection methods
│   ├── give.py                 # Feature analysis script
│   ├── shandong.py             # Experiments on Shandong dataset
│   ├── tiqv.py                 # Experiments on TIQV dataset
│   └── real_feature.py         # Feature extraction for experiments
│
├── diff dataset                # Datasets folder
│   ├── bankfear/               # Bankfear dataset
│   ├── fannie/                 # Fannie Mae dataset
│   ├── give/                   # Give dataset
│   ├── shandong/               # Shandong dataset
│
├── main
│   ├── adaptive_bayesian_stacking.py # Main model implementation
│   ├── feature_selection.py          # Feature selection logic
│   ├── important_score/              # Feature importance utilities
│   ├── read.py                       # Data loading utilities
│   ├── test.py                       # Entry point for main experiments
│
├── old                              # Deprecated or backup scripts
│   ├── cal_metrics.py               # Metric calculation utilities
│   ├── read.py                      # Data loading utilities
│   ├── realtext.py                  # Real dataset analysis script
│   ├── Tree-based heterogeneous cascade
│   └── yizhi.py                     # "Yizhi" model experiment script
│
├── venv                             # Python virtual environment (optional)
├── img.png                          # Supporting image for documentation
├── lime.ipynb                       # LIME explanation method experiments
└── stacking.py                      # Stacking model implementation
```

---

## Requirements

Before running the code, make sure the following dependencies are installed:

- **Python 3.8+**
- Libraries:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `xgboost`
  - `lightgbm`
  - `scipy`
  - `matplotlib`

Install dependencies using the following command:

```bash
pip install -r requirements.txt
```

---

## Data Path Configuration

The following paths need to be configured properly in the main script:

- **Dataset path:**
  - `dataset_path`: Root directory for credit scoring datasets (e.g., `D:/study/credit_scoring_datasets/`)
- **Output path:**
  - `save_path`: Directory for saving model results (e.g., `D:/study/second/outcome/`)

Ensure that these paths are correctly set before running the code.

---

## How to Run

### 1. Run the Main Program

To perform feature selection and train the model, execute the following command:

```bash
python main/test.py
```

### 2. Run Dataset-specific Experiments

You can run specific experiments on datasets using scripts in the `compare` directory. For example, to run the Fannie Mae dataset experiment:

```bash
python compare/fannie.py
```

### 3. View Results

All results will be saved in the `save_path` directory as `.pickle` files. You can load and inspect them using the following code:

```python
import pickle

file_path = "path_to_saved_file"
with open(file_path, 'rb') as f:
    results = pickle.load(f)
print(results)
```

---

## Features

1. **Adaptive Bayesian Stacking:**

   - Dynamically assigns weights to classifiers using a Bayesian Ridge model.
   - Combines multiple base models (RandomForest, XGBoost, LightGBM) for robust predictions.

2. **Automated Feature Selection:**

   - Iteratively applies feature selection techniques with Pareto efficiency optimization.
   - Includes methods such as classifier-based feature evaluation and correlation-based filtering.

3. **Comprehensive Evaluation Metrics:**

   - Supports multiple classification metrics: AUC, Precision, Recall, F1, Brier Score, and more.

4. **GPU Support:**

   - Leverages GPU acceleration for XGBoost and LightGBM, making it suitable for large datasets.

---

## Results

The best model's performance is evaluated using the following metrics:

- **Accuracy**
- **ROC AUC**
- **Precision and Recall**
- **F1 Score**
- **Brier Score**
- **Log Loss**
- **Type 1 and Type 2 Errors**

The saved results include these metrics for each experiment and grid search combination.

---

## Contact

If you have any questions or suggestions, please feel free to reach out to the project author.

---



