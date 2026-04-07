

```
# Replication Package for *Adaptive Voting Stacking with Attention Mechanism for the Enhanced Credit Scoring*

**Authors:** Shanshan Jiang, Lingyi Meng, Min Xia

## OVERVIEW

This repository contains the replication package for the paper *Adaptive Voting Stacking with Attention Mechanism for the Enhanced Credit Scoring*. It includes the source code, raw data files, processed shuffle index files, and output folders required to reproduce the main experimental results reported in the manuscript.

The package implements the proposed **Adaptive Attention-Enhanced Sequential Stacking (AAESS)** framework, together with the experimental workflows used in the paper, including:
- the main AAESS model,
- feature selection experiments,
- homogeneous versus heterogeneous stacking comparisons,
- performance evaluation and ablation studies.

All experiment scripts can be executed independently.

---

## REPOSITORY STRUCTURE

```text
.
├── data-raw/
│   ├── Fannie and Bankfear Dataset Readme
│   ├── australian.csv
│   ├── give_me_some_credit_cleaned.csv
│   ├── give_me_some_credit_cleaned.zip
│   ├── shandong.csv
│   └── shandong.zip
├── data/
│   ├── bankfear_shuffle_index.pickle
│   ├── fannie_shuffle_index.pickle
│   ├── give_shuffle_index.pickle
│   ├── shandong_shuffle_index.pickle
│   └── shuffle_index.pickle
├── models/
│   ├── HO_ablation.py
│   ├── LGB-FS-OD-RW.py
│   ├── aaess_attention_stacking.py
│   ├── feature_ablation.py
│   ├── feature_selection.py
│   ├── main.py
│   ├── outlier_detection.py
│   └── robust_weighting.py
├── plots/
├── result/
│   └── outcome.zip
├── tables/
└── README.md
```

### Directory Description

- **`data-raw/`**: raw datasets used in the experiments
- **`data/`**: processed shuffle index files for reproducible train/validation/test splits
- **`models/`**: implementation of the AAESS framework and all experiment scripts
- **`plots/`**: generated plots and visualization outputs
- **`result/`**: archived experimental outputs
- **`tables/`**: generated tables reported in the manuscript

### Main Scripts

The `models/` directory contains the main implementation and experiment scripts:

- **`aaess_attention_stacking.py`**: implementation of the proposed AAESS stacking model
- **`feature_selection.py`**: implementation of the multi-objective feature selection module
- **`outlier_detection.py`**: ensemble outlier detection module used in the AAESS framework
- **`robust_weighting.py`**: robust sample weighting module
- **`main.py`**: unified entry script for reproducing the main AAESS experiments across multiple datasets
- **`feature_ablation.py`**: script for feature ablation experiments
- **`HO_ablation.py`**: script for comparing homogeneous and heterogeneous stacking ensembles
- **`LGB-FS-OD-RW.py`**: LightGBM baseline enhanced with feature selection (FS), outlier detection (OD), and robust weighting (RW), without stacking

------

## ENVIRONMENT AND REQUIREMENTS

The experiments were conducted using Python 3.8.

### Recommended Hardware

- **GPU**: NVIDIA GeForce RTX 3090
- **CPU**: Multi-core processor (tested on Intel Core i5-13400F)

### Python Dependencies

Before running the experiments, install the required packages using:

Bash

```
pip install -r requirements.txt
```

The main Python packages used in this project include:

```
numpy`, `pandas`, `scikit-learn`, `xgboost`, `lightgbm`, `scipy`, `matplotlib
```

Please refer to `requirements.txt` for the complete dependency list.

------

## FILE PATHS AND EXECUTION

All scripts are configured to use paths relative to the repository root. Hardcoded machine-specific absolute paths have been removed. To avoid import or path issues, please run the scripts from the repository root. For example:

Bash

```
python models/main.py
```

or

Bash

```
python models/feature_ablation.py
```

The package structure has been adjusted so that the scripts can be executed on different local machines and operating systems without manual path rewriting.

------

## DATA AVAILABILITY AND PROVENANCE

### Raw Data

The raw datasets used in the experiments are stored in `data-raw/`. Examples include:

- `data-raw/shandong.csv`
- `data-raw/australian.csv`
- `data-raw/give_me_some_credit_cleaned.csv`

### Processed Split Indices

The processed shuffle index files used for reproducible data partitioning are stored in `data/`. Examples include:

- `data/shandong_shuffle_index.pickle`
- `data/give_shuffle_index.pickle`
- `data/fannie_shuffle_index.pickle`
- `data/bankfear_shuffle_index.pickle`

These shuffle index files ensure that all experiments use the same train/validation/test splits.

### Fannie Mae and BankFear Datasets

Additional instructions for the Fannie Mae and BankFear datasets are provided in:

```
data-raw/Fannie and Bankfear Dataset Readme
```

Please follow the instructions in that file to obtain, verify, and place the corresponding data files in the correct directory before running the relevant experiments. If a persistent archive link, DOI, or checksum information is provided for these datasets, users should use those records to verify file integrity after download.

------

## RUNNING THE EXPERIMENTS

All experiment scripts can be run independently.

### Main AAESS Experiment

To reproduce the main AAESS experiments, run:

Bash

```
python models/main.py
```

### Feature Ablation Experiment

To reproduce the feature ablation experiments, run:

Bash

```
python models/feature_ablation.py
```

This script evaluates the contribution of the feature selection component to model performance.

### Homogeneous vs. Heterogeneous Stacking Experiment

To reproduce the stacking comparison experiments, run:

Bash

```
python models/HO_ablation.py
```

This script compares homogeneous stacking, heterogeneous stacking, and the proposed AAESS model.

### Baseline Experiment

To run the LightGBM-based baseline without stacking, run:

Bash

```
python models/LGB-FS-OD-RW.py
```

------

## OUTPUTS

Experimental outputs are stored in the corresponding repository folders:

- **`plots/`**: figures and visualization outputs
- **`tables/`**: tabulated experimental results
- **`result/`**: archived result files, including `outcome.zip`

Unless otherwise specified, the figures reported in the paper are generated from intermediate or final result files produced by the experiment scripts.

------

## NOTES ON REPRODUCIBILITY

To improve reproducibility, the provided scripts fix random seeds where possible and use predefined shuffle index files for data partitioning. However, small numerical differences may still arise across computing environments due to factors such as:

- library version differences,
- multi-threading behavior,
- hardware differences,
- nondeterministic low-level implementations.

These differences are expected to be minor and should not affect the overall conclusions of the study. Unless otherwise stated, the running times reported in the paper correspond to the training stage of the learning models after feature selection.
