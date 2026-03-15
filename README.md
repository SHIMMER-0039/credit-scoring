# Replication package for "Adaptive Voting Stacking with Attention Mechanism for the Enhanced Credit Scoring"

Shanshan Jiang, Lingyi Meng, Min Xia

## Overview & contents

The plots/ directory contains the source files and scripts for all visualizations reported in the manuscript to ensure full reproducibility.

The code in this replication material reproduces the experimental results reported in the paper "Adaptive Voting Stacking with Attention Mechanism for the Enhanced Credit Scoring".

The repository implements the proposed Adaptive Attention-Enhanced Sequential Stacking (AAESS) framework together with the experiments used in the paper, including:

the main AAESS model

feature selection experiments

homogeneous vs heterogeneous stacking comparisons

performance evaluation and ablation studies

Each experiment can be executed independently using the corresponding Python script.

The main contents of the repository are the following:

plots/: folder of generated plots from the experiments

tables/: folder of generated experimental result tables

data-raw/: folder of raw data files used in the experiments

data/: folder of processed data files and shuffle index files for reproducible data splits

models/: folder containing the implementation of the AAESS framework and experimental scripts

Inside the models/ folder, the main scripts include:

aaess_attention_stacking.py: implementation of the proposed AAESS stacking model

feature_selection.py: implementation of the multi-objective feature selection module

outlier_detection.py: ensemble outlier detection module used in the AAESS framework

robust_weighting.py: robust sample weighting module

fannie_test.py: main experiment script used to train and evaluate the AAESS model

feature_ablation.py: script used to run the feature ablation experiments

HO_ablation.py: script used to compare homogeneous and heterogeneous stacking ensembles

noatt_give_test.py:Model without stacking

## Instructions & computational requirements.

All file paths are relative to the root of the replication package.

Before running the experiments, please install the required Python dependencies using:

pip install -r requirements.txt


The experiments were conducted using Python 3.8.

GPU: NVIDIA GeForce RTX
3090 GPU


Multi-core CPU: recommended Core i5-13400F


The main Python packages used in the experiments include:

numpy, pandas, scikit-learn, xgboost, lightgbm, scipy, matplotlib

A comprehensive list of dependencies can be found in the file:
requirements.txt


All experiment scripts can be run individually, in any order.

## Data availability and provenance
The raw datasets used in the experiments are located in:

data-raw/


Example:

data-raw/shandong.csv


These files contain the original credit scoring data used in the experiments.

Processed datasets

The processed datasets and shuffle index files used for reproducible train/validation/test splits are stored in:

data/


Example:

data/shandong_shuffle_index.pickle


These shuffle index files ensure that all experiments use the same data partitions for reproducibility.
### Running the experiments

To reproduce the main experimental results reported in the paper, run:

python models/fannie_test.py

This script trains and evaluates the proposed AAESS model on the credit scoring dataset.

### Feature ablation experiment

To reproduce the feature ablation experiments, run:

python models/feature_ablation.py


This script evaluates the effect of the feature selection component on model performance.


### Homogeneous vs heterogeneous stacking experiment

To reproduce the stacking ensemble comparison experiments, run:

python models/HO_ablation.py


This script compares homogeneous stacking, heterogeneous stacking, and the proposed AAESS model.




### Additional note on experimental results

Due to the stochastic nature of machine learning algorithms and data sampling procedures, the results of individual runs may exhibit slight variations across different executions. To mitigate this effect and obtain more stable results, the example scripts included in this repository perform a limited grid search over several model parameters in order to identify better-performing configurations.

In addition, the running times reported in the main paper correspond to the execution time of the base learning models after the feature selection stage has been completed. That is, the reported time measurements do not include the feature selection process itself but only the training time of the subsequent classification models.

## References

If you use this code, please cite the following paper:

Jiang, S., Meng, L., Xia, M.
Adaptive Voting Stacking with Attention Mechanism for the Enhanced Credit Scoring.
