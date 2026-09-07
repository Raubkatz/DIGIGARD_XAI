# DIGIGARD XAI/Machine-Learning Pipeline

This repository contains the DIGIGARD binary-classification pipeline for project-history data. The workflow merges and cleans project CSV files, creates target-specific train/test splits, trains CatBoost or LightGBM classifiers, and provides several post-training evaluation and explainability stages.

The pipeline currently supports the following ground truths:

* `isBugPresent`
* `isBugfix`
* `isSZZBugIntroducer`

Stage 03 always trains **one target and one model per execution**. Stages 04–07 operate on a previously trained **CatBoostClassifier**.

## Pipeline

| Stage | Script                                                          | Purpose                                                                                                          |
| ----- | --------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| 01    | `01_DIGIGARD_merge_and_clean_project_histories.py`              | Merge all project-history CSVs, perform light cleaning, and create the shared feature/target contract.           |
| 02    | `02_DIGIGARD_create_train_test_splits.py`                       | Create independent stratified train/test splits for each ground truth.                                           |
| 03    | `03_DIGIGARD_train_classifier.py`                               | Train CatBoost or LightGBM for one target, with optional Optuna hyperparameter optimization.                     |
| 04    | `04_DIGIGARD_evaluate_catboost_classifier.py`                   | Evaluate a saved CatBoost model on full, training, and testing data, including repeated balanced evaluation.     |
| 05    | `05_DIGIGARD_catboost_shap_analysis.py`                         | Perform native CatBoost SHAP analysis, global/local feature analysis, plots, and train/test/full comparisons.    |
| 06    | `06_DIGIGARD_catboost_lime_ice_analysis.py`                     | Perform LIME local explanations and ICE/PDP response analysis for a saved CatBoost model.                        |
| 07    | `07_DIGIGARD_catboost_predicted_class_feature_distributions.py` | Compare every feature distribution between CatBoost-predicted classes using boxplots and descriptive statistics. |

## Basic Workflow

Place the original project-history CSV files in:

```text
DATA_ESEIW_final/
├── project_1.csv
├── project_2.csv
├── project_3.csv
└── ...
```

Run the stages in order:

```bash
python 01_DIGIGARD_merge_and_clean_project_histories.py
python 02_DIGIGARD_create_train_test_splits.py
python 03_DIGIGARD_train_classifier.py
python 04_DIGIGARD_evaluate_catboost_classifier.py
python 05_DIGIGARD_catboost_shap_analysis.py
python 06_DIGIGARD_catboost_lime_ice_analysis.py
python 07_DIGIGARD_catboost_predicted_class_feature_distributions.py
```

Before running Stages 03–07, set the desired target and model run in the configuration section at the top of the respective script, for example:

```python
TARGET = "isBugPresent"
MODEL_RUN = "catboost_default_rs42"
```

Stage 03 additionally controls the model and optimization mode:

```python
MODEL_TYPE = "CATBOOST"   # or "LIGHTGBM"

USE_HYPERPARAMETER_OPTIMIZATION = False
# True -> Optuna/TPE optimization
```

## Generated Folder Structure

The scripts generate their own stage-specific output directories:

```text
DIGIGARD_01_merged_data/
DIGIGARD_02_train_test_splits/
DIGIGARD_03_models/
DIGIGARD_04_catboost_evaluation/
DIGIGARD_05_catboost_shap/
DIGIGARD_06_catboost_lime_ice/
DIGIGARD_07_catboost_predicted_class_distributions/
```

A typical trained-model branch looks approximately like:

```text
DIGIGARD_03_models/
└── isBugPresent/
    └── catboost_default_rs42/
        ├── best_model.cbm
        ├── best_model.joblib
        ├── features.json
        ├── model_manifest.json
        └── reports / predictions / feature importance
```

The later CatBoost analysis stages reuse the saved model and feature contract; they do not retrain the model.

## Main Outputs

Depending on the stage, the pipeline produces:

* train/test CSV files and split reports;
* trained CatBoost or LightGBM models;
* classification metrics, confusion matrices, ROC and precision-recall curves;
* repeated balanced evaluations;
* native CatBoost SHAP values and feature rankings;
* LIME local explanations;
* ICE, centered ICE, and PDP plots;
* feature-distribution boxplots by predicted class;
* CSV, JSON, TXT, PNG, EPS, and optional HTML reports.

The held-out `test.csv` results should be used for final generalization claims. Results calculated on the training data or on the combined full dataset are descriptive diagnostics.

## Installation

A dedicated environment is recommended. The following pinned environment can be used for the complete DIGIGARD pipeline:

```text
python==3.11.9
numpy==2.1.3
scipy==1.14.1
pandas==2.2.3
scikit-learn==1.5.2
joblib==1.4.2
catboost==1.2.8
lightgbm==4.5.0
optuna==4.1.0
matplotlib==3.9.2
shap==0.46.0
lime==0.2.0.1
```

Create and activate the environment:

```bash
conda create -n digigard python=3.11.9
conda activate digigard
```

Install all packages required by the seven pipeline scripts:

```bash
pip install \
    numpy==2.1.3 \
    scipy==1.14.1 \
    pandas==2.2.3 \
    scikit-learn==1.5.2 \
    joblib==1.4.2 \
    catboost==1.2.8 \
    lightgbm==4.5.0 \
    optuna==4.1.0 \
    matplotlib==3.9.2 \
    shap==0.46.0 \
    lime==0.2.0.1
```

`LightGBM` is required for the LightGBM option in Stage 03, `Optuna` is required when hyperparameter optimization is enabled, `SHAP` adds the optional standard SHAP plots in Stage 05, and `LIME` is required by Stage 06. No TensorFlow or seaborn installation is required for the current DIGIGARD scripts.
