# Network Intrusion Detection & Zero-Day Threat Classification System

**Group Members**
1. Rodgers Kipkirui — SKIRRO2411
2. Joan Cherono — SCHEKI2412
3. Amos Kiprotich — SAMOKI2411

## 1. Project overview

This project applies supervised machine learning to labelled network-flow
records. It compares Decision Tree, Random Forest and XGBoost classifiers to
identify benign traffic and attack categories represented in the training data.

> Important research limitation: a supervised classifier trained on labelled
> attack categories does not automatically prove detection of genuinely unseen
> zero-day attacks. That claim requires a separate evaluation in which one or
> more attack families are withheld from training and used only for testing.
> Until that experiment is completed, describe the system as an intrusion
> detection classifier with potential to support anomaly/novel-threat detection.

## 2. Requirements

- Python 3.10 or newer
- A labelled network-flow CSV dataset
- Internet access for the first-time installation of Python packages

## 3. Installation

```bash
# Clone the repository
git clone https://github.com/kipAmos/network-intrusion-detection-ml.git
cd network-intrusion-detection-ml

# Create and activate a virtual environment (Windows)
py -m venv .venv
.venv\Scripts\activate

# On Linux/macOS, use:
# python3 -m venv .venv
# source .venv/bin/activate

pip install -r requirements.txt
```

## 4. Add the dataset

Place the CSV file at:

`data/network_traffic.csv`

The target column is detected automatically when named `Label`, `Class`,
`Attack`, or `Target`. For a differently named target column, supply `--target`.

The dataset should contain labelled flow records, for example benign traffic
and attack categories. This repository does not include CIC-IDS2017 or another
third-party dataset; obtain it from its official source and respect its licence.

## 5. Train and evaluate

```bash
python train.py
```

To specify a different dataset or target column:

```bash
python train.py --data data/your_dataset.csv --target Label
```

The default evaluation uses an 80/20 stratified train/test split and up to
10-fold stratified cross-validation on the training portion. The best model is
selected using mean cross-validation weighted F1-score, then evaluated once on
the held-out test set.

Generated files:

- `models/best_intrusion_model.joblib` — fitted preprocessing and classifier
- `reports/metrics.json` — test metrics and class-level report
- `reports/model_comparison.csv` — cross-validation comparison
- `reports/confusion_matrix.csv` — confusion matrix

Metrics include accuracy, weighted precision, weighted recall, weighted F1,
micro false-positive rate and weighted one-vs-rest ROC-AUC when it can be
calculated.

## 6. Classify new traffic records

Prepare a CSV containing the same feature columns used for training, without
the target column, then run:

```bash
python predict.py --input data/new_network_flows.csv
```

Predictions are written to `reports/predictions.csv`. Missing training features
are filled as missing values and handled by the saved preprocessing pipeline.
For meaningful results, the input feature definitions and units must match the
training dataset.

## 7. Methodology

1. Load and clean the labelled flow records.
2. Remove duplicate records, invalid target rows and identifier-like columns.
3. Impute missing values, scale numerical fields and one-hot encode categorical
   fields inside a scikit-learn pipeline.
4. Compare Decision Tree, Random Forest and XGBoost using stratified
   cross-validation.
5. Select the model with the best mean cross-validation weighted F1-score.
6. Evaluate the selected model on a separate stratified test set.
7. Save the model and evaluation reports for reproducibility.

## 8. Interpreting results

Use the values actually generated in `reports/metrics.json` in the submitted
report. Do not copy example or expected scores. Report the dataset, number of
records after cleaning, class distribution, split, cross-validation method and
per-class precision/recall/F1 alongside overall metrics.

A low false-positive rate is important because incorrectly flagging benign
traffic can create unnecessary alerts. Recall is important because missed
attacks can remain undetected. Weighted metrics account for class frequency;
also inspect per-class scores and the confusion matrix, especially when the
dataset is imbalanced.

## 9. Limitations and future work

- Benchmark data may not represent all live network environments.
- Results depend on dataset quality, feature definitions and class balance.
- A high score on known classes is not proof of zero-day detection.
- Future work should test unseen attack families, evaluate on temporally
  separated traffic, monitor inference latency, and investigate drift-aware
  retraining and carefully controlled deployment.


