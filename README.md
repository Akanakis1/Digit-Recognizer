# Digit Recognizer (MNIST): Reproducible Classification Workflow

This repository contains an end-to-end implementation of a supervised classification workflow
using the MNIST handwritten digit dataset. The emphasis is on **reproducibility, evaluation discipline,
and clear model comparison**, rather than leaderboard optimization.

---

## Project Overview

**Objective**
Classify handwritten digits (0–9) from pixel-level data using a structured and reproducible
machine learning pipeline.

**Workflow**
Data loading → stratified train / validation / holdout split → model pipelines →
evaluation (Accuracy, weighted F1) → best-model selection → holdout check → submission export.

**Kaggle notebook**
https://www.kaggle.com/code/alexandroskanakis/digit-recognizer

**Data**
The files come from the Kaggle competition [Digit Recognizer](https://www.kaggle.com/c/digit-recognizer).
They are about 125 MB, so they are **not stored in this repository**. To run the code, download them
and place `train.csv` and `test.csv` in the `data/` folder:

```bash
kaggle competitions download -c digit-recognizer
# unzip, then move train.csv and test.csv into data/
```

---

## Key Results

- Compared **Logistic Regression**, **Decision Tree**, and **XGBoost** on the same stratified splits.
- Evaluated with **accuracy** and **weighted F1-score** on train, validation and holdout data.
- The best model was chosen on the validation set. The holdout set was evaluated afterwards and
  never used for fitting, early stopping or selection.

**Best holdout performance**
- **XGBoost**: accuracy **0.9755**, weighted F1 **0.9755**

---

## Model Comparison (80/10/10 stratified split, `random_state=42`)

| Model               | Train accuracy | Validation accuracy | Holdout accuracy | Holdout weighted F1 |
|---------------------|---------------:|--------------------:|-----------------:|--------------------:|
| Logistic Regression | 0.9591         | 0.8988              | 0.8981           | 0.8980              |
| Decision Tree       | 0.9286         | 0.8583              | 0.8612           | 0.8612              |
| **XGBoost**         | 0.9999         | **0.9712**          | **0.9755**       | **0.9755**          |

XGBoost fits the training data almost perfectly (0.9999) and scores about 2.4 to 2.9 points lower on
validation and holdout. The validation and holdout scores agree with each other, so this gap is
ordinary overfitting and not a sign of a broken evaluation.

---

## Repository Structure

```
├── data/
│   ├── train.csv          # Kaggle training data (download it, see Data above; not tracked)
│   ├── test.csv           # Kaggle test data (download it; not tracked)
│   └── final/
│       └── Digit_Recognizer.csv  # Generated submission file
├── notebooks/
│   └── Exploratory_Data_Analysis_(EDA).ipynb
├── Digit_Recognizer.py     # End-to-end training & evaluation pipeline
├── requirements.txt        # Project dependencies
└── README.md               # Project documentation
```

---

## Methodology

### Data
- `train.csv`: digit labels (0–9) and pixel values
- `test.csv`: pixel values only

### Training Setup
- **Split:** 80% train / 10% validation / 10% holdout, all stratified by digit
- **Metrics:** Accuracy, weighted F1-score

### Models
- **Logistic Regression**
  - Pipeline with StandardScaler
  - Class-balanced training
- **Decision Tree**
  - Entropy criterion
  - Depth constrained to reduce overfitting
- **XGBoost**
  - Tuned depth and learning rate
  - Early stopping against the validation set

### Output
Automatically generates a Kaggle-compatible submission file at
`data/final/Digit_Recognizer.csv`.

### Limitations
- A single random split was used, so the numbers carry some sampling noise.
- No cross-validation and no systematic hyperparameter search.

---

## How to Run

1. Clone the repository:
```bash
   git clone https://github.com/Akanakis1/Digit-Recognizer.git
   cd Digit-Recognizer
```

2. Download `train.csv` and `test.csv` from Kaggle into `data/` (see Data above).

3. Install dependencies:
```bash
   pip install -r requirements.txt
```

4. Run the pipeline (XGBoost takes several minutes on a CPU):
```bash
   python Digit_Recognizer.py
```

Reported numbers were produced with Python 3.13, scikit-learn 1.9.1 and xgboost 3.4.1.
Other library versions can shift the last digits.
