# Digit Recognizer (MNIST) — Reproducible Classification Workflow
 
This repository contains a clean, end-to-end implementation of a supervised classification workflow
using the MNIST handwritten digit dataset. The emphasis is on **reproducibility, evaluation discipline,
and clear model comparison**, rather than leaderboard optimization.
 
---
 
## Project Overview
 
**Objective**
Classify handwritten digits (0–9) from pixel-level data using a structured and reproducible
machine learning pipeline.
 
**Workflow**
Data loading → stratified train/validation split → model pipelines →
evaluation (Accuracy, weighted F1) → best-model selection → submission export.
 
**Kaggle notebook**
https://www.kaggle.com/code/alexandroskanakis/digit-recognizer
 
---
 
## Key Results
 
- Compared **Logistic Regression**, **Decision Tree**, and **XGBoost** using the same stratified split.
- Used pipelines and feature scaling where appropriate.
- Evaluated models with **Accuracy** and **weighted F1-score**, on both the training and validation splits.
- Automatically selected the best-performing model (by validation accuracy) and generated a Kaggle-ready submission file.
**Best validation performance**
- **XGBoost** — Accuracy **0.9724**, weighted F1 **0.9724**
---
 
## Model Comparison (80/20 Stratified Split)
 
| Model               | Accuracy (Validation) | Weighted F1 (Validation) |
|---------------------|-----------------------|--------------------------|
| Logistic Regression | 0.9019                | 0.9017                   |
| Decision Tree       | 0.8650                | 0.8650                   |
| XGBoost             | 0.9724                | 0.9724                   |
 
<!-- TODO: the pipeline also computes and prints Accuracy/F1 on the training
split for every model (see console output when you run Digit_Recognizer.py).
Pasting those numbers in as a second pair of columns here — Train Accuracy /
Train F1 next to the Validation columns above — is stronger evidence of
"evaluation discipline" than validation numbers alone, since it lets a
reader see directly that no model is wildly overfitting. -->
 
---
 
## Repository Structure
 
```
├── data/
│   ├── train.csv          # Kaggle training data (local)
│   ├── test.csv            # Kaggle test data (local)
│   └── final/
│       └── Digit_Recognizer.csv  # Generated submission file
├── notebooks/
│   └── Exploratory_Data_Analysis_EDA.ipynb
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
- **Split:** 80/20 stratified train–validation
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
 
---
 
## How to Run
 
1. Clone the repository:
```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
```
 
2. Install dependencies:
```bash
   pip install -r requirements.txt
```
 
3. Run the pipeline:
```bash
   python Digit_Recognizer.py
```
 


