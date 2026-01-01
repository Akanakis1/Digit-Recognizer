# ✍️ Digit Recognizer (MNIST) — Handwritten Digit Classification (Kaggle)

Classify handwritten digits (0–9) from pixel data using a clean, reproducible ML workflow:  
**load data → stratified train/validation split → model pipelines → evaluation (Accuracy + F1) → best model selection → Kaggle submission export**.

🔗 Kaggle Notebook / Solution:
https://www.kaggle.com/code/alexandroskanakis/digit-recognizer

---

## ⭐ Highlights
- Compared **Logistic Regression**, **Decision Tree**, and **XGBoost** on the same stratified split.
- Used **pipelines + scaling** where appropriate (LogReg + StandardScaler).
- Evaluated models using **Accuracy** and **Weighted F1-score** (train + validation).
- Auto-selected the best model by **validation accuracy** and generated a submission file.

✅ Best validation result: **XGBoost — 0.9724 Accuracy / 0.9724 Weighted F1**.

---

## 🏆 Model Results (80/20 stratified split)

| Model | Accuracy (Train) | F1 (Train) | Accuracy (Valid) | F1 (Valid) |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.9568 | 0.9568 | 0.9019 | 0.9017 |
| Decision Tree | 0.9276 | 0.9276 | 0.8651 | 0.8650 |
| **XGBoost** | **0.9999** | **0.9999** | **0.9724** | **0.9724** |

---

## 🧠 What’s inside (approach)
### 1) Data
- `train.csv`: labels (0–9) + pixel values
- `test.csv`: pixel values only

### 2) Training setup
- Split: **80/20 stratified** train-validation
- Metrics: **Accuracy** + **Weighted F1**

### 3) Models
- **Logistic Regression** (with StandardScaler, class_weight="balanced")
- **Decision Tree** (entropy criterion, max_depth=10, class_weight="balanced")
- **XGBoost Classifier** (hist tree method, 300 estimators, tuned learning rate / depth)

### 4) Output
Creates a Kaggle-ready submission file:
`data/final/Digit_Recognizer.csv`

---

## 📁 Repository contents
- `Digit_Recognizer.py` — end-to-end script: training, evaluation, best-model selection, submission export
- `Exploratory_Data_Analysis_(EDA).ipynb` — optional EDA notebook (visual exploration)
- `requirements.txt` — dependencies
- `data/train.csv`, `data/test.csv` — Kaggle dataset files (place locally)

---

## 🚀 How to run

### 1) Install dependencies
```bash
pip install -r requirements.txt
