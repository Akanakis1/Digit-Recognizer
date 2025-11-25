# 🧮 Digit Recognizer - Handwritten Digit Classification

**Dataset:** [Digit Recognizer – Kaggle](https://www.kaggle.com/code/alexandroskanakis/digit-recognizer)  
**Notebook & Code:** [View My Solution](https://www.kaggle.com/code/alexandroskanakis/digit-recognizer)

---

## 📊 Project Overview

This project tackles the handwritten digit classification problem using the popular MNIST dataset from Kaggle's Digit Recognizer competition. It employs three machine learning models — Logistic Regression, Decision Tree, and XGBoost — wrapped in pipelines to preprocess and classify digit images based on their pixel values. Automated evaluation and best-model selection are included.

---

## 🔍 Motivation

Digit recognition is a classic image classification challenge that serves as a benchmark for many machine learning techniques. This project explores various algorithmic approaches and demonstrates how pipelines and model evaluation can yield robust, high-accuracy classifiers on this visual dataset.

---

## 📘 Dataset Overview

The dataset consists of:

<div align="center">

| File                   | Description                                      |
|------------------------|--------------------------------------------------|
| `train.csv`            | Training data with labeled digit classes (0-9)   |
| `test.csv`             | Unlabeled test images for prediction             |
| `data/final/`          | Folder to save final prediction submission CSVs  |
| `requirements.txt`     | Python package dependencies                      |

</div>

### ✨ Key Variables

<div align="center">

| Variable    | Description                           |
|-------------|---------------------------------------|
| `label`     | The digit class label (0 to 9)        |
| All others  | Pixel values representing the images  |

</div>

---

## 🎯 Project Objective

Build, evaluate, and compare multiple machine learning models to classify handwritten digits accurately, featuring:

- Data import and preprocessing with scaling when needed  
- Stratified train-validation split for model assessment  
- Use of Logistic Regression, Decision Tree, and XGBoost algorithms  
- Evaluation using accuracy and weighted F1-score metrics  
- Automated best model selection based on validation accuracy  
- Final training on all data and submission file generation  

---

## 🏆 Achievements

- Achieved high validation accuracy over 97% using XGBoost  
- Implemented balanced class weights and early stopping for improved model training  
- Built reusable pipelines for preprocessing and modeling  
- Generated reproducible submission file ready for Kaggle upload

---

## 📊 Model Evaluation Results

After training and validation on the stratified 80/20 split, the models yielded the following performance metrics:

<div align="center">

| Model Name              | Accuracy Train | F1 Train   | Accuracy Valid | F1 Valid   |
|-------------------------|----------------|------------|----------------|------------|
| **Logistic Regression** | 0.9568         | 0.9568     | 0.9019         | 0.9017     |
| **Decision Tree**       | 0.9276         | 0.9276     | 0.8651         | 0.8650     |
| **XGBoost**             | **0.9999**     | **0.9999** | **0.9724**     | **0.9724** |

</div>

**Best Model Selected:** **XGBoost** (Validation Accuracy: 0.9724).
A submission file (`Digit_Recognizer.csv`) was generated using the best model, trained on the full dataset.

---

## 🔧 Tools & Technologies

- **Programming Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost  
- **Platform:** Local environment and Kaggle competition  

---

## 📁 Repository Contents

<div align="center">

| File                              | Description                                                   |
|-----------------------------------|---------------------------------------------------------------|
| `digit_recognizer.py`             | Full pipeline: data loading, modeling, evaluation, prediction |
| `train.csv`                       | Training images with labels                                   |
| `test.csv`                        | Test images for inference                                     |
| `data/final/Digit_Recognizer.csv` | Submission file with predicted labels                         |
| `requirements.txt`                | Python dependencies                                           |

</div>

---

## 📂 Project Directory Structure

Digit-Recognizer/  
├── data/  
│   ├── final/  
│   │   └── Digit_Recognize.csv  
│   ├── test.csv  
│   └── train.csv  
├── digit_recognizer.py  
├── README.md  
└── requirements.txt  

- **data/**: Contains dataset files and output submission folder  
- **digit_recognizer.py**: Main script for training, evaluating, and predicting  
- **requirements.txt**: List of Python packages and versions  

---

## 🚀 Project Workflow

A[Load Data] --> B[Train-Validation Split]  
B --> C[Define Models & Pipelines]  
C --> D[Train & Evaluate Models]  
D --> E[Select Best Model]  
E --> F[Retrain on Full Data]  
F --> G[Predict on Test Data]  
G --> H[Export Submission File]
