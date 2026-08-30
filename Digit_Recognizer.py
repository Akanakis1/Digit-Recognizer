# =====================================================
# 1. Import Libraries
# =====================================================
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
 
# =====================================================
# 2. Data Import
# =====================================================
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")
 
X = train_df.drop("label", axis=1).values
y = train_df["label"].values
X_test = test_df.values
 
# =====================================================
# 3. Train-Validation Split
# =====================================================
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
 
# =====================================================
# 4. Models as Pipelines
# =====================================================
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
            verbose=0
        ))
    ]),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=10,
        criterion="entropy",
        random_state=42,
        class_weight="balanced"
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.9,
        tree_method="hist",
        reg_alpha=0.1,
        reg_lambda=1,
        random_state=42,
        eval_metric="mlogloss",
        verbosity=0,
        n_jobs=-1,
        early_stopping_rounds=20,
        # FIX: eval_set was already being passed to .fit() below, but with
        # no early_stopping_rounds it was only ever used for logging — all
        # 300 trees got built regardless of validation performance. This
        # now actually stops training once mlogloss on X_val stops improving.
    )
}
 
# =====================================================
# 5. Training & Evaluation Function
# =====================================================
def evaluate_model(name, model, X_train, y_train, X_val, y_val):
    fit_params = {}
    if name == "XGBoost":
        fit_params = {
            "eval_set": [(X_val, y_val)],
            "verbose": False
        }
    model.fit(X_train, y_train, **fit_params)
 
    metrics = {}
    for split, X_split, y_split in [("Train", X_train, y_train), ("Valid", X_val, y_val)]:
        y_pred = model.predict(X_split)
        metrics[f"Accuracy {split}"] = accuracy_score(y_split, y_pred)
        metrics[f"F1 {split}"] = f1_score(y_split, y_pred, average="weighted")
    return metrics
 
# =====================================================
# 6. Run All Models
# =====================================================
results = {
    name: evaluate_model(name, model, X_train, y_train, X_val, y_val)
    for name, model in models.items()
}
 
for name, metrics in results.items():
    print(f"\n== {name} ==")
    for k, v in metrics.items():
        # FIX: removed the dead `if "Matrix" in k` branch — no key produced
        # by evaluate_model() has ever contained "Matrix" (it looks like a
        # leftover from an earlier version that stored a confusion matrix),
        # so every metric now just prints its value directly.
        print(f"{k}: {v:.4f}")
 
# =====================================================
# 7. Select Best Model
# =====================================================
# NOTE: selection uses validation Accuracy specifically. Weighted F1 is also
# computed and printed above for inspection; on this dataset the two track
# each other closely since MNIST's 10 classes are near-balanced, so the
# choice of tiebreaker metric doesn't change which model wins here.
best_model_name = max(results, key=lambda x: results[x]["Accuracy Valid"])
best_model = models[best_model_name]
print(f"\nBest model selected: {best_model_name}")
 
# =====================================================
# 8. Predict on Test Set & Save Submission
# =====================================================
test_preds = best_model.predict(X_test)
submission = pd.DataFrame({
    "ImageId": np.arange(1, len(test_preds) + 1),
    "Label": test_preds
})
 
output_dir = "data/final"
os.makedirs(output_dir, exist_ok=True)
submission.to_csv(os.path.join(output_dir, "Digit_Recognizer.csv"), index=False)
print("Submission file saved as 'Digit_Recognizer.csv'")
 


