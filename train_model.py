import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

df = pd.read_csv("data/student_performance.csv")

df["Result"] = df["FinalScore"].apply(lambda x: 1 if x >= 60 else 0)

X = df.drop(["FinalScore", "Result"], axis=1)
y = df["Result"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "LogisticRegression": LogisticRegression(class_weight="balanced", max_iter=2000),
    "RandomForest": RandomForestClassifier(class_weight="balanced"),
    "GradientBoosting": GradientBoostingClassifier(),
}

results = []
best_score = 0
best_model = None

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    results.append([name, acc])

    if acc > best_score:
        best_score = acc
        best_model = model

results_df = pd.DataFrame(results, columns=["Model", "Accuracy"])

os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/best_model.pkl")
joblib.dump(results_df, "models/model_results.pkl")
