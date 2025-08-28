import pandas as pd 
import joblib
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

from src.model import get_models
from src.evaluate import evaluate_model, print_classification_report


data = pd.read_csv("Data\processed.csv")

X= data.drop("Churn", axis=1)
y= data["Churn"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
#stratify=y keeps the same proportion of churn vs. non-churn in train and test sets.

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("Before SMOTE:")
print(y_train.value_counts())

print("\nAfter SMOTE:")
print(y_train_res.value_counts())

models = get_models()

results= []

for name, model in models.items():
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred)
    metrics["Model"] = name
    results.append(metrics)

#to compare models 
df_results = pd.DataFrame(results).sort_values(by="F1-Score", ascending=False)
print("\nModel Comparison:\n", df_results)

best_model_name = "Logistic Regression"
best_model = models[best_model_name]
print(f"Selected model: {best_model_name}")

joblib.dump(best_model, f"models/{best_model_name.replace(' ', '_').lower()}_model.pkl")
print(f"{best_model_name} saved successfully in models folder.")
