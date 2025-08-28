from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def get_models():
    return {
        "Logistic Regression": LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        "XGBoost": XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'),
        "LightGBM": LGBMClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    }
