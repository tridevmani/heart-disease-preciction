from src.data_loader import load_data
from src.eda import perform_eda
from src.preprocessing import preprocess
from src.split import split_data
from src.model import train_models, plot_feature_importance
from src.evaluate import evaluate, plot_model_comparison, plot_roc_curve, plot_confusion
from src.predict import predict_user
from src.shap_explainer import run_shap, explain_single

import joblib
import pandas as pd
import os

# ================== SETUP ==================

print("\n Starting Cardiovascular Risk Prediction System...\n")

# Create folders
os.makedirs("models", exist_ok=True)
os.makedirs("outputs/plots", exist_ok=True)

# ================== LOAD ==================

print(" Loading dataset...")
df = load_data("data/heart.csv")

# ================== EDA ==================

print(" Performing EDA...")
perform_eda(df)

# ================== PREPROCESS ==================

print(" Preprocessing data...")
X, y, scaler, feature_names = preprocess(df)

# ================== SPLIT ==================

print(" Splitting data...")
X_train, X_test, y_train, y_test = split_data(X, y)

# ================== TRAIN ==================

print(" Training models...")
models = train_models(X_train, y_train)

# ================== EVALUATE ==================

print("📈 Evaluating models...")
results = evaluate(models, X_test, y_test)

#  Model comparison plot
plot_model_comparison(results)

# ================== BEST MODEL ==================

# FIX: results should store ROC-AUC properly
best_name = max(results, key=results.get)
best_model = models[best_name]

print(f"\n Best Model Selected: {best_name}")

# ================== SAVE ==================

print(" Saving model...")
joblib.dump(best_model, "models/model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

# ================== PERFORMANCE VISUALS ==================

print(" Generating performance plots...")
plot_roc_curve(best_model, X_test, y_test)
plot_confusion(best_model, X_test, y_test)

# Feature importance only for tree-based models
if best_name in ["Random Forest", "Decision Tree", "Gradient Boosting"]:
    plot_feature_importance(best_model, feature_names)

# ================== SHAP ==================

print("🔍 Running SHAP explainability...")

# Convert to DataFrame (IMPORTANT FIX)
X_sample_df = pd.DataFrame(X_train[:10], columns=feature_names)

explainer, shap_values = run_shap(best_model, X_sample_df, feature_names)

explain_single(shap_values, feature_names)

# ================== USER INPUT ==================

print("\n Enter Patient Details:\n")

user_input = [
    int(input("Age: ")),
    input("Sex (Male/Female): "),
    input("Chest Pain (typical angina / atypical angina / non-anginal pain / asymptomatic): "),
    int(input("BP: ")),
    int(input("Cholesterol: ")),
    input("Fasting Sugar (Yes/No): "),
    input("ECG (normal / ST-T abnormality / left ventricular hypertrophy): "),
    int(input("Max HR: ")),
    input("Exercise Angina (Yes/No): "),
    float(input("Oldpeak: ")),
    input("Slope (upsloping / flat / downsloping): "),
    input("CA (0-3): "),
    input("Thal (normal / fixed defect / reversible defect): ")
]

# ================== PREDICTION ==================

pred, prob, risk = predict_user(best_model, scaler, user_input, feature_names)

print("\n===== RESULT =====")
print("Prediction:", "Disease" if pred == 1 else "No Disease")
print(f"Probability: {prob:.2f}")
print("Risk Level:", risk)

print("\n✅ System Execution Completed Successfully!")