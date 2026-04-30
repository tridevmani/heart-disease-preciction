from src.data_loader import load_data
from src.eda import perform_eda
from src.preprocessing import preprocess
from src.split import split_data
from src.model import train_models, plot_feature_importance
from src.evaluate import evaluate, plot_model_comparison, plot_roc_curve, plot_confusion
from src.shap_explainer import run_shap

import joblib
import pandas as pd
import os

# ================== CONFIG ==================

RUN_EDA = True          # Set False to skip EDA
RUN_SHAP = True         # Set False to skip SHAP (faster training)

# ================== SETUP ==================

print("\n🚀 Starting Cardiovascular Risk Prediction System...\n")

os.makedirs("models", exist_ok=True)
os.makedirs("outputs/plots", exist_ok=True)

# ================== LOAD ==================

print("📥 Loading dataset...")
df = load_data("data/heart.csv")

# ================== EDA ==================

if RUN_EDA:
    print("📊 Performing EDA...")
    perform_eda(df)

# ================== PREPROCESS ==================

print("⚙️ Preprocessing data...")
X, y, scaler, feature_names = preprocess(df)

# ================== SPLIT ==================

print("✂️ Splitting data...")
X_train, X_test, y_train, y_test = split_data(X, y)

# ================== TRAIN ==================

print("🤖 Training models...")
models = train_models(X_train, y_train)

# ================== EVALUATE ==================

print("📈 Evaluating models...")
results = evaluate(models, X_test, y_test)

plot_model_comparison(results)

# ================== BEST MODEL ==================

best_name = max(results, key=results.get)
best_model = models[best_name]

print(f"\n🏆 Best Model Selected: {best_name}")

# ================== SAVE ==================

print("💾 Saving model artifacts...")

joblib.dump(best_model, "models/model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(feature_names, "models/features.pkl")

# ================== PERFORMANCE VISUALS ==================

print("📊 Generating performance plots...")

plot_roc_curve(best_model, X_test, y_test)
plot_confusion(best_model, X_test, y_test)

if best_name in ["Random Forest", "Decision Tree", "Gradient Boosting"]:
    plot_feature_importance(best_model, feature_names)

# ================== SHAP ==================

if RUN_SHAP:
    print("🔍 Running SHAP explainability...")

    X_sample_df = pd.DataFrame(X_train[:50], columns=feature_names)

    explainer, shap_values = run_shap(best_model, X_sample_df, feature_names)

    # 🔥 SAVE SHAP FOR STREAMLIT (IMPORTANT)
    joblib.dump(explainer, "models/shap_explainer.pkl")
    joblib.dump(shap_values, "models/shap_values.pkl")

# ================== END ==================

print("\n✅ Model Training Completed Successfully!")