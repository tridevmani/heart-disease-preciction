import numpy as np
import pandas as pd

def predict_user(model, scaler, user_input, feature_names):

    # Original column names (before encoding)
    columns = [
        "age", "sex", "chest_pain_type", "resting_blood_pressure",
        "cholestoral", "fasting_blood_sugar", "rest_ecg",
        "Max_heart_rate", "exercise_induced_angina",
        "oldpeak", "slope", "vessels_colored_by_flourosopy",
        "thalassemia"
    ]

    # Convert input to DataFrame
    df = pd.DataFrame([user_input], columns=columns)

    # Apply same encoding
    df_encoded = pd.get_dummies(df)

    # Align with training features
    df_encoded = df_encoded.reindex(columns=feature_names, fill_value=0)

    # Scale
    X_scaled = scaler.transform(df_encoded)

    # Predict
    pred = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0][1]

    # Risk levels
    if prob >= 0.7:
        risk = "HIGH"
    elif prob >= 0.4:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return pred, prob, risk