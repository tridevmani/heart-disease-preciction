import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess(df):

    # Encode categorical features
    df_encoded = pd.get_dummies(df, drop_first=True)

    # Split features & target
    X = df_encoded.drop("target", axis=1)
    y = df_encoded["target"]

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, X.columns