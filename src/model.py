from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt
import numpy as np
import os


# ================== TRAIN MODELS ==================

def train_models(X_train, y_train):

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=200),
        "Decision Tree": DecisionTreeClassifier(),
        "SVM": SVC(probability=True),
        "KNN": KNeighborsClassifier(),
        "Gradient Boosting": GradientBoostingClassifier()
    }

    trained_models = {}

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model

    return trained_models


# ================== FEATURE IMPORTANCE ==================

def plot_feature_importance(model, feature_names):

    try:
        # Only works for tree-based models
        importances = model.feature_importances_

        # Sort features
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(12, 6))
        plt.title("Feature Importance")

        plt.bar(range(len(importances)), importances[indices])

        plt.xticks(
            range(len(importances)),
            [feature_names[i] for i in indices],
            rotation=90
        )

        plt.xlabel("Features")
        plt.ylabel("Importance")

        plt.tight_layout()

        # Save plot
        os.makedirs("outputs/plots", exist_ok=True)
        plt.savefig("outputs/plots/feature_importance.png")

        plt.show()

    except AttributeError:
        print("⚠️ Feature importance not available for this model")