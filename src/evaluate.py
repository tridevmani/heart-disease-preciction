import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    RocCurveDisplay,
    ConfusionMatrixDisplay
)

# ================== EVALUATION ==================

def evaluate(models, X_test, y_test):

    results = {}
    table_data = []

    for name, model in models.items():

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_prob)

        print(f"\n===== {name} =====")
        print("Accuracy:", round(acc, 3))
        print("Precision:", round(prec, 3))
        print("Recall:", round(rec, 3))
        print("F1 Score:", round(f1, 3))
        print("ROC-AUC:", round(roc, 3))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

        results[name] = roc
        table_data.append([name, acc, prec, rec, f1, roc])

    # Create comparison table
    df_results = pd.DataFrame(table_data, columns=[
        "Model", "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"
    ])

    print("\n📊 MODEL COMPARISON TABLE:\n")
    print(df_results)

    # Save results
    os.makedirs("outputs", exist_ok=True)
    df_results.to_csv("outputs/model_comparison.csv", index=False)

    return results


# ================== MODEL COMPARISON ==================

def plot_model_comparison(results):

    models = list(results.keys())
    scores = list(results.values())

    plt.figure(figsize=(10, 5))
    plt.bar(models, scores)

    plt.xticks(rotation=45)
    plt.ylabel("ROC-AUC Score")
    plt.title("Model Comparison")

    os.makedirs("outputs/plots", exist_ok=True)
    plt.tight_layout()
    plt.savefig("outputs/plots/model_comparison.png")

    plt.show()


# ================== ROC CURVE ==================

def plot_roc_curve(model, X_test, y_test):

    plt.figure()

    RocCurveDisplay.from_estimator(model, X_test, y_test)

    plt.title("ROC Curve")

    os.makedirs("outputs/plots", exist_ok=True)
    plt.savefig("outputs/plots/roc_curve.png")

    plt.show()


# ================== CONFUSION MATRIX ==================

def plot_confusion(model, X_test, y_test):

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()

    plt.title("Confusion Matrix")

    os.makedirs("outputs/plots", exist_ok=True)
    plt.savefig("outputs/plots/confusion_matrix.png")

    plt.show()