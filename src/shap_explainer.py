import shap
import matplotlib.pyplot as plt
from src.utils import save_plot

def run_shap(model, X_sample, feature_names):

    explainer = shap.Explainer(model, X_sample)
    shap_values = explainer(X_sample)

    shap.summary_plot(shap_values.values, X_sample, show=False)
    save_plot("shap_summary")

    return explainer, shap_values


def explain_single(shap_values, feature_names):

    print("\nTop Feature Contributions:")

    values = shap_values.values[0]

    if len(values.shape) > 1:
        values = values[:, 1]

    impact = list(zip(feature_names, values))
    impact = sorted(impact, key=lambda x: abs(x[1]), reverse=True)

    for f, v in impact[:5]:
        print(f"{f}: {'↑' if v > 0 else '↓'} ({v:.3f})")