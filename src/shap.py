import shap
import matplotlib.pyplot as plt

def explain(model,sample):
    explainer=shap.TreeExplainer(model)
    shap_values=explainer(sample)
    shap.summary_plot(shap_values, sample,show=False)

    plt.show()