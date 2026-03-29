import shap
import matplotlib

# ✅ Force non-GUI backend (VERY IMPORTANT)
matplotlib.use('Agg')

import matplotlib.pyplot as plt

def explain_model(model, X_test):
    print("\nGenerating SHAP summary plot...")

    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)

    # ✅ Do NOT show plot (this avoids tkinter error)
    shap.summary_plot(shap_values, X_test, show=False)

    # ✅ Save plot
    plt.savefig("shap_summary.png", bbox_inches='tight')
    plt.close()

    print("✅ SHAP plot saved as shap_summary.png")