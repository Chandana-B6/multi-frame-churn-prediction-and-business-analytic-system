import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importance(model, X):
    importance = model.feature_importances_
    features = np.array(X.columns)

    # Sort
    sorted_idx = np.argsort(importance)[::-1]

    # ✅ Take top 10 only (IMPORTANT FIX)
    top_n = 10
    top_features = features[sorted_idx][:top_n]
    top_importance = importance[sorted_idx][:top_n]

    plt.figure(figsize=(8,6))
    plt.barh(top_features[::-1], top_importance[::-1])
    plt.xlabel("Importance")
    plt.title("Top 10 Feature Importance")

    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.close()

    print("✅ Clean feature importance saved!")