import matplotlib.pyplot as plt

def plot_model_comparison(results):
    models = list(results.keys())
    scores = list(results.values())

    plt.figure(figsize=(8,5))
    plt.bar(models, scores)

    plt.ylabel("AUC Score")
    plt.title("Model Performance Comparison")

    plt.xticks(rotation=20)
    plt.tight_layout()

    plt.savefig("model_comparison.png")
    plt.close()

    print("✅ Model comparison graph saved!")