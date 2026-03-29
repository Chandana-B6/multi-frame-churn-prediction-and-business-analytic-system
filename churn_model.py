from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.close()

    print("✅ Confusion matrix saved!")


def train_models(df):

    # Split data
    X = df.drop("Churn_Yes", axis=1)
    y = df["Churn_Yes"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Random Forest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(eval_metric='logloss'),
        "LightGBM": LGBMClassifier()
    }

    results = {}

    plt.figure(figsize=(8, 6))

    for name, model in models.items():
        print(f"\n{name} Results:")

        # Train
        model.fit(X_train, y_train)

        # Predict
        preds = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]

        # Metrics
        print(classification_report(y_test, preds))

        auc = roc_auc_score(y_test, prob)
        results[name] = auc

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")

    # ROC Plot
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig("roc_curve.png", dpi=300)
    plt.close()

    print("\n✅ ROC curve saved as roc_curve.png")

    # 📊 Print comparison
    print("\n📊 Model AUC Comparison:")
    for model_name, score in results.items():
        print(f"{model_name}: {score:.4f}")

    # 🎯 Choose best model (you can change if needed)
    best_model = models["XGBoost"]

    # 🔥 Generate confusion matrix using best model
    y_pred = best_model.predict(X_test)
    plot_confusion_matrix(y_test, y_pred)

    return best_model, X_test, y_test, results