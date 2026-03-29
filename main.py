from preprocessing import load_and_preprocess
from churn_model import train_models
from survival_analysis import run_survival
from explainability import explain_model
from revenue_model import calculate_revenue_risk
from feature_importance import plot_feature_importance
from model_comparison import plot_model_comparison
from metrics_curve import plot_metrics_curve   # ✅ NEW

def main():
    print("Loading data...")
    df = load_and_preprocess()

    print("\nTraining multiple models...")
    model, X_test, y_test, results = train_models(df)

    print("\nPlotting model comparison...")
    plot_model_comparison(results)

    print("\nPlotting feature importance...")
    plot_feature_importance(model, X_test)

    print("\nPlotting metrics curves...")   # ✅ NEW
    plot_metrics_curve(model, X_test, y_test)

    print("\nRunning survival analysis...")
    run_survival(df)

    print("\nExplainable AI (SHAP)...")
    explain_model(model, X_test)

    print("\nRevenue Risk Calculation...")
    calculate_revenue_risk(df, model)

    print("\n✅ ALL TASKS COMPLETED SUCCESSFULLY!")
    print("Check generated files:")
    print("- roc_curve.png")
    print("- model_comparison.png")
    print("- feature_importance.png")
    print("- shap_summary.png")
    print("- survival_plot.png")
    print("- roc_curve_detailed.png")        # ✅ NEW
    print("- precision_recall_curve.png")    # ✅ NEW

if __name__ == "__main__":
    main()