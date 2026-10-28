
import streamlit as st
import os
import main

st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction")
st.subheader("Multi-Frame Churn Prediction and Business Analytics System")

st.write(
    "Analyze customer churn using machine learning, "
    "survival analysis, explainable AI, and revenue-risk analysis."
)

st.info(
    "This demo runs the existing analysis pipeline. "
    "Make sure your dataset and required Python packages are available."
)

if st.button("Run Churn Analysis"):
    with st.spinner("Running analysis. Please wait..."):
        try:
            main.main()
            st.success("Analysis completed!")

            st.subheader("Generated Analysis Charts")

            charts = [
                "model_comparison.png",
                "feature_importance.png",
                "roc_curve.png",
                "roc_curve_detailed.png",
                "precision_recall_curve.png",
                "shap_summary.png",
                "survival_plot.png",
            ]

            for chart in charts:
                if os.path.exists(chart):
                    st.image(chart, caption=chart, use_container_width=True)

        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.info(
                "Check the dataset, dependencies, and application logs."
            )
