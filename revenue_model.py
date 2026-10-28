
import streamlit as st
import pandas as pd


def calculate_revenue_risk(df, model):
    """Estimate revenue at risk from predicted customer churn."""

    st.subheader("Revenue Risk Analysis")

    # Find a revenue-related column in the dataset
    possible_columns = [
        "MonthlyCharges",
        "monthly_charges",
        "MonthlyRevenue",
        "revenue",
        "Revenue",
        "TotalCharges",
        "total_charges",
    ]

    revenue_column = next(
        (col for col in possible_columns if col in df.columns),
        None,
    )

    if revenue_column is None:
        st.warning(
            "No recognized revenue column was found. "
            "Please check your dataset's column names."
        )
        return

    # Use the same features and preprocessing that the model expects.
    # The test-set features are not available here, so do not assume
    # the raw dataframe can be passed directly to every model.
    st.info(
        f"Revenue column detected: {revenue_column}. "
        "To calculate accurate risk, this function needs the "
        "same feature-preprocessing pipeline used during training."
    )
