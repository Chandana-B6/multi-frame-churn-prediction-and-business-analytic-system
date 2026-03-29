import pandas as pd

def load_and_preprocess():
    df = pd.read_csv("data.csv")

    # Drop customerID
    df.drop("customerID", axis=1, inplace=True)

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')

    # Fill missing values
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # Convert categorical to numeric
    df = pd.get_dummies(df, drop_first=True)

    return df