import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from lifelines import CoxPHFitter


def run_survival(df):
    print("\nRunning survival analysis...")

    # Copy dataset
    df_survival = df.copy()

    # Duration column
    df_survival['duration'] = df_survival['tenure']

    # 🔥 AUTO-DETECT CHURN COLUMN (FIXES YOUR ERROR)
    target_col = [col for col in df.columns if 'churn' in col.lower()]
    
    if len(target_col) == 0:
        raise Exception("❌ No Churn column found in dataset")
    
    target_col = target_col[0]
    df_survival['event'] = df_survival[target_col]

    # Select required columns
    df_survival = df_survival[['duration', 'event', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']]

    # Train Cox model
    cph = CoxPHFitter()
    cph.fit(df_survival, duration_col='duration', event_col='event')

    print("✅ Survival Analysis Done!")

    # Plot
    plot_survival(cph)


def plot_survival(cph):

    plt.figure(figsize=(12, 6))

    cph.plot()

    plt.title("Survival Analysis (Hazard Ratios)", fontsize=16)
    plt.xlabel("log(HR) (95% CI)", fontsize=12)
    plt.ylabel("Features", fontsize=12)

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.tight_layout()

    plt.savefig("survival_plot.png", dpi=300)
    plt.close()

    print("✅ Survival plot saved as survival_plot.png")