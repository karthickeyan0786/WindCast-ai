"""
WindCast AI — Exploratory Data Analysis
=========================================
Generates the EDA figures into outputs/figures/ and prints the insight
each figure supports. Run: python src/eda.py
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(str(Path(__file__).resolve().parent))
from data_pipeline import load_dataset, FEATURE_COLUMNS, TARGET

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "outputs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


def run():
    df = load_dataset(ROOT / "data" / "TexasTurbine.csv")
    cols = ["wind_speed", "wind_direction", "pressure", "temperature", TARGET]

    # 1. Histograms — why: shows the SHAPE of each variable's distribution
    # (is it normal, skewed, bimodal?). Power generation is strongly
    # right-skewed with a spike near zero (calm periods / cut-in speed not
    # yet reached) — worth calling out explicitly in a review.
    df[cols].hist(figsize=(14, 8), bins=40)
    plt.suptitle("Feature Distributions")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_histograms.png", dpi=120)
    plt.close()

    # 2. Boxplots — why: quickly reveals outliers and spread per feature
    # on a common visual scale; used here to justify the IQR clipping step
    # in the cleaning pipeline.
    fig, axes = plt.subplots(1, len(cols), figsize=(18, 5))
    for ax, c in zip(axes, cols):
        sns.boxplot(y=df[c], ax=ax, color="#006194")
        ax.set_title(c)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_boxplots.png", dpi=120)
    plt.close()

    # 3. Correlation heatmap — why: shows LINEAR relationships between
    # every pair of variables at a glance. Expect wind_speed and
    # wind_speed_cubed to correlate most strongly with power — this is
    # the number-one figure faculty ask about, because it previews which
    # feature will dominate model importance later.
    plt.figure(figsize=(9, 7))
    corr = df[FEATURE_COLUMNS + [TARGET]].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="viridis")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_correlation_heatmap.png", dpi=120)
    plt.close()

    # 4. Scatterplots vs target — why: histograms/correlation only show
    # marginal shape and LINEAR strength; a scatterplot exposes the actual
    # functional relationship — here, the classic wind turbine power curve
    # (S-shaped: near-zero below cut-in, then a steep rise, then a flat
    # rated-power plateau at high wind speed).
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, c in zip(axes, ["wind_speed", "wind_direction", "temperature"]):
        ax.scatter(df[c], df[TARGET], s=4, alpha=0.3, color="#006c4a")
        ax.set_xlabel(c)
        ax.set_ylabel(TARGET)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "04_scatter_vs_target.png", dpi=120)
    plt.close()

    print("Saved 4 figures to", FIG_DIR)
    print("\nKey talking points for a review:")
    print("- Power generated is right-skewed with a mass near 0 kW (calm hours).")
    print(f"- wind_speed correlates with power at r={corr.loc['wind_speed', TARGET]:.2f}")
    print(f"- wind_speed_cubed correlates with power at r={corr.loc['wind_speed_cubed', TARGET]:.2f}")
    print("- The wind_speed vs power scatterplot shows the textbook turbine")
    print("  power curve shape, which is WHY a purely linear model struggles")
    print("  and why the cubed feature / tree ensembles help.")


if __name__ == "__main__":
    run()
