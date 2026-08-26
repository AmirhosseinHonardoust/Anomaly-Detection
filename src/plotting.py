"""Chart generation for the anomaly-detection pipeline.

Split out of ``detect_anomalies.main()`` so the plotting logic can be unit
tested directly (input in, figure out) instead of only indirectly through
"does a PNG file exist" CLI assertions.
"""

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure


def plot_amount_over_time(
    df: pd.DataFrame, sample_size: int = 20000, random_state: int = 42
) -> Figure:
    """Line plot of transaction amount over time, downsampled for readability."""
    df_plot = df.sample(n=min(sample_size, len(df)), random_state=random_state).sort_values("date")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df_plot["date"], df_plot["amount"])
    ax.set_title("Transaction Amount over Time (sample)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Amount")
    fig.tight_layout()
    return fig


def plot_amount_histogram(df: pd.DataFrame, bins: int = 60) -> Figure:
    """Histogram of the transaction amount distribution."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df["amount"], bins=bins)
    ax.set_title("Amount Distribution")
    ax.set_xlabel("Amount")
    ax.set_ylabel("Count")
    fig.tight_layout()
    return fig


def save_report_figures(df: pd.DataFrame, outdir: str) -> None:
    """Render and save both report figures to ``outdir``, closing them after."""
    fig1 = plot_amount_over_time(df)
    fig1.savefig(f"{outdir}/fig_amount_time.png", dpi=160)
    plt.close(fig1)

    fig2 = plot_amount_histogram(df)
    fig2.savefig(f"{outdir}/fig_amount_hist.png", dpi=160)
    plt.close(fig2)
