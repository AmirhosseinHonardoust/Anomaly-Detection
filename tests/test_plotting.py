import matplotlib

matplotlib.use("Agg")  # headless backend for CI/sandbox, no display needed

import pandas as pd
from src.plotting import plot_amount_histogram, plot_amount_over_time, save_report_figures


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=10, freq="D"),
            "amount": [10.0, 12.0, 9.5, 11.0, 500.0, 13.0, 14.0, 10.5, 9.0, 12.5],
        }
    )


def test_plot_amount_over_time_returns_figure_with_one_axes():
    fig = plot_amount_over_time(_sample_df())
    assert len(fig.axes) == 1
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Date"
    assert ax.get_ylabel() == "Amount"


def test_plot_amount_over_time_downsamples_when_larger_than_sample_size():
    df = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=100, freq="D"),
            "amount": range(100),
        }
    )
    fig = plot_amount_over_time(df, sample_size=10)
    line = fig.axes[0].get_lines()[0]
    assert len(line.get_xdata()) == 10


def test_plot_amount_histogram_returns_figure_with_one_axes():
    fig = plot_amount_histogram(_sample_df())
    assert len(fig.axes) == 1
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Amount"
    assert ax.get_ylabel() == "Count"


def test_save_report_figures_writes_both_pngs(tmp_path):
    save_report_figures(_sample_df(), str(tmp_path))
    assert (tmp_path / "fig_amount_time.png").exists()
    assert (tmp_path / "fig_amount_hist.png").exists()
