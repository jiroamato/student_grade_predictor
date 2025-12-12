import os
import warnings
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings('ignore')

import click
import altair as alt
import altair_ally as aly
import pandas as pd


TARGET = "G3"
PASSING_GRADE = 10

@click.command()
@click.option('--processed-training-data', type=str, help="Path to processed training data")
@click.option('--plot-to', type=str, help="Path to directory where the plots will be written to")
def main(processed_training_data, plot_to):
    """
    Generate exploratory data analysis visualizations for student data.

    This script creates and saves the following visualizations
    (matching the original notebook figures):
    1. Distribution of Target Variable G3 (Figure 1)
    2. Pairwise Correlations Heatmap (Figure 2)

    Parameters
    ----------
    processed_training_data : str
        Path to the processed training data CSV file.
    plot_to : str
        Path to directory where plots will be saved as PNG files.

    Returns
    -------
    None
        The following files are saved to plot_to (results/figures/):
        - target_distribution.png
        - correlation_heatmap.png
    """
    aly.alt.data_transformers.enable('vegafusion')

    print(f"\nLoading data from {processed_training_data}...")
    student_train = pd.read_csv(processed_training_data)
    print(f"Loaded {len(student_train)} rows")

    os.makedirs(plot_to, exist_ok=True)

    print("\nCreating target distribution plot...")
    target_plot = alt.Chart(student_train[[TARGET]]).mark_bar().encode(
        x=alt.X(TARGET, type='quantitative', bin=alt.Bin(maxbins=30)),
        y=alt.Y('count()', scale=alt.Scale(domain=[0, 80])),
        color=alt.Color(
            "Grade:N",
            scale=alt.Scale(
                domain=["Pass", "Fail"],
                range=["steelblue", "firebrick"]
            )
        )
    ).transform_calculate(
        Grade=alt.expr.if_(alt.datum[TARGET] >= PASSING_GRADE, "Pass", "Fail")
    ).properties(
        title="Distribution of the target feature (G3)",
        width=400,
        height=300
    )
    target_plot.save(os.path.join(plot_to, "target_distribution.png"), scale_factor=2.0)
    print(f"Saved: {plot_to}/target_distribution.png")

    print("\nCreating correlation heatmap...")
    corr_plot = aly.corr(student_train.drop(columns=[TARGET]))
    corr_plot.save(os.path.join(plot_to, "correlation_heatmap.png"), scale_factor=2.0)
    print(f"Saved: {plot_to}/correlation_heatmap.png")

    print("\nEDA complete!")


if __name__ == '__main__':
    main()
