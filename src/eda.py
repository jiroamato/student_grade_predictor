import click
import os
import altair as alt
import altair_ally as aly
import pandas as pd


TARGET = "G3"

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
    # return None # stub
    ...


if __name__ == '__main__':
    main()