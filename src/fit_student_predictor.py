import click
import os
import altair as alt
import numpy as np
import pandas as pd
import pickle
from deepchecks.tabular.checks import FeatureLabelCorrelation, FeatureFeatureCorrelation
from deepchecks.tabular import Dataset
from sklearn import set_config
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="deepchecks")

TARGET = "G3"


@click.command()
@click.option('--training-data', type=str, help="Path to training data")
@click.option('--preprocessor', type=str, help="Path to preprocessor object")
@click.option('--pipeline-to', type=str, help="Path to directory where the pipeline object will be written to")
@click.option('--plot-to', type=str, help="Path to directory where the plot will be written to")
@click.option('--seed', type=int, help="Random seed", default=123)
def main(training_data, preprocessor, pipeline_to, plot_to, seed):
    """
    Fit a Ridge regression model to the training data and save the pipeline.

    Parameters
    ----------
    training_data : str
        Path to the preprocessed training data CSV file.
    preprocessor : str
        Path to the pickled preprocessor object.
    pipeline_to : str
        Path to directory where the pipeline object will be written.
    plot_to : str
        Path to directory where the tuning plot will be written.
    seed : int
        Random seed for reproducibility. Default is 123.

    Returns
    -------
    None
        Saves the fitted pipeline and tuning plot.

    Raises
    ------
    ValueError
        If correlation checks fail.
    """
    # return None # stub
    ...


if __name__ == '__main__':
    main()