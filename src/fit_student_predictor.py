import os
import warnings
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings('ignore')

import click
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
    np.random.seed(seed)
    set_config(transform_output="pandas")

    # Read in data & preprocessor
    print(f"\nLoading training data from {training_data}...")
    student_train = pd.read_csv(training_data)
    student_preprocessor = pickle.load(open(preprocessor, "rb"))

    # Validate training data for anomalous correlations
    print("\nValidating data for anomalous correlations...")
    student_train_ds = Dataset(student_train, label=TARGET, cat_features=[])

    check_feat_lab_corr = FeatureLabelCorrelation().add_condition_feature_pps_less_than(0.9)
    check_feat_lab_corr_result = check_feat_lab_corr.run(dataset=student_train_ds)

    check_feat_feat_corr = FeatureFeatureCorrelation().add_condition_max_number_of_pairs_above_threshold(threshold=0.92, n_pairs=0)
    check_feat_feat_corr_result = check_feat_feat_corr.run(dataset=student_train_ds)

    if not check_feat_lab_corr_result.passed_conditions():
        raise ValueError("Feature-Label correlation exceeds the maximum acceptable threshold.")

    if not check_feat_feat_corr_result.passed_conditions():
        raise ValueError("Feature-feature correlation exceeds the maximum acceptable threshold.")

    print("Correlation checks passed!")

    # Tune model (find optimal alpha for Ridge using cross-validation)
    print("\nTuning Ridge hyperparameters...")
    ridge = Ridge()
    student_tune_pipe = make_pipeline(student_preprocessor, ridge)

    param_dist = {
        "ridge__alpha": loguniform(1e-3, 1e3),
    }

    cv = 10
    student_tune_search = RandomizedSearchCV(
        estimator=student_tune_pipe,
        param_distributions=param_dist,
        n_iter=100,
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        random_state=seed
    )

    student_fit = student_tune_search.fit(
        student_train.drop(columns=[TARGET]),
        student_train[TARGET]
    )

    best_alpha = student_fit.best_params_["ridge__alpha"]
    best_score = -student_fit.best_score_
    print(f"Best alpha: {best_alpha:.4f}")
    print(f"Best CV MAE: {best_score:.3f}")

    print(f"\nSaving model...")

    # Save pipeline
    os.makedirs(pipeline_to, exist_ok=True)
    with open(os.path.join(pipeline_to, "student_pipeline.pickle"), 'wb') as f:
        pickle.dump(student_fit, f)
    print(f"Saved pipeline to {pipeline_to}/student_pipeline.pickle")

    # Create and save hyperparameter tuning plot
    os.makedirs(plot_to, exist_ok=True)
    accuracies_grid = pd.DataFrame(student_fit.cv_results_)

    accuracies_grid = (
        accuracies_grid[[
            "param_ridge__alpha",
            "mean_test_score",
            "std_test_score"
        ]]
        .assign(
            sem_test_score=accuracies_grid["std_test_score"] / cv**(1/2),
            sem_test_score_lower=lambda df: df["mean_test_score"] - (df["sem_test_score"]/2),
            sem_test_score_upper=lambda df: df["mean_test_score"] + (df["sem_test_score"]/2),
            neg_mae=lambda df: -df["mean_test_score"]
        )
        .rename(columns={"param_ridge__alpha": "alpha"})
        .drop(columns=["std_test_score"])
    )

    line_n_point = alt.Chart(accuracies_grid, width=600).mark_line(color="black").encode(
        x=alt.X("alpha:Q", scale=alt.Scale(type='log')).title("Alpha (log scale)"),
        y=alt.Y("neg_mae:Q").scale(zero=False).title("Mean Absolute Error")
    )

    plot = line_n_point + line_n_point.mark_circle(color='black')
    plot.save(os.path.join(plot_to, "student_tune_alpha.png"), scale_factor=2.0)
    print(f"Saved tuning plot to {plot_to}/student_tune_alpha.png")

    # Save best parameters
    params_df = pd.DataFrame([{"best_alpha": best_alpha, "best_cv_mae": best_score}])
    params_df.to_csv(os.path.join(pipeline_to, "best_params.csv"), index=False)
    print(f"Saved best parameters to {pipeline_to}/best_params.csv")

    print("\nModel fitting complete!")


if __name__ == '__main__':
    main()