import click
import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn import set_config
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    PredictionErrorDisplay,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

TARGET = "G3"


@click.command()
@click.option('--test-data', type=str, help="Path to test data")
@click.option('--pipeline-from', type=str, help="Path to the fit pipeline pickle file")
@click.option('--tables-to', type=str, help="Path to directory where table results will be written to")
@click.option('--plot-to', type=str, help="Path to directory where plots will be written to")
@click.option('--seed', type=int, help="Random seed", default=123)
def main(test_data, pipeline_from, tables_to, plot_to, seed):
    """
    Evaluate the student grade predictor on test data and save results.

    Parameters
    ----------
    test_data : str
        Path to the test data CSV file.
    pipeline_from : str
        Path to the pickled pipeline object from training.
    tables_to : str
        Path to directory where table results will be written.
    plot_to : str
        Path to directory where plots will be written.
    seed : int
        Random seed for reproducibility. Default is 123.

    Returns
    -------
    None
        Saves test scores, predictions, and prediction error plot.
    """
    np.random.seed(seed)
    set_config(transform_output="pandas")

    # Read in data & pipeline object
    print(f"\nLoading test data from {test_data}...")
    student_test = pd.read_csv(test_data)

    print(f"Loading pipeline from {pipeline_from}...")
    with open(pipeline_from, 'rb') as f:
        final_model_pipe = pickle.load(f)

    # Separate features and target
    X_test = student_test.drop(columns=[TARGET])
    y_test = student_test[TARGET]

    print("\nResults:")

    print(f"Test set: {len(X_test)} samples")

    # Generate predictions
    y_pred = final_model_pipe.predict(X_test)

    # Compute metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"Test MAE: {mae:.3f}")
    print(f"Test RMSE: {rmse:.3f}")
    print(f"Test R2: {r2:.3f}")

    # Save test scores to results/tables/
    os.makedirs(tables_to, exist_ok=True)
    test_scores = pd.DataFrame({
        'MAE': [mae],
        'RMSE': [rmse],
        'R2': [r2]
    })
    test_scores.to_csv(os.path.join(tables_to, "test_scores.csv"), index=False)
    print(f"\nSaved test scores to {tables_to}/test_scores.csv")

    # Extract and save top 5 ridge coefficients
    # Get transformed feature names by applying preprocessing steps
    preprocessing_steps = list(final_model_pipe.best_estimator_.named_steps.items())[:-1]
    preprocessor_pipeline = Pipeline(preprocessing_steps)
    X_test_transformed = preprocessor_pipeline.transform(X_test)
    feature_names = X_test_transformed.columns

    ridge_coeffs = pd.DataFrame(
        data=final_model_pipe.best_estimator_.named_steps['ridge'].coef_,
        index=feature_names,
        columns=['Coefficient']
    ).sort_values(by='Coefficient', ascending=False).head(5)
    ridge_coeffs.to_csv(os.path.join(tables_to, "top_coefficients.csv"))
    print(f"Saved top 5 coefficients to {tables_to}/top_coefficients.csv")

    # Create and save prediction error plot (Figure 6 - Residuals)
    os.makedirs(plot_to, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    PredictionErrorDisplay.from_estimator(
        final_model_pipe,
        X_test,
        y_test,
        ax=ax,
        scatter_kwargs={'alpha': 0.5, 's': 20}
    )
    ax.set_title('Residuals vs Predicted Values')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_to, "prediction_error.png"), dpi=150)
    plt.close()
    print(f"Saved prediction error plot to {plot_to}/prediction_error.png")

    print("\nModel evaluation complete!")


if __name__ == '__main__':
    main()