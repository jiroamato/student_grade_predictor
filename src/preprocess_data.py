import click
import os
import numpy as np
import pandas as pd
import pandera.pandas as pa
import pickle
from sklearn.model_selection import train_test_split
from sklearn import set_config
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.compose import make_column_transformer


def create_schema():
    """
    Create and return the pandera validation schema for student data.

    This function defines the expected schema for the Student Performance
    dataset, including column types, value ranges, and data quality checks.

    Parameters
    ----------
    None

    Returns
    -------
    pa.DataFrameSchema
        A pandera schema object that can be used to validate student
        performance dataframes.

    Examples
    --------
    >>> schema = create_schema()
    >>> validated_df = schema.validate(student_df, lazy=True)
    """
    schema = pa.DataFrameSchema(
        {
            "G3": pa.Column(int, pa.Check.between(0, 20), nullable=False),
            "G1": pa.Column(int, pa.Check.between(0, 20), nullable=False),
            "G2": pa.Column(int, pa.Check.between(0, 20), nullable=False),
            "age": pa.Column(int, pa.Check.between(15, 22), nullable=False),
            "Medu": pa.Column(int, pa.Check.between(0, 4), nullable=False),
            "Fedu": pa.Column(int, pa.Check.between(0, 4), nullable=False),
            "traveltime": pa.Column(int, pa.Check.between(1, 4), nullable=False),
            "studytime": pa.Column(int, pa.Check.between(1, 4), nullable=False),
            "failures": pa.Column(int, pa.Check.between(0, 4), nullable=False),
            "famrel": pa.Column(int, pa.Check.between(1, 5), nullable=False),
            "freetime": pa.Column(int, pa.Check.between(1, 5), nullable=False),
            "goout": pa.Column(int, pa.Check.between(1, 5), nullable=False),
            "Dalc": pa.Column(int, pa.Check.between(1, 5), nullable=False),
            "Walc": pa.Column(int, pa.Check.between(1, 5), nullable=False),
            "health": pa.Column(int, pa.Check.between(1, 5), nullable=False),
            "absences": pa.Column(int, pa.Check.between(0, 100), nullable=False),
            "school": pa.Column(str, pa.Check.isin(["GP", "MS"]), nullable=False),
            "sex": pa.Column(str, pa.Check.isin(["M", "F"]), nullable=False),
            "address": pa.Column(str, pa.Check.isin(["U", "R"]), nullable=False),
            "famsize": pa.Column(str, pa.Check.isin(["LE3", "GT3"]), nullable=False),
            "Pstatus": pa.Column(str, pa.Check.isin(["T", "A"]), nullable=False),
            "schoolsup": pa.Column(str, pa.Check.isin(["yes", "no"]), nullable=False),
            "famsup": pa.Column(str, pa.Check.isin(["yes", "no"]), nullable=False),
            "paid": pa.Column(str, pa.Check.isin(["yes", "no"]), nullable=False),
            "activities": pa.Column(str, pa.Check.isin(["yes", "no"]), nullable=False),
            "nursery": pa.Column(str, pa.Check.isin(["yes", "no"]), nullable=False),
            "higher": pa.Column(str, pa.Check.isin(["yes", "no"]), nullable=False),
            "internet": pa.Column(str, pa.Check.isin(["yes", "no"]), nullable=False),
            "romantic": pa.Column(str, pa.Check.isin(["yes", "no"]), nullable=False),
            "Mjob": pa.Column(str, pa.Check.isin(["teacher", "health", "services", "at_home", "other"]), nullable=False),
            "Fjob": pa.Column(str, pa.Check.isin(["teacher", "health", "services", "at_home", "other"]), nullable=False),
            "reason": pa.Column(str, pa.Check.isin(["home", "reputation", "course", "other"]), nullable=False),
            "guardian": pa.Column(str, pa.Check.isin(["mother", "father", "other"]), nullable=False),
        },
        checks=[
            pa.Check(lambda df: ~df.duplicated().any(), error="Duplicate rows found."),
            pa.Check(lambda df: ~(df.isna().all(axis=1)).any(), error="Empty rows found.")
        ]
    )
    return schema


def create_preprocessor():
    """
    Create a column transformer for preprocessing student features.

    This function creates a scikit-learn ColumnTransformer that applies
    appropriate transformations to different feature types:
    - StandardScaler for numeric features (G1, G2, age)
    - RobustScaler for absences (handles outliers)
    - OneHotEncoder for binary and nominal categorical features

    Parameters
    ----------
    None

    Returns
    -------
    sklearn.compose.ColumnTransformer
        A fitted column transformer ready to transform feature matrices.

    Examples
    --------
    >>> preprocessor = create_preprocessor()
    >>> preprocessor.fit(X_train)
    >>> X_transformed = preprocessor.transform(X_test)
    """
    numeric_features = ["G1", "G2", "age"]
    absences = ["absences"]
    binary_features = [
        "school", "sex", "address", "famsize", "Pstatus",
        "schoolsup", "famsup", "paid", "activities",
        "nursery", "higher", "internet", "romantic"
    ]
    nominal_features = ["Mjob", "Fjob", "reason", "guardian"]

    preprocessor = make_column_transformer(
        (StandardScaler(), numeric_features),
        (RobustScaler(), absences),
        (OneHotEncoder(drop="if_binary", dtype=int, sparse_output=False), binary_features),
        (OneHotEncoder(handle_unknown="ignore", sparse_output=False), nominal_features),
        remainder="passthrough",
        verbose_feature_names_out=False
    )
    return preprocessor


@click.command()
@click.option('--raw-data', type=str, help="Path to raw data")
@click.option('--data-to', type=str, help="Path to directory where processed data will be written to")
@click.option('--preprocessor-to', type=str, help="Path to directory where the preprocessor object will be written to")
@click.option('--seed', type=int, help="Random seed", default=123)
def main(raw_data, data_to, preprocessor_to, seed):
    """
    Validate, split, and preprocess the student performance data.

    This script performs the following operations:
    1. Loads raw student data from CSV
    2. Validates data against a predefined schema
    3. Splits data into training (70%) and test (30%) sets
    4. Creates and saves a preprocessor for feature transformation
    5. Transforms and saves the processed datasets

    Parameters
    ----------
    raw_data : str
        Path to the raw CSV data file (semicolon-separated).
    data_to : str
        Path to directory where processed train/test data will be saved.
    preprocessor_to : str
        Path to directory where the preprocessor pickle file will be saved.
    seed : int, optional
        Random seed for reproducibility (default: 123).

    Returns
    -------
    None
        Outputs are saved to disk:
        - student_train.csv, student_test.csv
        - transformed_student_train.csv, transformed_student_test.csv
        - student_preprocessor.pickle
    """
    np.random.seed(seed)
    set_config(transform_output="pandas")

    print(f"Loading data from {raw_data}...")
    student_df = pd.read_csv(raw_data, sep=";")
    print(f"Loaded {len(student_df)} rows")

    print("\nValidating data against schema...")
    schema = create_schema()
    schema.validate(student_df, lazy=True)
    print("All validation checks passed!")

    student_train, student_test = train_test_split(
        student_df, train_size=0.70, random_state=seed
    )
    print(f"\nTrain set: {len(student_train)} rows")
    print(f"Test set: {len(student_test)} rows")

    os.makedirs(data_to, exist_ok=True)
    student_train.to_csv(os.path.join(data_to, "student_train.csv"), index=False)
    student_test.to_csv(os.path.join(data_to, "student_test.csv"), index=False)

    os.makedirs(preprocessor_to, exist_ok=True)
    student_preprocessor = create_preprocessor()
    pickle.dump(student_preprocessor, open(os.path.join(preprocessor_to, "student_preprocessor.pickle"), "wb"))

    student_preprocessor.fit(student_train.drop(columns=["G3"]))
    transformed_train = student_preprocessor.transform(student_train.drop(columns=["G3"]))
    transformed_test = student_preprocessor.transform(student_test.drop(columns=["G3"]))

    transformed_train["G3"] = student_train["G3"].values
    transformed_test["G3"] = student_test["G3"].values

    transformed_train.to_csv(os.path.join(data_to, "transformed_student_train.csv"), index=False)
    transformed_test.to_csv(os.path.join(data_to, "transformed_student_test.csv"), index=False)

    print(f"\nSaved training data to {data_to}")
    print(f"Saved preprocessor to {preprocessor_to}")
    print("\nData preprocessing complete!")


if __name__ == '__main__':
    main()