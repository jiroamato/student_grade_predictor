import pytest
import io
import zipfile
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

@pytest.fixture
def sample_train_df():
    """
    Create a sample training DataFrame for testing EDA and model scripts.

    Returns
    -------
    pd.DataFrame
        Sample DataFrame matching the processed student dataset schema.
    """
    return pd.DataFrame({
        'school': ['GP', 'GP', 'MS', 'GP', 'MS'],
        'sex': ['F', 'M', 'F', 'M', 'F'],
        'age': [18, 17, 16, 18, 19],
        'address': ['U', 'R', 'U', 'U', 'R'],
        'famsize': ['GT3', 'LE3', 'GT3', 'GT3', 'LE3'],
        'Pstatus': ['T', 'T', 'A', 'T', 'T'],
        'Medu': [4, 2, 3, 4, 1],
        'Fedu': [4, 2, 3, 4, 1],
        'Mjob': ['teacher', 'other', 'services', 'health', 'at_home'],
        'Fjob': ['teacher', 'other', 'services', 'health', 'at_home'],
        'reason': ['home', 'reputation', 'course', 'other', 'home'],
        'guardian': ['mother', 'father', 'mother', 'mother', 'other'],
        'traveltime': [1, 2, 1, 1, 3],
        'studytime': [2, 3, 2, 4, 1],
        'failures': [0, 0, 1, 0, 2],
        'schoolsup': ['no', 'yes', 'no', 'no', 'yes'],
        'famsup': ['yes', 'no', 'yes', 'yes', 'no'],
        'paid': ['no', 'no', 'yes', 'no', 'no'],
        'activities': ['yes', 'no', 'yes', 'yes', 'no'],
        'nursery': ['yes', 'yes', 'no', 'yes', 'yes'],
        'higher': ['yes', 'yes', 'yes', 'yes', 'no'],
        'internet': ['yes', 'no', 'yes', 'yes', 'no'],
        'romantic': ['no', 'yes', 'no', 'no', 'yes'],
        'famrel': [4, 3, 4, 5, 2],
        'freetime': [3, 4, 3, 2, 5],
        'goout': [3, 2, 4, 3, 5],
        'Dalc': [1, 2, 1, 1, 3],
        'Walc': [1, 3, 2, 1, 4],
        'health': [3, 4, 5, 3, 2],
        'absences': [4, 2, 10, 0, 25],
        'G1': [10, 12, 8, 15, 6],
        'G2': [11, 13, 9, 16, 5],
        'G3': [12, 14, 10, 17, 4]
    })


@pytest.fixture
def sample_test_df():
    """
    Create a sample test DataFrame for testing model evaluation.

    Returns
    -------
    pd.DataFrame
        Sample DataFrame matching the processed student dataset schema.
    """
    return pd.DataFrame({
        'school': ['GP', 'MS'],
        'sex': ['F', 'M'],
        'age': [18, 17],
        'address': ['U', 'R'],
        'famsize': ['GT3', 'LE3'],
        'Pstatus': ['T', 'T'],
        'Medu': [4, 2],
        'Fedu': [4, 2],
        'Mjob': ['teacher', 'other'],
        'Fjob': ['teacher', 'other'],
        'reason': ['home', 'reputation'],
        'guardian': ['mother', 'father'],
        'traveltime': [1, 2],
        'studytime': [2, 3],
        'failures': [0, 0],
        'schoolsup': ['no', 'yes'],
        'famsup': ['yes', 'no'],
        'paid': ['no', 'no'],
        'activities': ['yes', 'no'],
        'nursery': ['yes', 'yes'],
        'higher': ['yes', 'yes'],
        'internet': ['yes', 'no'],
        'romantic': ['no', 'yes'],
        'famrel': [4, 3],
        'freetime': [3, 4],
        'goout': [3, 2],
        'Dalc': [1, 2],
        'Walc': [1, 3],
        'health': [3, 4],
        'absences': [4, 2],
        'G1': [10, 12],
        'G2': [11, 13],
        'G3': [12, 14]
    })

@pytest.fixture
def sample_raw_df():
    """
    Create a sample raw DataFrame for testing preprocess_data.

    Returns
    -------
    pd.DataFrame
        Sample DataFrame matching the student dataset schema.
    """
    return pd.DataFrame({
        'school': ['GP', 'GP', 'MS', 'GP', 'MS'],
        'sex': ['F', 'M', 'F', 'M', 'F'],
        'age': [18, 17, 16, 18, 19],
        'address': ['U', 'R', 'U', 'U', 'R'],
        'famsize': ['GT3', 'LE3', 'GT3', 'GT3', 'LE3'],
        'Pstatus': ['T', 'T', 'A', 'T', 'T'],
        'Medu': [4, 2, 3, 4, 1],
        'Fedu': [4, 2, 3, 4, 1],
        'Mjob': ['teacher', 'other', 'services', 'health', 'at_home'],
        'Fjob': ['teacher', 'other', 'services', 'health', 'at_home'],
        'reason': ['home', 'reputation', 'course', 'other', 'home'],
        'guardian': ['mother', 'father', 'mother', 'mother', 'other'],
        'traveltime': [1, 2, 1, 1, 3],
        'studytime': [2, 3, 2, 4, 1],
        'failures': [0, 0, 1, 0, 2],
        'schoolsup': ['no', 'yes', 'no', 'no', 'yes'],
        'famsup': ['yes', 'no', 'yes', 'yes', 'no'],
        'paid': ['no', 'no', 'yes', 'no', 'no'],
        'activities': ['yes', 'no', 'yes', 'yes', 'no'],
        'nursery': ['yes', 'yes', 'no', 'yes', 'yes'],
        'higher': ['yes', 'yes', 'yes', 'yes', 'no'],
        'internet': ['yes', 'no', 'yes', 'yes', 'no'],
        'romantic': ['no', 'yes', 'no', 'no', 'yes'],
        'famrel': [4, 3, 4, 5, 2],
        'freetime': [3, 4, 3, 2, 5],
        'goout': [3, 2, 4, 3, 5],
        'Dalc': [1, 2, 1, 1, 3],
        'Walc': [1, 3, 2, 1, 4],
        'health': [3, 4, 5, 3, 2],
        'absences': [4, 2, 10, 0, 25],
        'G1': [10, 12, 8, 15, 6],
        'G2': [11, 13, 9, 16, 5],
        'G3': [12, 14, 10, 17, 4]
    })

@pytest.fixture
def mock_preprocessor():
    """
    Create a mock preprocessor for testing model fitting.

    Returns
    -------
    MagicMock
        Mock preprocessor object.
    """
    mock = MagicMock()
    mock.fit.return_value = mock
    mock.transform.return_value = pd.DataFrame({
        'G1': [10, 12, 8, 15, 6],
        'G2': [11, 13, 9, 16, 5],
        'age': [18, 17, 16, 18, 19]
    })
    return mock
  
@pytest.fixture
def mock_pipeline():
    """
    Create a mock pipeline for testing model evaluation.

    Returns
    -------
    MagicMock
        Mock pipeline object with predict method and named_steps.
    """
    mock = MagicMock()
    mock.predict.return_value = np.array([12.0, 14.0])

    # Mock preprocessor step
    mock_preprocessor = MagicMock()
    mock_preprocessor.transform.return_value = pd.DataFrame({
        'feature1': [0.1, 0.2],
        'feature2': [0.3, 0.4],
        'feature3': [0.5, 0.6],
        'feature4': [0.7, 0.8],
        'feature5': [0.9, 1.0]
    })

    # Mock ridge step with coefficients
    mock_ridge = MagicMock()
    mock_ridge.coef_ = np.array([0.8, 0.5, 0.3, -0.2, -0.4])

    # Set up best_estimator_ with named_steps (for RandomizedSearchCV)
    mock.best_estimator_ = MagicMock()
    mock.best_estimator_.named_steps = {
        'preprocessor': mock_preprocessor,
        'ridge': mock_ridge
    }

    return mock

@pytest.fixture
def mock_zip_content():
    """
    Create a mock zip file containing a nested student.zip with CSV files.

    Returns
    -------
    bytes
        Bytes content of the outer zip file.
    """
    csv_content = b'"school";"sex";"age"\n"GP";"F";18'

    # Create inner zip (student.zip)
    inner_zip_buffer = io.BytesIO()
    with zipfile.ZipFile(inner_zip_buffer, 'w') as inner_zip:
        inner_zip.writestr("student-mat.csv", csv_content)
        inner_zip.writestr("student-por.csv", csv_content)
    inner_zip_buffer.seek(0)

    # Create outer zip containing student.zip
    outer_zip_buffer = io.BytesIO()
    with zipfile.ZipFile(outer_zip_buffer, 'w') as outer_zip:
        outer_zip.writestr("student.zip", inner_zip_buffer.read())
    outer_zip_buffer.seek(0)

    return outer_zip_buffer.read()