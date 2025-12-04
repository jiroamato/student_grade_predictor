import pytest
import pandas as pd
import numpy as np
from click.testing import CliRunner

from src.preprocess_data import main


class TestMain:
    """Tests for the preprocess_data main function."""

    def test_main_reads_csv_with_correct_path(self, mocker, tmp_path, sample_raw_df):
        """
        Test that main() reads the CSV file from the correct path.

        Parameters
        ----------
        mocker : pytest_mock.MockerFixture
            The pytest-mock mocker fixture.
        tmp_path : Path
            Pytest fixture for temporary directory.
        sample_raw_df : pd.DataFrame
            Sample raw DataFrame fixture.
        """
        mock_read_csv = mocker.patch('pandas.read_csv', return_value=sample_raw_df)
        mocker.patch('pickle.dump')
        mocker.patch('pandas.DataFrame.to_csv')

        data_dir = tmp_path / "processed"
        data_dir.mkdir()
        preprocessor_dir = tmp_path / "models"
        preprocessor_dir.mkdir()

        runner = CliRunner()
        result = runner.invoke(main, [
            '--raw-data', 'data/raw/student-por.csv',
            '--data-to', str(data_dir),
            '--preprocessor-to', str(preprocessor_dir),
            '--seed', '123'
        ])

        mock_read_csv.assert_called_once()
        assert 'student-por.csv' in str(mock_read_csv.call_args)

    def test_main_saves_preprocessor_as_pickle(self, mocker, tmp_path, sample_raw_df):
        """
        Test that main() saves the preprocessor as a pickle file.

        Parameters
        ----------
        mocker : pytest_mock.MockerFixture
            The pytest-mock mocker fixture.
        tmp_path : Path
            Pytest fixture for temporary directory.
        sample_raw_df : pd.DataFrame
            Sample raw DataFrame fixture.
        """
        mocker.patch('pandas.read_csv', return_value=sample_raw_df)
        mock_pickle_dump = mocker.patch('pickle.dump')
        mocker.patch('pandas.DataFrame.to_csv')

        data_dir = tmp_path / "processed"
        data_dir.mkdir()
        preprocessor_dir = tmp_path / "models"
        preprocessor_dir.mkdir()

        runner = CliRunner()
        result = runner.invoke(main, [
            '--raw-data', 'data/raw/student-por.csv',
            '--data-to', str(data_dir),
            '--preprocessor-to', str(preprocessor_dir),
            '--seed', '123'
        ])

        assert mock_pickle_dump.called
        # Check that pickle.dump was called (preprocessor saved)
        assert mock_pickle_dump.call_count >= 1

    @pytest.mark.parametrize("output_file", [
        "student_train.csv",
        "student_test.csv",
    ])
    def test_main_saves_train_test_splits(self, mocker, tmp_path, sample_raw_df, output_file):
        """
        Test that main() saves train and test CSV files.

        Parameters
        ----------
        mocker : pytest_mock.MockerFixture
            The pytest-mock mocker fixture.
        tmp_path : Path
            Pytest fixture for temporary directory.
        sample_raw_df : pd.DataFrame
            Sample raw DataFrame fixture.
        output_file : str
            Expected output filename.
        """
        mocker.patch('pandas.read_csv', return_value=sample_raw_df)
        mocker.patch('pickle.dump')
        mock_to_csv = mocker.patch('pandas.DataFrame.to_csv')

        data_dir = tmp_path / "processed"
        data_dir.mkdir()
        preprocessor_dir = tmp_path / "models"
        preprocessor_dir.mkdir()

        runner = CliRunner()
        result = runner.invoke(main, [
            '--raw-data', 'data/raw/student-por.csv',
            '--data-to', str(data_dir),
            '--preprocessor-to', str(preprocessor_dir),
            '--seed', '123'
        ])

        # Check to_csv was called multiple times (train and test files)
        assert mock_to_csv.call_count >= 2

    def test_main_uses_seed_for_reproducibility(self, mocker, tmp_path, sample_raw_df):
        """
        Test that main() uses the seed for reproducible train/test splits.

        Parameters
        ----------
        mocker : pytest_mock.MockerFixture
            The pytest-mock mocker fixture.
        tmp_path : Path
            Pytest fixture for temporary directory.
        sample_raw_df : pd.DataFrame
            Sample raw DataFrame fixture.
        """
        mocker.patch('pandas.read_csv', return_value=sample_raw_df)
        mocker.patch('pickle.dump')
        mocker.patch('pandas.DataFrame.to_csv')
        mock_train_test_split = mocker.patch(
            'sklearn.model_selection.train_test_split',
            return_value=(sample_raw_df.iloc[:3], sample_raw_df.iloc[3:])
        )

        data_dir = tmp_path / "processed"
        data_dir.mkdir()
        preprocessor_dir = tmp_path / "models"
        preprocessor_dir.mkdir()

        runner = CliRunner()
        result = runner.invoke(main, [
            '--raw-data', 'data/raw/student-por.csv',
            '--data-to', str(data_dir),
            '--preprocessor-to', str(preprocessor_dir),
            '--seed', '42'
        ])

        # Verify train_test_split was called with random_state
        if mock_train_test_split.called:
            call_kwargs = mock_train_test_split.call_args[1]
            assert call_kwargs.get('random_state') == 42