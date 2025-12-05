import pytest
import pandas as pd
import numpy as np
from click.testing import CliRunner
from unittest.mock import MagicMock, mock_open

from src.evaluate_student_predictor import main


class TestMain:
    """Tests for the model evaluation main function."""

    def test_main_reads_test_csv(self, mocker, tmp_path, sample_test_df, mock_pipeline):
        """
        Test that main() reads the test CSV from correct path.
        """
        mock_read_csv = mocker.patch('pandas.read_csv', return_value=sample_test_df)
        mocker.patch('builtins.open', mock_open())
        mocker.patch('pickle.load', return_value=mock_pipeline)
        mocker.patch('pandas.DataFrame.to_csv')
        mocker.patch('matplotlib.pyplot.savefig')
        mocker.patch('matplotlib.pyplot.close')

        tables_dir = tmp_path / "tables"
        tables_dir.mkdir()
        figures_dir = tmp_path / "figures"
        figures_dir.mkdir()

        runner = CliRunner()
        result = runner.invoke(main, [
            '--test-data', 'data/processed/student_test.csv',
            '--pipeline-from', 'results/models/student_pipeline.pickle',
            '--tables-to', str(tables_dir),
            '--plot-to', str(figures_dir),
            '--seed', '123'
        ])

        mock_read_csv.assert_called_once()
        assert 'student_test.csv' in str(mock_read_csv.call_args)

    def test_main_loads_pipeline_pickle(self, mocker, tmp_path, sample_test_df, mock_pipeline):
        """
        Test that main() loads the fitted pipeline from pickle file.
        """
        mocker.patch('pandas.read_csv', return_value=sample_test_df)
        mocker.patch('builtins.open', mock_open())
        mock_pickle_load = mocker.patch('pickle.load', return_value=mock_pipeline)
        mocker.patch('pandas.DataFrame.to_csv')
        mocker.patch('matplotlib.pyplot.savefig')
        mocker.patch('matplotlib.pyplot.close')

        tables_dir = tmp_path / "tables"
        tables_dir.mkdir()
        figures_dir = tmp_path / "figures"
        figures_dir.mkdir()

        runner = CliRunner()
        result = runner.invoke(main, [
            '--test-data', 'data/processed/student_test.csv',
            '--pipeline-from', 'results/models/student_pipeline.pickle',
            '--tables-to', str(tables_dir),
            '--plot-to', str(figures_dir),
            '--seed', '123'
        ])

        mock_pickle_load.assert_called_once()

    def test_main_calls_predict_on_pipeline(self, mocker, tmp_path, sample_test_df, mock_pipeline):
        """
        Test that main() calls predict on the loaded pipeline.
        """
        mocker.patch('pandas.read_csv', return_value=sample_test_df)
        mocker.patch('builtins.open', mock_open())
        mocker.patch('pickle.load', return_value=mock_pipeline)
        mocker.patch('pandas.DataFrame.to_csv')
        mocker.patch('matplotlib.pyplot.savefig')
        mocker.patch('matplotlib.pyplot.close')

        tables_dir = tmp_path / "tables"
        tables_dir.mkdir()
        figures_dir = tmp_path / "figures"
        figures_dir.mkdir()

        runner = CliRunner()
        result = runner.invoke(main, [
            '--test-data', 'data/processed/student_test.csv',
            '--pipeline-from', 'results/models/student_pipeline.pickle',
            '--tables-to', str(tables_dir),
            '--plot-to', str(figures_dir),
            '--seed', '123'
        ])

        mock_pipeline.predict.assert_called_once()

    @pytest.mark.parametrize("output_file", [
    "test_scores.csv",
    "top_coefficients.csv",
    ])
    def test_main_saves_output_files(self, mocker, tmp_path, sample_test_df, mock_pipeline, output_file):
        """
        Test that main() saves the expected output CSV files.
        """
        mocker.patch('pandas.read_csv', return_value=sample_test_df)
        mocker.patch('pickle.load', return_value=mock_pipeline)
        mocker.patch('src.evaluate_student_predictor.plt.savefig')
        mocker.patch('src.evaluate_student_predictor.plt.close')
        mocker.patch('src.evaluate_student_predictor.plt.subplots', return_value=(MagicMock(), MagicMock()))
        mocker.patch('src.evaluate_student_predictor.PredictionErrorDisplay')

        # Mock the Pipeline used for coefficient extraction
        mock_preprocessor_pipeline = MagicMock()
        mock_preprocessor_pipeline.transform.return_value = pd.DataFrame({
            'feature1': [0.1, 0.2],
            'feature2': [0.3, 0.4],
            'feature3': [0.5, 0.6],
            'feature4': [0.7, 0.8],
            'feature5': [0.9, 1.0]
        })
        mocker.patch('src.evaluate_student_predictor.Pipeline', return_value=mock_preprocessor_pipeline)

        # Track to_csv calls
        to_csv_calls = []

        def mock_to_csv(self, path_or_buf=None, *args, **kwargs):
            if path_or_buf is not None:
                to_csv_calls.append(str(path_or_buf))

        mocker.patch('pandas.DataFrame.to_csv', mock_to_csv)

        tables_dir = tmp_path / "tables"
        tables_dir.mkdir()
        figures_dir = tmp_path / "figures"
        figures_dir.mkdir()

        runner = CliRunner()
        result = runner.invoke(main, [
            '--test-data', 'data/processed/student_test.csv',
            '--pipeline-from', 'results/models/student_pipeline.pickle',
            '--tables-to', str(tables_dir),
            '--plot-to', str(figures_dir),
            '--seed', '123'
        ])

        # Check output file was saved
        assert any(output_file in path for path in to_csv_calls), \
            f"Expected {output_file} to be saved. Saved: {to_csv_calls}"


    def test_main_saves_prediction_error_plot(self, mocker, tmp_path, sample_test_df, mock_pipeline):
        """
        Test that main() saves the prediction error plot.
        """
        mocker.patch('pandas.read_csv', return_value=sample_test_df)
        mocker.patch('pickle.load', return_value=mock_pipeline)
        mocker.patch('pandas.DataFrame.to_csv')
        mock_savefig = mocker.patch('src.evaluate_student_predictor.plt.savefig')
        mocker.patch('src.evaluate_student_predictor.plt.close')
        mocker.patch('src.evaluate_student_predictor.plt.subplots', return_value=(MagicMock(), MagicMock()))
        mocker.patch('src.evaluate_student_predictor.plt.tight_layout')
        mocker.patch('src.evaluate_student_predictor.PredictionErrorDisplay')

        # Mock the Pipeline used for coefficient extraction
        mock_preprocessor_pipeline = MagicMock()
        mock_preprocessor_pipeline.transform.return_value = pd.DataFrame({
            'feature1': [0.1, 0.2],
            'feature2': [0.3, 0.4],
            'feature3': [0.5, 0.6],
            'feature4': [0.7, 0.8],
            'feature5': [0.9, 1.0]
        })
        mocker.patch('src.evaluate_student_predictor.Pipeline', return_value=mock_preprocessor_pipeline)

        tables_dir = tmp_path / "tables"
        tables_dir.mkdir()
        figures_dir = tmp_path / "figures"
        figures_dir.mkdir()

        runner = CliRunner()
        result = runner.invoke(main, [
            '--test-data', 'data/processed/student_test.csv',
            '--pipeline-from', 'results/models/student_pipeline.pickle',
            '--tables-to', str(tables_dir),
            '--plot-to', str(figures_dir),
            '--seed', '123'
        ])

        mock_savefig.assert_called_once()
        assert 'prediction_error.png' in str(mock_savefig.call_args)