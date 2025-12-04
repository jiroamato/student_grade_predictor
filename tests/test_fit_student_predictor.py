import pytest
import pandas as pd
import numpy as np
from click.testing import CliRunner
from unittest.mock import MagicMock, mock_open

from src.fit_student_predictor import main


class TestMain:
    """Tests for the model fitting main function."""

    def test_main_reads_training_csv(self, mocker, tmp_path, sample_train_df, mock_preprocessor):
        """
        Test that main() reads the training CSV from correct path.

        Parameters
        ----------
        mocker : pytest_mock.MockerFixture
            The pytest-mock mocker fixture.
        tmp_path : Path
            Pytest fixture for temporary directory.
        sample_train_df : pd.DataFrame
            Sample training DataFrame fixture.
        mock_preprocessor : MagicMock
            Mock preprocessor fixture.
        """
        mock_read_csv = mocker.patch('pandas.read_csv', return_value=sample_train_df)
        mocker.patch('builtins.open', mock_open())
        mocker.patch('pickle.load', return_value=mock_preprocessor)
        mocker.patch('pickle.dump')

        # Mock deepchecks
        mock_check = MagicMock()
        mock_check.add_condition_feature_pps_less_than.return_value = mock_check
        mock_check.add_condition_max_number_of_pairs_above_threshold.return_value = mock_check
        mock_check_result = MagicMock()
        mock_check_result.passed_conditions.return_value = True
        mock_check.run.return_value = mock_check_result
        mocker.patch('deepchecks.tabular.checks.FeatureLabelCorrelation', return_value=mock_check)
        mocker.patch('deepchecks.tabular.checks.FeatureFeatureCorrelation', return_value=mock_check)
        mocker.patch('deepchecks.tabular.Dataset')

        # Mock sklearn
        mock_search = MagicMock()
        mock_search.fit.return_value = mock_search
        mock_search.best_params_ = {'ridge__alpha': 1.0}
        mock_search.best_score_ = -2.5
        mock_search.cv_results_ = {
            'param_ridge__alpha': [0.1, 1.0, 10.0],
            'mean_test_score': [-3.0, -2.5, -2.8],
            'std_test_score': [0.5, 0.4, 0.6]
        }
        mocker.patch('sklearn.model_selection.RandomizedSearchCV', return_value=mock_search)

        # Mock altair
        mock_chart = MagicMock()
        mocker.patch('altair.Chart', return_value=mock_chart)
        mock_chart.mark_line.return_value = mock_chart
        mock_chart.mark_circle.return_value = mock_chart
        mock_chart.encode.return_value = mock_chart
        mock_chart.__add__ = lambda self, other: mock_chart
        mock_chart.save = MagicMock()

        mocker.patch('pandas.DataFrame.to_csv')

        pipeline_dir = tmp_path / "models"
        pipeline_dir.mkdir()
        plot_dir = tmp_path / "figures"
        plot_dir.mkdir()

        runner = CliRunner()
        result = runner.invoke(main, [
            '--training-data', 'data/processed/student_train.csv',
            '--preprocessor', 'results/models/student_preprocessor.pickle',
            '--pipeline-to', str(pipeline_dir),
            '--plot-to', str(plot_dir),
            '--seed', '123'
        ])

        mock_read_csv.assert_called_once()

    def test_main_loads_preprocessor_pickle(self, mocker, tmp_path, sample_train_df, mock_preprocessor):
        """
        Test that main() loads the preprocessor from pickle file.
        """
        mocker.patch('pandas.read_csv', return_value=sample_train_df)
        mocker.patch('builtins.open', mock_open())
        mock_pickle_load = mocker.patch('pickle.load', return_value=mock_preprocessor)
        mocker.patch('pickle.dump')

        # Mock deepchecks
        mock_check = MagicMock()
        mock_check.add_condition_feature_pps_less_than.return_value = mock_check
        mock_check.add_condition_max_number_of_pairs_above_threshold.return_value = mock_check
        mock_check_result = MagicMock()
        mock_check_result.passed_conditions.return_value = True
        mock_check.run.return_value = mock_check_result
        mocker.patch('deepchecks.tabular.checks.FeatureLabelCorrelation', return_value=mock_check)
        mocker.patch('deepchecks.tabular.checks.FeatureFeatureCorrelation', return_value=mock_check)
        mocker.patch('deepchecks.tabular.Dataset')

        # Mock sklearn
        mock_search = MagicMock()
        mock_search.fit.return_value = mock_search
        mock_search.best_params_ = {'ridge__alpha': 1.0}
        mock_search.best_score_ = -2.5
        mock_search.cv_results_ = {
            'param_ridge__alpha': [0.1, 1.0, 10.0],
            'mean_test_score': [-3.0, -2.5, -2.8],
            'std_test_score': [0.5, 0.4, 0.6]
        }
        mocker.patch('sklearn.model_selection.RandomizedSearchCV', return_value=mock_search)

        # Mock altair
        mock_chart = MagicMock()
        mocker.patch('altair.Chart', return_value=mock_chart)
        mock_chart.mark_line.return_value = mock_chart
        mock_chart.mark_circle.return_value = mock_chart
        mock_chart.encode.return_value = mock_chart
        mock_chart.__add__ = lambda self, other: mock_chart
        mock_chart.save = MagicMock()

        mocker.patch('pandas.DataFrame.to_csv')

        pipeline_dir = tmp_path / "models"
        pipeline_dir.mkdir()
        plot_dir = tmp_path / "figures"
        plot_dir.mkdir()

        runner = CliRunner()
        result = runner.invoke(main, [
            '--training-data', 'data/processed/student_train.csv',
            '--preprocessor', 'results/models/student_preprocessor.pickle',
            '--pipeline-to', str(pipeline_dir),
            '--plot-to', str(plot_dir),
            '--seed', '123'
        ])

        mock_pickle_load.assert_called_once()

    def test_main_saves_pipeline_pickle(self, mocker, tmp_path, sample_train_df, mock_preprocessor):
        """
        Test that main() saves the fitted pipeline as a pickle file.
        """
        mocker.patch('pandas.read_csv', return_value=sample_train_df)
        mocker.patch('pickle.load', return_value=mock_preprocessor)

        # Mock deepchecks
        mock_check = MagicMock()
        mock_check.add_condition_feature_pps_less_than.return_value = mock_check
        mock_check.add_condition_max_number_of_pairs_above_threshold.return_value = mock_check
        mock_check_result = MagicMock()
        mock_check_result.passed_conditions.return_value = True
        mock_check.run.return_value = mock_check_result
        mocker.patch('src.fit_student_predictor.FeatureLabelCorrelation', return_value=mock_check)
        mocker.patch('src.fit_student_predictor.FeatureFeatureCorrelation', return_value=mock_check)
        mocker.patch('src.fit_student_predictor.Dataset')

        # Mock sklearn
        mock_search = MagicMock()
        mock_search.fit.return_value = mock_search
        mock_search.best_params_ = {'ridge__alpha': 1.0}
        mock_search.best_score_ = -2.5
        mock_search.cv_results_ = {
            'param_ridge__alpha': [0.1, 1.0, 10.0],
            'mean_test_score': [-3.0, -2.5, -2.8],
            'std_test_score': [0.5, 0.4, 0.6]
        }
        mocker.patch('src.fit_student_predictor.RandomizedSearchCV', return_value=mock_search)

        # Mock altair
        mock_chart = MagicMock()
        mock_chart.mark_line.return_value = mock_chart
        mock_chart.mark_circle.return_value = mock_chart
        mock_chart.encode.return_value = mock_chart
        mock_chart.__add__ = lambda self, other: mock_chart
        mock_chart.save = MagicMock()
        mocker.patch('src.fit_student_predictor.alt.Chart', return_value=mock_chart)

        pipeline_dir = tmp_path / "models"
        pipeline_dir.mkdir()
        plot_dir = tmp_path / "figures"
        plot_dir.mkdir()

        runner = CliRunner()
        result = runner.invoke(main, [
            '--training-data', 'data/processed/student_train.csv',
            '--preprocessor', 'results/models/student_preprocessor.pickle',
            '--pipeline-to', str(pipeline_dir),
            '--plot-to', str(plot_dir),
            '--seed', '123'
        ])

        # Check that the pipeline file was created
        assert (pipeline_dir / "student_pipeline.pickle").exists()

    def test_main_raises_error_on_failed_correlation_check(
        self, mocker, tmp_path, sample_train_df, mock_preprocessor
    ):
        """
        Test that main() raises ValueError when correlation checks fail.
        """
        mocker.patch('pandas.read_csv', return_value=sample_train_df)
        mocker.patch('builtins.open', mock_open())
        mocker.patch('pickle.load', return_value=mock_preprocessor)
        mocker.patch('deepchecks.tabular.Dataset')

        # Create failing check result
        failing_check = MagicMock()
        failing_check.add_condition_feature_pps_less_than.return_value = failing_check
        failing_check.add_condition_max_number_of_pairs_above_threshold.return_value = failing_check
        failing_result = MagicMock()
        failing_result.passed_conditions.return_value = False
        failing_check.run.return_value = failing_result

        mocker.patch('deepchecks.tabular.checks.FeatureLabelCorrelation', return_value=failing_check)
        mocker.patch('deepchecks.tabular.checks.FeatureFeatureCorrelation', return_value=failing_check)

        pipeline_dir = tmp_path / "models"
        pipeline_dir.mkdir()
        plot_dir = tmp_path / "figures"
        plot_dir.mkdir()

        runner = CliRunner()
        result = runner.invoke(main, [
            '--training-data', 'data/processed/student_train.csv',
            '--preprocessor', 'results/models/student_preprocessor.pickle',
            '--pipeline-to', str(pipeline_dir),
            '--plot-to', str(plot_dir),
            '--seed', '123'
        ])

        # Should fail due to correlation check
        assert result.exit_code != 0 or "correlation" in str(result.output).lower()