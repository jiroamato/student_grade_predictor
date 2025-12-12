import pytest
import pandas as pd
from click.testing import CliRunner
from unittest.mock import MagicMock

from src.eda import main


class TestMain:
    """Tests for the EDA main function."""

    def test_main_reads_csv_with_correct_path(self, mocker, tmp_path, sample_train_df):
        """
        Test that main() reads the processed training CSV from correct path.

        Parameters
        ----------
        mocker : pytest_mock.MockerFixture
            The pytest-mock mocker fixture.
        tmp_path : Path
            Pytest fixture for temporary directory.
        sample_train_df : pd.DataFrame
            Sample training DataFrame fixture.
        """
        mock_read_csv = mocker.patch('pandas.read_csv', return_value=sample_train_df)

        mock_chart = MagicMock()
        mock_chart.properties.return_value = mock_chart
        mock_chart.save = MagicMock()

        # Mock altair_ally corr function
        mocker.patch('src.eda.aly.corr', return_value=mock_chart)

        # Mock altair.Chart for the target distribution plot
        mock_alt_chart = MagicMock()
        mock_alt_chart.mark_bar.return_value = mock_alt_chart
        mock_alt_chart.encode.return_value = mock_alt_chart
        mock_alt_chart.transform_calculate.return_value = mock_alt_chart
        mock_alt_chart.properties.return_value = mock_alt_chart
        mock_alt_chart.save = MagicMock()
        mocker.patch('src.eda.alt.Chart', return_value=mock_alt_chart)

        plot_dir = tmp_path / "figures"
        plot_dir.mkdir()

        runner = CliRunner()
        result = runner.invoke(main, [
            '--processed-training-data', 'data/processed/student_train.csv',
            '--plot-to', str(plot_dir)
        ])

        mock_read_csv.assert_called_once()
        assert 'student_train.csv' in str(mock_read_csv.call_args)

    @pytest.mark.parametrize("expected_plot", [
        "target_distribution.png",
        "correlation_heatmap.png",
    ])
    def test_main_saves_expected_plots(self, mocker, tmp_path, sample_train_df, expected_plot):
        """
        Test that main() saves the expected plot files.

        Parameters
        ----------
        mocker : pytest_mock.MockerFixture
            The pytest-mock mocker fixture.
        tmp_path : Path
            Pytest fixture for temporary directory.
        sample_train_df : pd.DataFrame
            Sample training DataFrame fixture.
        expected_plot : str
            Expected plot filename.
        """
        mocker.patch('pandas.read_csv', return_value=sample_train_df)

        save_calls = []

        def track_save(path, **kwargs):
            save_calls.append(path)

        mock_chart = MagicMock()
        mock_chart.properties.return_value = mock_chart
        mock_chart.save = track_save

        # Mock altair_ally corr function
        mocker.patch('src.eda.aly.corr', return_value=mock_chart)

        # Mock altair.Chart for the target distribution plot
        mock_alt_chart = MagicMock()
        mock_alt_chart.mark_bar.return_value = mock_alt_chart
        mock_alt_chart.encode.return_value = mock_alt_chart
        mock_alt_chart.transform_calculate.return_value = mock_alt_chart
        mock_alt_chart.properties.return_value = mock_alt_chart
        mock_alt_chart.save = track_save
        mocker.patch('src.eda.alt.Chart', return_value=mock_alt_chart)

        plot_dir = tmp_path / "figures"
        plot_dir.mkdir()

        runner = CliRunner()
        result = runner.invoke(main, [
            '--processed-training-data', 'data/processed/student_train.csv',
            '--plot-to', str(plot_dir)
        ])

        saved_files = [str(p) for p in save_calls]
        assert any(expected_plot in f for f in saved_files), \
            f"Expected {expected_plot} to be saved. Saved: {saved_files}"

    def test_main_handles_missing_directory(self, mocker, tmp_path, sample_train_df):
        """
        Test that main() creates the plot directory if it doesn't exist.

        Parameters
        ----------
        mocker : pytest_mock.MockerFixture
            The pytest-mock mocker fixture.
        tmp_path : Path
            Pytest fixture for temporary directory.
        sample_train_df : pd.DataFrame
            Sample training DataFrame fixture.
        """
        mocker.patch('pandas.read_csv', return_value=sample_train_df)

        mock_chart = MagicMock()
        mock_chart.properties.return_value = mock_chart
        mock_chart.save = MagicMock()

        # Mock altair_ally corr function
        mocker.patch('src.eda.aly.corr', return_value=mock_chart)

        # Mock altair.Chart for the target distribution plot
        mock_alt_chart = MagicMock()
        mock_alt_chart.mark_bar.return_value = mock_alt_chart
        mock_alt_chart.encode.return_value = mock_alt_chart
        mock_alt_chart.transform_calculate.return_value = mock_alt_chart
        mock_alt_chart.properties.return_value = mock_alt_chart
        mock_alt_chart.save = MagicMock()
        mocker.patch('src.eda.alt.Chart', return_value=mock_alt_chart)

        plot_dir = tmp_path / "new_figures_dir"

        runner = CliRunner()
        result = runner.invoke(main, [
            '--processed-training-data', 'data/processed/student_train.csv',
            '--plot-to', str(plot_dir)
        ])

        assert result.exit_code == 0