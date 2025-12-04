import pytest
import os
import io
import zipfile
from click.testing import CliRunner

from src.download_data import read_zip, main


class TestReadZip:
    """Tests for the read_zip function."""

    @pytest.mark.parametrize("status_code,expected_error", [
        (404, "URL provided does not exist"),
        (500, "URL provided does not exist"),
    ])
    def test_read_zip_raises_error_for_bad_http_status(
        self, mocker, tmp_path, status_code, expected_error
    ):
        """
        Test that read_zip raises ValueError for non-200 HTTP responses.

        Parameters
        ----------
        mocker : pytest_mock.MockerFixture
            The pytest-mock mocker fixture.
        tmp_path : Path
            Pytest fixture for temporary directory.
        status_code : int
            The HTTP status code to simulate.
        expected_error : str
            The expected error message substring.
        """
        mock_get = mocker.patch('src.download_data.requests.get')
        mock_get.return_value.status_code = status_code

        with pytest.raises(ValueError, match=expected_error):
            read_zip("http://example.com/data.zip", str(tmp_path))

    def test_read_zip_raises_error_for_non_zip_url(self, tmp_path):
        """
        Test that read_zip raises ValueError when URL doesn't point to a zip file.

        Parameters
        ----------
        tmp_path : Path
            Pytest fixture for temporary directory.
        """
        with pytest.raises(ValueError, match="does not point to a zip file"):
            read_zip("http://example.com/file.txt", str(tmp_path))

    def test_read_zip_raises_error_for_nonexistent_directory(self):
        """Test that read_zip raises ValueError when directory doesn't exist."""
        with pytest.raises(ValueError, match="directory provided does not exist"):
            read_zip("http://example.com/data.zip", "/nonexistent/path")

    def test_read_zip_extracts_files_successfully(self, mocker, tmp_path, mock_zip_content):
        """
        Test that read_zip successfully extracts files from a valid zip.

        Parameters
        ----------
        mocker : pytest_mock.MockerFixture
            The pytest-mock mocker fixture.
        tmp_path : Path
            Pytest fixture for temporary directory.
        mock_zip_content : bytes
            Mock zip file content fixture.
        """
        mock_get = mocker.patch('src.download_data.requests.get')
        mock_get.return_value.status_code = 200
        mock_get.return_value.content = mock_zip_content

        read_zip("http://example.com/data.zip", str(tmp_path))

        assert (tmp_path / "student-por.csv").exists()


class TestMain:
    """Tests for the CLI main function."""

    def test_main_calls_read_zip_with_correct_arguments(self, mocker, tmp_path):
        """
        Test that main() passes correct arguments to read_zip.

        Parameters
        ----------
        mocker : pytest_mock.MockerFixture
            The pytest-mock mocker fixture.
        tmp_path : Path
            Pytest fixture for temporary directory.
        """
        mock_read_zip = mocker.patch('src.download_data.read_zip')

        runner = CliRunner()
        result = runner.invoke(main, [
            '--url', 'http://example.com/data.zip',
            '--write-to', str(tmp_path)
        ])

        assert result.exit_code == 0
        mock_read_zip.assert_called_once_with(
            "http://example.com/data.zip",
            str(tmp_path)
        )

    def test_main_creates_output_directory_if_missing(self, mocker, tmp_path, mock_zip_content):
        """
        Test that main() creates the output directory if it doesn't exist.

        Parameters
        ----------
        mocker : pytest_mock.MockerFixture
            The pytest-mock mocker fixture.
        tmp_path : Path
            Pytest fixture for temporary directory.
        mock_zip_content : bytes
            Mock zip file content fixture.
        """
        mock_get = mocker.patch('src.download_data.requests.get')
        mock_get.return_value.status_code = 200
        mock_get.return_value.content = mock_zip_content

        output_dir = tmp_path / "new_subdir"
        runner = CliRunner()
        result = runner.invoke(main, [
            '--url', 'http://example.com/data.zip',
            '--write-to', str(output_dir)
        ])

        assert result.exit_code == 0
        assert output_dir.exists()
