# src/download_data.py
import click
import os
import pandas as pd
import requests
from io import BytesIO
from zipfile import ZipFile


def read_zip(url, directory):
    """
    Download and extract a zip file from a URL to a local directory.

    This function downloads a zip file from the specified URL and extracts
    its contents directly to the target directory using in-memory processing.
    It handles nested zip files (e.g., student+performance.zip containing
    student.zip) and validates that the extracted CSV files are readable.

    Parameters
    ----------
    url : str
        The URL of the zip file to be downloaded. Must end with '.zip'.
    directory : str
        The local directory where files will be extracted.
        Must exist before calling this function.

    Returns
    -------
    None
        Files are extracted to the specified directory.

    Raises
    ------
    ValueError
        If the URL does not exist (HTTP status != 200).
        If the URL does not point to a zip file.
        If the directory does not exist.
        If extracted CSV files are not valid.

    Examples
    --------
    >>> read_zip(
    ...     "https://archive.ics.uci.edu/static/public/320/student+performance.zip",
    ...     "data/raw"
    ... )
    Validated: data/raw/student-mat.csv
    Validated: data/raw/student-por.csv
    """
    filename_from_url = os.path.basename(url)

    if not filename_from_url.endswith('.zip'):
        raise ValueError('The URL provided does not point to a zip file.')

    if not os.path.isdir(directory):
        raise ValueError('The directory provided does not exist.')

    request = requests.get(url)

    if request.status_code != 200:
        raise ValueError('The URL provided does not exist.')

    outer_zip_bytes = BytesIO(request.content)

    with ZipFile(outer_zip_bytes, 'r') as outer_zip:
        outer_zip.extractall(directory)

        if 'student.zip' in outer_zip.namelist():
            inner_zip_bytes = BytesIO(outer_zip.read('student.zip'))
            with ZipFile(inner_zip_bytes, 'r') as inner_zip:
                inner_zip.extractall(directory)

    csv_files = ["student-mat.csv", "student-por.csv"]
    for csv_file in csv_files:
        filepath = os.path.join(directory, csv_file)
        if os.path.exists(filepath):
            try:
                pd.read_csv(filepath, sep=";", nrows=5)
                print(f"Validated: {filepath}")
            except Exception as e:
                raise ValueError(f"File '{filepath}' is not a valid CSV: {e}")


@click.command()
@click.option('--url', type=str, help="URL of dataset to be downloaded")
@click.option('--write-to', type=str, help="Path to directory where raw data will be written to")
def main(url, write_to):
    """Download data zip from the web and extract it to a local directory."""
    try:
        read_zip(url, write_to)
    except ValueError:
        os.makedirs(write_to)
        read_zip(url, write_to)
    print("\nData download and extraction complete!")


if __name__ == '__main__':
    main()
