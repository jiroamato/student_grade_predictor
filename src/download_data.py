# src/download_data.py
import click
import os
import zipfile
import requests
import pandas as pd


def read_zip(url, directory):
    """
    Read a zip file from the given URL and extract its contents.

    Parameters:
    ----------
    url : str
        The URL of the zip file to be read.
    directory : str
        The directory where contents will be extracted.

    Returns:
    -------
    None
    """
    ...  # STUB - implement after tests are written


@click.command()
@click.option('--url', type=str, help="URL of dataset to be downloaded")
@click.option('--write-to', type=str, help="Path to directory where raw data will be written to")
def main(url, write_to):
    '''Downloads data zip from the web to a local filepath and extracts it.'''
    ...  # STUB - implement after tests are written


if __name__ == '__main__':
    main()
