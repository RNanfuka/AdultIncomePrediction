import click
import os
import sys
from pathlib import Path
from typing import Optional

import requests

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def download(url: str, write_to: str, target_name: Optional[str] = None) -> Path:
    """Download a file from ``url`` and extract it if it is a ZIP archive."""
    destination_dir = Path(write_to)
    destination_dir.mkdir(parents=True, exist_ok=True)

    filename = target_name or (Path(url).name if Path(url).name else "dataset")
    download_path = destination_dir / filename

    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    with download_path.open("wb") as file_obj:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file_obj.write(chunk)


    return destination_dir

@click.command()
@click.option(
    "--url",
    type=str,
    default="https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
    show_default=True,
    help="URL of dataset to be downloaded",
)
@click.option(
    "--write_to",
    type=str,
    default="data/raw",
    show_default=True,
    help="Path to directory where raw data will be written to",
)
@click.option(
    "--target-name",
    type=str,
    default="adult.csv",
    help="Optional filename for the downloaded file (default: infer from URL).",
)
def main(url, write_to, target_name):
    """Downloads data from the web to a local filepath."""
    try:
        download(url, write_to, target_name=target_name)
    except FileNotFoundError:
        os.makedirs(write_to, exist_ok=True)
        download(url, write_to, target_name=target_name)



if __name__ == "__main__":
    main()
