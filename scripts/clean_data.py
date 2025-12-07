"""Clean and consolidate the Adult Income dataset."""

from __future__ import annotations

from pathlib import Path
from typing import List

import click
import pandas as pd

COLUMN_NAMES = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]

DEFAULT_INPUT = Path("data/raw/adult.csv")
DEFAULT_OUTPUT = Path("data/processed/clean_adult.csv")

education_mapping = {
    "Preschool": "dropout",
    "10th": "dropout",
    "11th": "dropout",
    "12th": "dropout",
    "1st-4th": "dropout",
    "5th-6th": "dropout",
    "7th-8th": "dropout",
    "9th": "dropout",
    "HS-Grad": "HighGrad",
    "HS-grad": "HighGrad",
    "Some-college": "CommunityCollege",
    "Assoc-acdm": "CommunityCollege",
    "Assoc-voc": "CommunityCollege",
    "Masters": "Masters",
    "Prof-school": "Masters",
}

marital_status_mapping = {
    "Never-married": "NotMarried",
    "Married-AF-spouse": "Married",
    "Married-civ-spouse": "Married",
    "Married-spouse-absent": "NotMarried",
    "Separated": "Separated",
    "Divorced": "Separated",
    "Widowed": "Widowed",
}


@click.command()
@click.option(
    "--input-path",
    type=click.Path(path_type=Path),
    default=str(DEFAULT_INPUT),
    show_default=True,
    help="Path to the Adult Income raw data file.",
)
@click.option(
    "--output-path",
    type=click.Path(path_type=Path),
    default=str(DEFAULT_OUTPUT),
    show_default=True,
    help="Destination for the cleaned CSV.",
)
def main(input_path: Path, output_path: Path) -> None:
    """Clean the raw Adult Income file and save the processed CSV."""
    dataframes = [load_raw_file(input_path)]

    cleaned = clean_dataset(dataframes)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False)
    click.echo(f"Wrote cleaned data to {output_path}")


def load_raw_file(path: Path) -> pd.DataFrame:
    """Load a raw Adult Income file; returns empty DataFrame if path is missing."""
    if path is None or not path.exists():
        return pd.DataFrame(columns=COLUMN_NAMES)

    return pd.read_csv(
        path,
        header=None,
        names=COLUMN_NAMES,
        sep=",",
        skipinitialspace=True,
        comment="|",
        engine="python",
    )


def clean_dataset(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """Combine, trim whitespace, and normalize the income labels."""
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.dropna(how="all")

    # Strip whitespace from string columns.
    for column in combined.select_dtypes(include="object").columns:
        combined[column] = combined[column].str.strip()

    combined = combined.replace("?", pd.NA)

    combined = combined[combined["income"].notna()]
    combined["income"] = combined["income"].str.replace("<=50K.", "<=50K", regex=False)
    combined["income"] = combined["income"].str.replace(">50K.", ">50K", regex=False)
    combined = combined[combined["income"] != ""]

    if "education" in combined.columns:
        combined["education"] = combined["education"].replace(education_mapping)
    if "marital-status" in combined.columns:
        combined["marital-status"] = combined["marital-status"].replace(
            marital_status_mapping
        )

    return combined


if __name__ == "__main__":
    main()
