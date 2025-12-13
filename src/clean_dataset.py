
from __future__ import annotations

from typing import Dict, Iterable, List, Optional
import pandas as pd


def clean_dataset(
    dfs: List[pd.DataFrame],
    *,
    column_mappings: Optional[Dict[str, Dict]] = None,
    required_columns: Optional[Iterable[str]] = None,
    strip_strings: bool = True,
    missing_values: Iterable[str] = ("?",),
) -> pd.DataFrame:
    """
    Dataset cleaning utility.

    Parameters
    ----------
    dfs : list[pd.DataFrame]
        DataFrames to combine and clean.

    column_mappings : dict[str, dict], optional
        Mapping per column for value normalization.
        Example: {"education": {"HS-grad": "HighGrad"}}

    required_columns : iterable[str], optional
        Rows missing values in these columns will be dropped.

    strip_strings : bool, default=True
        Whether to strip whitespace from object columns.

    missing_values : iterable[str], default=("?",)
        Values to be treated as missing.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame.
    """
    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.dropna(how="all")

    if strip_strings:
        for column in combined.select_dtypes(include="object").columns:
            combined[column] = combined[column].str.strip()

    combined = combined.replace(list(missing_values), pd.NA)

    if required_columns:
        combined = combined.dropna(subset=required_columns)

    if column_mappings:
        for column, mapping in column_mappings.items():
            if column in combined.columns:
                combined[column] = combined[column].replace(mapping)

    return combined