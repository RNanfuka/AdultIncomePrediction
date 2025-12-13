import pandas as pd
from src.clean_dataset import clean_dataset


def test_generic_string_cleaning_and_missing_values():
    df = pd.DataFrame(
        {
            "name": [" Alice ", "Bob", "?"],
            "age": [30, 40, 50],
        }
    )

    result = clean_dataset([df], required_columns=["name"])

    assert len(result) == 2
    assert result.loc[0, "name"] == "Alice"


def test_column_mapping_is_optional_and_generic():
    df = pd.DataFrame(
        {
            "status": ["A", "B", "C"],
        }
    )

    mapping = {"status": {"A": "Active", "B": "Blocked"}}

    result = clean_dataset([df], column_mappings=mapping)

    assert list(result["status"]) == ["Active", "Blocked", "C"]


def test_required_columns_drop_rows():
    df = pd.DataFrame(
        {
            "x": [1, 2, None],
            "y": [None, 3, 4],
        }
    )

    result = clean_dataset([df], required_columns=["x", "y"])

    assert len(result) == 1
    assert result.iloc[0]["x"] == 2


def test_multiple_dataframes_supported():
    df1 = pd.DataFrame({"a": [1]})
    df2 = pd.DataFrame({"a": [2]})

    result = clean_dataset([df1, df2])

    assert list(result["a"]) == [1, 2]