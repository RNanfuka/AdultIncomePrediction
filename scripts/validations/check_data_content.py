import pandas as pd
import pandera.pandas as pa

# Bounds sourced from summary statistics of the canonical Adult Income dataset.
NUMERIC_BOUNDS = {
    "age": (17, 90),
    "fnlwgt": (12285, 1490400),
    "education-num": (1, 16),
}

REQUIRED_COLUMNS = [
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
    "income",
]

NULL_RATE_LIMITS = {
    "age": 0.10,
    "workclass": 0.10,
    "fnlwgt": 0.10,
    "education": 0.10,
    "education-num": 0.10,
    "marital-status": 0.10,
    "occupation": 0.10,
    "relationship": 0.10,
    "race": 0.10,
    "sex": 0.10,
    "income": 0.10,
}

# Enumerations ensure we only keep the well-known category spelling variants.
CATEGORY_LEVELS = {
    "workclass": {
        "?",
        "Federal-gov",
        "Local-gov",
        "Never-worked",
        "Private",
        "Self-emp-inc",
        "Self-emp-not-inc",
        "State-gov",
        "Without-pay",
    },
    "education": {
        "10th",
        "11th",
        "12th",
        "1st-4th",
        "5th-6th",
        "7th-8th",
        "9th",
        "Assoc-acdm",
        "Assoc-voc",
        "Bachelors",
        "Doctorate",
        "HS-grad",
        "Masters",
        "Preschool",
        "Prof-school",
        "Some-college",
    },
    "marital-status": {
        "Divorced",
        "Married-AF-spouse",
        "Married-civ-spouse",
        "Married-spouse-absent",
        "Never-married",
        "Separated",
        "Widowed",
    },
    "occupation": {
        "?",
        "Adm-clerical",
        "Armed-Forces",
        "Craft-repair",
        "Exec-managerial",
        "Farming-fishing",
        "Handlers-cleaners",
        "Machine-op-inspct",
        "Other-service",
        "Priv-house-serv",
        "Prof-specialty",
        "Protective-serv",
        "Sales",
        "Tech-support",
        "Transport-moving",
    },
    "relationship": {
        "Husband",
        "Not-in-family",
        "Other-relative",
        "Own-child",
        "Unmarried",
        "Wife",
    },
    "race": {
        "Amer-Indian-Eskimo",
        "Asian-Pac-Islander",
        "Black",
        "Other",
        "White",
    },
    "sex": {
        "Female",
        "Male",
    },
}

TARGET_LABEL_GROUPS = {
    "<=50K": {"<=50K", "<=50K."},
    ">50K": {">50K", ">50K."},
}
EXPECTED_TARGET_DISTRIBUTION = {"<=50K": 0.76, ">50K": 0.24}
TARGET_DISTRIBUTION_TOLERANCE = 0.10


def _check_no_outliers(df: pd.DataFrame) -> bool:
    """Ensure numeric values stay within historically observed Adult Income bounds."""
    for column, (lower, upper) in NUMERIC_BOUNDS.items():
        if column not in df.columns:
            return False
        series = df[column].dropna()
        if series.empty:
            return False
        if ((series < lower) | (series > upper)).any():
            return False
    return True


def _check_null_rates(df: pd.DataFrame) -> bool:
    """Confirm columns stay within allowed null-rate limits."""
    for column, limit in NULL_RATE_LIMITS.items():
        if column not in df.columns:
            return False
        if df[column].isna().mean() > limit:
            return False
    return True


def _check_category_levels(df: pd.DataFrame) -> bool:
    """Confirm categorical values use the expected spellings and retain multiple levels."""
    for column, allowed in CATEGORY_LEVELS.items():
        if column not in df.columns:
            return False
        series = df[column].dropna()
        unique_values = set(series.unique())
        if not unique_values:
            return False
        if len(unique_values) == 1:
            return False
        unexpected = unique_values.difference(allowed)
        if unexpected:
            return False
    return True


def _check_target_distribution(df: pd.DataFrame) -> bool:
    """Verify the income target keeps the expected class balance within tolerance."""
    if "income" not in df.columns:
        return False
    counts = df["income"].value_counts(normalize=True)
    for group, labels in TARGET_LABEL_GROUPS.items():
        observed = counts[counts.index.isin(labels)].sum()
        if observed == 0:
            return False
        expected = EXPECTED_TARGET_DISTRIBUTION[group]
        if abs(observed - expected) > TARGET_DISTRIBUTION_TOLERANCE:
            return False
    return True


def test_required_columns(df: pd.DataFrame, raise_on_fail: bool = True) -> bool:
    """Return True if required columns are present; optionally raise on failure."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing and raise_on_fail:
        raise ValueError(f"Missing required columns: {missing}")
    return not missing


def test_no_empty_rows(df: pd.DataFrame, raise_on_fail: bool = True) -> bool:
    """Return True if no rows are entirely empty; optionally raise on failure."""
    has_empty = (df.isna().all(axis=1)).any()
    if has_empty and raise_on_fail:
        raise ValueError("Empty rows found.")
    return not has_empty


def test_no_outliers(df: pd.DataFrame, raise_on_fail: bool = True) -> bool:
    """Return True if numeric values are within bounds; optionally raise on failure."""
    ok = _check_no_outliers(df)
    if not ok and raise_on_fail:
        raise ValueError("Found numeric values outside expected bounds.")
    return ok


def test_null_rates(df: pd.DataFrame, raise_on_fail: bool = True) -> bool:
    """Return True if columns meet null-rate limits; optionally raise on failure."""
    ok = _check_null_rates(df)
    if not ok and raise_on_fail:
        raise ValueError("Columns exceed allowed null-rate limits.")
    return ok


def test_category_levels(df: pd.DataFrame, raise_on_fail: bool = True) -> bool:
    """Return True if categories have expected values and multiple levels; optionally raise on failure."""
    ok = _check_category_levels(df)
    if not ok and raise_on_fail:
        raise ValueError("Unexpected category labels or single-level categorical columns detected.")
    return ok


def test_target_distribution(df: pd.DataFrame, raise_on_fail: bool = True) -> bool:
    """Return True if target class balance is within tolerance; optionally raise on failure."""
    ok = _check_target_distribution(df)
    if not ok and raise_on_fail:
        raise ValueError("Income distribution drifts beyond allowed tolerance.")
    return ok


def test_no_duplicate_rows(df: pd.DataFrame, raise_on_fail: bool = True) -> bool:
    """Return True if no duplicate rows exist; optionally raise on failure."""
    has_dupes = df.duplicated().any()
    if has_dupes and raise_on_fail:
        raise ValueError("Duplicate rows found.")
    return not has_dupes


def test_data_content(df) -> None:
    schema = pa.DataFrameSchema(
        {
            "age": pa.Column(
                int,
                checks=[
                    pa.Check(
                        lambda s: s.isna().mean() <= 0.1,
                        element_wise=False,
                        error="Too many null values in 'age' column."
                    )
                ]
            ),
            "workclass": pa.Column(
                str,
                checks=[
                    pa.Check(
                        lambda s: s.isna().mean() <= 0.1,
                        element_wise=False,
                        error="Too many null values in 'workclass' column."
                    )
                ],
                nullable=True
            ),
            "fnlwgt": pa.Column(
                int,
                checks=[
                    pa.Check(
                        lambda s: s.isna().mean() <= 0.1,
                        element_wise=False,
                        error="Too many null values in 'fnlwgt' column."
                    )
                ]
            ),
            "education": pa.Column(
                str,
                checks=[
                    pa.Check(
                        lambda s: s.isna().mean() <= 0.1,
                        element_wise=False,
                        error="Too many null values in 'education' column."
                    )
                ]
            ),
            "education-num": pa.Column(
                int,
                checks=[
                    pa.Check(
                        lambda s: s.isna().mean() <= 0.1,
                        element_wise=False,
                        error="Too many null values in 'education-num' column."
                    )
                ]
            ),
            "marital-status": pa.Column(
                str,
                checks=[
                    pa.Check.isin([
                        "Divorced", "Married-AF-spouse", "Married-civ-spouse",
                        "Married-spouse-absent", "Never-married", "Separated", "Widowed"
                    ]),
                    pa.Check(
                        lambda s: s.isna().mean() <= 0.1,
                        element_wise=False,
                        error="Too many null values in 'marital-status' column."
                    )
                ]
            ),
            "occupation": pa.Column(
                str,
                checks=[
                    pa.Check(
                        lambda s: s.isna().mean() <= 0.1,
                        element_wise=False,
                        error="Too many null values in 'occupation' column."
                    )
                ],
                nullable=True
            ),
            "relationship": pa.Column(
                str,
                checks=[
                    pa.Check(
                        lambda s: s.isna().mean() <= 0.1,
                        element_wise=False,
                        error="Too many null values in 'relationship' column."
                    )
                ]
            ),
            "race": pa.Column(
                str,
                checks=[
                    pa.Check(
                        lambda s: s.isna().mean() <= 0.10,
                        element_wise=False,
                        error="Too many null values in 'race' column."
                    )
                ]
            ),
            "sex": pa.Column(
                str,
                checks=[
                    pa.Check(
                        lambda s: s.isna().mean() <= 0.10,
                        element_wise=False,
                        error="Too many null values in 'sex' column."
                    )
                ]
            ),
            "income": pa.Column(
                str,
                checks=[
                    pa.Check(
                        lambda s: s.isna().mean() <= 0.10,
                        element_wise=False,
                        error="Too many null values in 'income' column."
                    )
                ],
                nullable=False
            )
        },
        checks=[
            # Code adopted from https://ubc-dsci.github.io/reproducible-and-trustworthy-workflows-for-data-science/lectures/135-data_validation-python-pandera.html
            pa.Check(lambda df: ~(df.isna().all(axis=1)).any(), error="Empty rows found."),
            pa.Check(_check_no_outliers, error="Found numeric values outside expected bounds."),
            pa.Check(
                _check_category_levels,
                error="Unexpected category labels or single-level categorical columns detected.",
            ),
            pa.Check(
                _check_target_distribution,
                error="Income distribution drifts beyond allowed tolerance.",
            ),
            pa.Check(lambda df: ~df.duplicated().any(), error="Duplicate rows found.")

        ],
    )

    schema.validate(df)


def get_tests(df, raise_on_fail: bool = True):
    """Return a list of (name, callable) tests for data content validation."""
    return [
        ("required columns present", lambda: test_required_columns(df, raise_on_fail)),
        ("no empty rows found", lambda: test_no_empty_rows(df, raise_on_fail)),
        ("columns respect null limits", lambda: test_null_rates(df, raise_on_fail)),
        ("numeric values within bounds", lambda: test_no_outliers(df, raise_on_fail)),
        ("valid category levels", lambda: test_category_levels(df, raise_on_fail)),
        ("target distribution within tolerance", lambda: test_target_distribution(df, raise_on_fail)),
        ("no duplicate rows found", lambda: test_no_duplicate_rows(df, raise_on_fail)),
    ]
