import pandas as pd
import pandera.pandas as pa

# Bounds sourced from summary statistics of the canonical Adult Income dataset.
NUMERIC_BOUNDS = {
    "age": (17, 90),
    "fnlwgt": (12285, 1490400),
    "education-num": (1, 16),
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


def test_data_content(df) -> None:
    schema = pa.DataFrameSchema(
        {
            "age": pa.Column(
                int,
                checks=[
                    pa.Check.between(0, 150),
                    pa.Check(
                        lambda s: s.isna().mean() <= 0.05,
                        element_wise=False,
                        error="Too many null values in 'age' column."
                    )
                ]
            ),
            "workclass": pa.Column(
                str,
                checks=[
                    pa.Check.isin([
                        "?", "Federal-gov", "Local-gov", "Never-worked", "Private",
                        "Self-emp-inc", "Self-emp-not-inc", "State-gov", "Without-pay"
                    ]),
                    pa.Check(
                        lambda s: s.isna().mean() <= 0.05,
                        element_wise=False,
                        error="Too many null values in 'workclass' column."
                    )
                ],
                nullable=True
            ),
            "fnlwgt": pa.Column(
                int,
                checks=[
                    pa.Check.between(0, 2000000),
                    pa.Check(
                        lambda s: s.isna().mean() <= 0.05,
                        element_wise=False,
                        error="Too many null values in 'fnlwgt' column."
                    )
                ]
            ),
            "education": pa.Column(
                str,
                checks=[
                    pa.Check.isin([
                        "10th", "11th", "12th", "1st-4th", "5th-6th", "7th-8th", "9th",
                        "Assoc-acdm", "Assoc-voc", "Bachelors", "Doctorate", "HS-grad",
                        "Masters", "Preschool", "Prof-school", "Some-college"
                    ]),
                    pa.Check(
                        lambda s: s.isna().mean() <= 0.05,
                        element_wise=False,
                        error="Too many null values in 'education' column."
                    )
                ]
            ),
            "education-num": pa.Column(
                int,
                checks=[
                    pa.Check.between(0, 16),
                    pa.Check(
                        lambda s: s.isna().mean() <= 0.05,
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
                        lambda s: s.isna().mean() <= 0.05,
                        element_wise=False,
                        error="Too many null values in 'marital-status' column."
                    )
                ]
            ),
            "occupation": pa.Column(
                str,
                checks=[
                    pa.Check.isin([
                        "?", "Adm-clerical", "Armed-Forces", "Craft-repair", "Exec-managerial",
                        "Farming-fishing", "Handlers-cleaners", "Machine-op-inspct", "Other-service",
                        "Priv-house-serv", "Prof-specialty", "Protective-serv", "Sales",
                        "Tech-support", "Transport-moving"
                    ]),
                    pa.Check(
                        lambda s: s.isna().mean() <= 0.05,
                        element_wise=False,
                        error="Too many null values in 'occupation' column."
                    )
                ],
                nullable=True
            ),
            "relationship": pa.Column(
                str,
                checks=[
                    pa.Check.isin([
                        "Husband", "Not-in-family", "Other-relative",
                        "Own-child", "Unmarried", "Wife"
                    ]),
                    pa.Check(
                        lambda s: s.isna().mean() <= 0.05,
                        element_wise=False,
                        error="Too many null values in 'relationship' column."
                    )
                ]
            ),
            "race": pa.Column(
                str,
                checks=[
                    pa.Check.isin([
                        "Amer-Indian-Eskimo", "Asian-Pac-Islander",
                        "Black", "Other", "White"
                    ]),
                    pa.Check(
                        lambda s: s.isna().mean() <= 0.05,
                        element_wise=False,
                        error="Too many null values in 'race' column."
                    )
                ]
            ),
            "sex": pa.Column(
                str,
                checks=[
                    pa.Check.isin(["Female", "Male"]),
                    pa.Check(
                        lambda s: s.isna().mean() <= 0.05,
                        element_wise=False,
                        error="Too many null values in 'sex' column."
                    )
                ]
            ),
            "income": pa.Column(
                str,
                checks=[
                    pa.Check.isin(["<=50K", "<=50K.", ">50K", ">50K."]),
                    pa.Check(
                        lambda s: s.isna().mean() <= 0.05,
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
    print("Data content checks passed.")
