#!/usr/bin/env python3
"""
Data validation script for the Adult Income dataset.

Each check corresponds to the project's validation checklist and the script exits
with a non-zero code if any check fails.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from pandas.api import types as pdt

try:  # Optional dependency for schema validation
    import pandera as pa
    from pandera import Check
except ImportError:  # pragma: no cover - environment without pandera
    pa = None  # type: ignore[assignment]
    Check = None  # type: ignore[assignment]

EXPECTED_COLUMNS: List[str] = [
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

NUMERIC_COLUMNS: List[str] = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]

EXPECTED_DTYPES: Dict[str, str] = {
    "age": "numeric",
    "workclass": "string",
    "fnlwgt": "numeric",
    "education": "string",
    "education-num": "numeric",
    "marital-status": "string",
    "occupation": "string",
    "relationship": "string",
    "race": "string",
    "sex": "string",
    "capital-gain": "numeric",
    "capital-loss": "numeric",
    "hours-per-week": "numeric",
    "native-country": "string",
    "income": "string",
}

EXPECTED_CATEGORIES: Dict[str, Sequence[str]] = {
    "workclass": [
        "Federal-gov",
        "Local-gov",
        "Never-worked",
        "Private",
        "Self-emp-inc",
        "Self-emp-not-inc",
        "State-gov",
        "Without-pay",
    ],
    "education": [
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
    ],
    "marital-status": [
        "Divorced",
        "Married-AF-spouse",
        "Married-civ-spouse",
        "Married-spouse-absent",
        "Never-married",
        "Separated",
        "Widowed",
    ],
    "occupation": [
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
    ],
    "relationship": [
        "Husband",
        "Not-in-family",
        "Other-relative",
        "Own-child",
        "Unmarried",
        "Wife",
    ],
    "race": [
        "Amer-Indian-Eskimo",
        "Asian-Pac-Islander",
        "Black",
        "Other",
        "White",
    ],
    "sex": ["Female", "Male"],
    "native-country": [
        "Cambodia",
        "Canada",
        "China",
        "Columbia",
        "Cuba",
        "Dominican-Republic",
        "Ecuador",
        "El-Salvador",
        "England",
        "France",
        "Germany",
        "Greece",
        "Guatemala",
        "Haiti",
        "Holand-Netherlands",
        "Honduras",
        "Hong",
        "Hungary",
        "India",
        "Iran",
        "Ireland",
        "Italy",
        "Jamaica",
        "Japan",
        "Laos",
        "Mexico",
        "Nicaragua",
        "Outlying-US(Guam-USVI-etc)",
        "Peru",
        "Philippines",
        "Poland",
        "Portugal",
        "Puerto-Rico",
        "Scotland",
        "South",
        "Taiwan",
        "Thailand",
        "Trinadad&Tobago",
        "United-States",
        "Vietnam",
        "Yugoslavia",
    ],
    "income": ["<=50K", ">50K"],
}

# Minimum and maximum acceptable numeric values.
NUMERIC_RANGES: Dict[str, Tuple[float, float]] = {
    "age": (16, 90),
    "fnlwgt": (1, 2000000),
    "education-num": (1, 16),
    "capital-gain": (0, 100000),
    "capital-loss": (0, 5000),
    "hours-per-week": (1, 100),
}

VALUE_NORMALIZERS = {
    "income": {
        "<=50K.": "<=50K",
        ">50K.": ">50K",
    }
}
EXPECTED_TARGET_DISTRIBUTION = {"<=50K": 0.75, ">50K": 0.25}


@dataclass
class CheckResult:
    name: str
    passed: bool
    details: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Adult Income dataset validation checks."
    )
    default_data_path = Path("data") / "adult.csv"
    parser.add_argument(
        "data_path",
        type=Path,
        nargs="?",
        default=default_data_path,
        help=f"Path to the data file (default: {default_data_path}).",
    )
    parser.add_argument(
        "--allowed-formats",
        nargs="+",
        default=[".csv"],
        help="List of allowed file extensions (default: .csv).",
    )
    parser.add_argument(
        "--max-missing-ratio",
        type=float,
        default=0.1,
        help="Maximum allowed fraction of missing values per column.",
    )
    parser.add_argument(
        "--target-col",
        default="income",
        help="Target column used for distribution and correlation checks.",
    )
    parser.add_argument(
        "--target-tolerance",
        type=float,
        default=0.05,
        help="Allowed deviation from expected target distribution.",
    )
    parser.add_argument(
        "--target-corr-threshold",
        type=float,
        default=0.5,
        help="Absolute correlation threshold between target and features.",
    )
    parser.add_argument(
        "--feature-corr-threshold",
        type=float,
        default=0.95,
        help="Absolute correlation threshold between numeric features.",
    )
    parser.add_argument(
        "--na-values",
        nargs="+",
        default=["?", " ?"],
        help="Tokens to treat as missing when loading data.",
    )
    return parser.parse_args()


def normalize_allowed_formats(formats: Iterable[str]) -> List[str]:
    normalized: List[str] = []
    for fmt in formats:
        fmt = fmt.strip()
        if not fmt:
            continue
        if not fmt.startswith("."):
            fmt = f".{fmt}"
        normalized.append(fmt.lower())
    return normalized


def load_dataframe(path: Path, na_values: Sequence[str]) -> pd.DataFrame:
    if path.suffix.lower() != ".csv":
        raise ValueError(f"Unsupported file format: {path.suffix}")
    df = pd.read_csv(path, na_values=na_values)
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
        df[col] = df[col].replace("", np.nan)
    for col, replacements in VALUE_NORMALIZERS.items():
        if col in df.columns:
            df[col] = df[col].replace(replacements)
    return df


def check_file_format(path: Path, allowed_formats: Iterable[str]) -> CheckResult:
    allowed = set(normalize_allowed_formats(allowed_formats))
    suffix = path.suffix.lower()
    passed = suffix in allowed
    details = (
        f"Found {suffix or '<no extension>'}; allowed formats: {', '.join(sorted(allowed))}"
    )
    return CheckResult("Correct data file format", passed, details)


def check_column_names(df: pd.DataFrame) -> CheckResult:
    missing = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    extras = [col for col in df.columns if col not in EXPECTED_COLUMNS]
    passed = not missing and not extras
    if passed:
        details = "All expected columns present."
    else:
        detail_parts = []
        if missing:
            detail_parts.append(f"Missing: {', '.join(missing)}")
        if extras:
            detail_parts.append(f"Unexpected: {', '.join(extras)}")
        details = "; ".join(detail_parts)
    return CheckResult("Correct column names", passed, details)


def build_pandera_schema() -> "pa.DataFrameSchema":
    if pa is None:
        raise RuntimeError("pandera is not available")
    columns: Dict[str, pa.Column] = {}
    for col in EXPECTED_COLUMNS:
        checks: List[Check] = []
        if col in NUMERIC_RANGES:
            lower, upper = NUMERIC_RANGES[col]
            checks.append(Check.in_range(lower, upper))
        if col in EXPECTED_CATEGORIES:
            checks.append(Check.isin(EXPECTED_CATEGORIES[col]))
        if col in NUMERIC_COLUMNS:
            dtype = pa.Float64
        else:
            dtype = pa.String
        columns[col] = pa.Column(
            dtype,
            checks=checks or None,
            nullable=True,
            coerce=True,
        )
    return pa.DataFrameSchema(columns, coerce=True)


def check_pandera_schema(df: pd.DataFrame) -> CheckResult:
    if pa is None or Check is None:
        return CheckResult(
            "Pandera schema validation",
            True,
            "pandera not installed; run `pip install pandera` to enable this check.",
        )
    schema = build_pandera_schema()
    try:
        schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as err:
        failure_cases = err.failure_cases
        sample = failure_cases.head(5).to_dict("records")
        details = (
            f"{len(failure_cases)} schema violations detected via pandera; "
            f"sample: {sample}"
        )
        return CheckResult("Pandera schema validation", False, details)
    return CheckResult("Pandera schema validation", True, "pandera schema checks passed.")


def check_empty_rows(df: pd.DataFrame) -> CheckResult:
    empty_rows = df.isna().all(axis=1).sum()
    passed = empty_rows == 0
    details = f"{empty_rows} completely empty rows found."
    return CheckResult("No empty observations", passed, details)


def check_missingness(df: pd.DataFrame, threshold: float) -> CheckResult:
    ratios = df.isna().mean()
    failing = {col: ratio for col, ratio in ratios.items() if ratio > threshold}
    passed = not failing
    if passed:
        max_ratio = ratios.max()
        details = f"Max missingness {max_ratio:.2%} within threshold {threshold:.2%}."
    else:
        formatted = ", ".join(f"{col}: {ratio:.2%}" for col, ratio in sorted(failing.items()))
        details = f"Columns above {threshold:.2%}: {formatted}"
    return CheckResult("Missingness not beyond expected threshold", passed, details)


def check_dtypes(df: pd.DataFrame) -> CheckResult:
    mismatches: List[str] = []
    for col, expected in EXPECTED_DTYPES.items():
        if col not in df.columns:
            mismatches.append(f"{col}: column missing")
            continue
        series = df[col]
        if expected == "numeric":
            valid = pdt.is_numeric_dtype(series)
        elif expected == "string":
            valid = pdt.is_object_dtype(series) or isinstance(
                series.dtype, pd.CategoricalDtype
            )
        else:
            valid = False
        if not valid:
            mismatches.append(f"{col}: found {series.dtype}, expected {expected}")
    passed = not mismatches
    details = "; ".join(mismatches) if mismatches else "All column dtypes look correct."
    return CheckResult("Correct data types in each column", passed, details)


def check_duplicates(df: pd.DataFrame) -> CheckResult:
    dup_count = df.duplicated().sum()
    passed = dup_count == 0
    details = f"{dup_count} duplicate rows found."
    return CheckResult("No duplicate observations", passed, details)


def check_value_ranges(df: pd.DataFrame) -> CheckResult:
    out_of_bounds: List[str] = []
    for col, (lower, upper) in NUMERIC_RANGES.items():
        if col not in df.columns:
            out_of_bounds.append(f"{col}: column missing")
            continue
        series = df[col].dropna()
        if series.empty:
            continue
        min_val, max_val = series.min(), series.max()
        if min_val < lower or max_val > upper:
            out_of_bounds.append(
                f"{col}: observed [{min_val}, {max_val}] outside [{lower}, {upper}]"
            )
    passed = not out_of_bounds
    details = "; ".join(out_of_bounds) if out_of_bounds else "All numeric columns within expected ranges."
    return CheckResult("No outlier or anomalous values", passed, details)


def check_category_levels(df: pd.DataFrame) -> CheckResult:
    issues: List[str] = []
    for col, expected_values in EXPECTED_CATEGORIES.items():
        if col not in df.columns:
            issues.append(f"{col}: column missing")
            continue
        series = df[col].dropna()
        observed = set(series.unique())
        unexpected = observed.difference(expected_values)
        if unexpected:
            issues.append(f"{col}: unexpected values {', '.join(sorted(unexpected))}")
        if len(observed) <= 1:
            issues.append(f"{col}: only {len(observed)} distinct value(s) present")
    passed = not issues
    details = "; ".join(issues) if issues else "All categorical levels match the specification."
    return CheckResult("Correct category levels", passed, details)


def check_target_distribution(
    df: pd.DataFrame,
    target_col: str,
    expected_distribution: Dict[str, float],
    tolerance: float,
) -> CheckResult:
    if target_col not in df.columns:
        return CheckResult(
            "Target variable follows expected distribution",
            False,
            f"{target_col} column missing.",
        )
    series = df[target_col].dropna()
    observed = series.value_counts(normalize=True)
    deviations: Dict[str, float] = {}
    for value, expected_ratio in expected_distribution.items():
        observed_ratio = observed.get(value, 0.0)
        deviations[value] = abs(observed_ratio - expected_ratio)
    failing = {value: delta for value, delta in deviations.items() if delta > tolerance}
    passed = not failing
    if passed:
        details = ", ".join(
            f"{value}: {observed.get(value, 0.0):.2%}" for value in expected_distribution
        )
    else:
        details = "; ".join(
            f"{value}: deviation {delta:.2%} exceeds tolerance {tolerance:.2%}"
            for value, delta in failing.items()
        )
    return CheckResult("Target variable follows expected distribution", passed, details)


def _prepare_numeric(df: pd.DataFrame, columns: Iterable[str]) -> List[str]:
    available = [col for col in columns if col in df.columns]
    return [col for col in available if pdt.is_numeric_dtype(df[col])]


def check_target_correlations(
    df: pd.DataFrame,
    target_col: str,
    numeric_columns: Iterable[str],
    threshold: float,
) -> CheckResult:
    if target_col not in df.columns:
        return CheckResult(
            "No anomalous correlations between target and features",
            False,
            f"{target_col} column missing.",
        )
    numeric_cols = _prepare_numeric(df, numeric_columns)
    if not numeric_cols:
        return CheckResult(
            "No anomalous correlations between target and features",
            True,
            "No numeric columns available for correlation check.",
        )
    target_series = df[target_col].astype("category")
    target_codes = target_series.cat.codes.astype(float)
    target_codes[target_codes == -1] = np.nan
    issues: List[str] = []
    for col in numeric_cols:
        aligned = pd.concat([target_codes, df[col]], axis=1, join="inner").dropna()
        if aligned.empty:
            continue
        corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
        if pd.isna(corr):
            continue
        if abs(corr) > threshold:
            issues.append(f"{col}: |corr|={abs(corr):.2f}")
    passed = not issues
    details = "; ".join(issues) if issues else f"All |corr| <= {threshold:.2f}"
    return CheckResult(
        "No anomalous correlations between target and features",
        passed,
        details,
    )


def check_feature_correlations(
    df: pd.DataFrame,
    numeric_columns: Iterable[str],
    threshold: float,
) -> CheckResult:
    numeric_cols = _prepare_numeric(df, numeric_columns)
    if len(numeric_cols) < 2:
        return CheckResult(
            "No anomalous correlations between features",
            True,
            "Insufficient numeric columns for pairwise correlation.",
        )
    corr_matrix = df[numeric_cols].corr().abs()
    issues: List[str] = []
    for idx, col in enumerate(numeric_cols):
        for other in numeric_cols[idx + 1 :]:
            corr_value = corr_matrix.loc[col, other]
            if pd.notna(corr_value) and corr_value > threshold:
                issues.append(f"{col} vs {other}: |corr|={corr_value:.2f}")
    passed = not issues
    details = "; ".join(issues) if issues else f"No pair exceeds |corr| of {threshold:.2f}"
    return CheckResult(
        "No anomalous correlations between features",
        passed,
        details,
    )


def run_checks(df: pd.DataFrame, args: argparse.Namespace, data_path: Path) -> List[CheckResult]:
    return [
        check_file_format(data_path, args.allowed_formats),
        check_column_names(df),
        check_pandera_schema(df),
        check_empty_rows(df),
        check_missingness(df, args.max_missing_ratio),
        check_dtypes(df),
        check_duplicates(df),
        check_value_ranges(df),
        check_category_levels(df),
        check_target_distribution(
            df,
            args.target_col,
            EXPECTED_TARGET_DISTRIBUTION,
            args.target_tolerance,
        ),
        check_target_correlations(
            df,
            args.target_col,
            NUMERIC_COLUMNS,
            args.target_corr_threshold,
        ),
        check_feature_correlations(
            df,
            NUMERIC_COLUMNS,
            args.feature_corr_threshold,
        ),
    ]


def render_report(results: Sequence[CheckResult]) -> None:
    name_width = max(len(result.name) for result in results)
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"{result.name.ljust(name_width)} | {status} | {result.details}")


def main() -> int:
    args = parse_args()
    try:
        df = load_dataframe(args.data_path, args.na_values)
    except Exception as exc:  # pragma: no cover - defensive guard for CLI usage
        print(f"Failed to load data: {exc}", file=sys.stderr)
        return 2
    results = run_checks(df, args, args.data_path)
    render_report(results)
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
