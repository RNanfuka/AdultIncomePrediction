"""
Validate Adult Income correlations with pandera to catch collinearity issues.

Checks:
- No unusually high correlation between the encoded target (`income`) and any numeric feature.
- No unusually high correlation between pairs of numeric features.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import pandera.pandas as pa

TARGET_COLUMN = "income"
TARGET_MAPPING: Dict[str, int] = {"<=50K": 0, ">50K": 1, "<=50K.": 0, ">50K.": 1}
CORRELATION_LIMIT = 0.95


def _encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with the target encoded for correlation checks."""
    df = df.copy()
    df["_target_encoded"] = df[TARGET_COLUMN].map(TARGET_MAPPING)
    if df["_target_encoded"].isna().any():
        unexpected = df.loc[df["_target_encoded"].isna(), TARGET_COLUMN].unique()
        print(f"Unexpected target labels encountered: {unexpected}")
        raise ValueError("Found unmapped target labels.")
    return df


def check_target_correlations(df: pd.DataFrame) -> tuple[bool, dict]:
    """Ensure target-feature correlations are below the limit."""
    df = _encode_target(df)
    corr = df.select_dtypes(include="number").corr().abs()

    target_corr = corr["_target_encoded"].drop("_target_encoded")
    high = target_corr[target_corr > CORRELATION_LIMIT]

    flagged = high.to_dict()
    return high.empty, {
        "num_pairs": len(target_corr),
        "num_flagged": len(flagged),
        "flagged": flagged,
    }


def check_feature_correlations(df: pd.DataFrame) -> tuple[bool, dict]:
    """Ensure feature-feature correlations are below the limit."""
    df = _encode_target(df)
    corr = df.select_dtypes(include="number").corr().abs()

    feature_corr = corr.drop(index="_target_encoded", columns="_target_encoded")
    upper_triangle = feature_corr.where(np.triu(np.ones(feature_corr.shape), k=1).astype(bool))
    high_pairs = upper_triangle.stack()[lambda s: s > CORRELATION_LIMIT]

    flagged = {f"{left}~{right}": value for (left, right), value in high_pairs.items()}
    num_features = feature_corr.shape[0]
    num_pairs = int(num_features * (num_features - 1) / 2)
    return high_pairs.empty, {
        "num_pairs": num_pairs,
        "num_flagged": len(flagged),
        "flagged": flagged,
    }


def run_tests(df: pd.DataFrame) -> bool:
    """Run correlation checks, print a small report, and return True only if all pass."""
    checks = [
        ("target_correlations", check_target_correlations),
        ("feature_correlations", check_feature_correlations),
    ]

    results = []
    for name, func in checks:
        try:
            passed, meta = func(df)
        except ValueError:
            passed = False
            meta = {"num_pairs": 0, "num_flagged": 0, "flagged": {}}
        results.append(
            {
                "test": name,
                "pass_fail": "PASS" if passed else "FAIL",
                "num_pairs": meta["num_pairs"],
                "num_flagged": meta["num_flagged"],
                "flagged_pairs": meta["flagged"],
            }
        )

    report = pd.DataFrame(results, columns=["test", "pass_fail", "num_pairs", "num_flagged", "flagged_pairs"])
    print("**** Correlation validation ****")
    print(report.to_string(index=False))
    print(
        "Each row summarizes a check: num_pairs = total correlations examined, "
        "num_flagged = correlations over threshold, flagged_pairs = offenders."
    )

    return all(row["pass_fail"] == "PASS" for row in results)


correlation_schema = pa.DataFrameSchema(
    checks=[
        pa.Check(
            run_tests,
            error=(
                f"Correlations must be <= {CORRELATION_LIMIT} in absolute value "
                f"for both target-feature and feature-feature pairs."
            ),
        )
    ]
)


def validate_correlation_schema(file_path) -> None:
    df = pd.read_csv(file_path)
    correlation_schema.validate(df, lazy=True)
    print("Correlation checks passed.")
