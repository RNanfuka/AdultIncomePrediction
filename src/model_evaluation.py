from __future__ import annotations

from typing import Dict

import pandas as pd
from sklearn.model_selection import cross_validate


def evaluate_models(
    models: Dict[str, object],
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int,
    n_jobs: int,
) -> pd.DataFrame:
    """
    Run cross-validation for each estimator and summarize results.

    Parameters
    ----------
    models
        Mapping of model name to a scikit-learn compatible estimator.
    X
        Feature matrix.
    y
        Target vector.
    cv_folds
        Number of cross-validation folds (must be at least 2).
    n_jobs
        Number of parallel jobs to use in cross-validation.

    Returns
    -------
    pandas.DataFrame
        A dataframe indexed by model name with mean/std accuracy scores and timing metrics.

    Raises
    ------
    ValueError
        If no models are provided, cv_folds is less than 2, X and y lengths differ, or
        either X or y is empty.
    TypeError
        If any provided model does not implement a ``fit`` method.
    """
    if not models:
        raise ValueError("At least one model must be provided for evaluation.")
    if cv_folds < 2:
        raise ValueError("cv_folds must be at least 2.")
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of rows.")
    if len(X) == 0:
        raise ValueError("Training data is empty.")

    rows = []
    for name, model in models.items():
        if not hasattr(model, "fit"):
            raise TypeError(f"Model '{name}' must implement a fit method.")
        scores = cross_validate(
            model,
            X,
            y,
            cv=cv_folds,
            return_train_score=True,
            n_jobs=n_jobs,
        )
        rows.append(
            {
                "model": name,
                "train_accuracy_mean": scores["train_score"].mean(),
                "train_accuracy_std": scores["train_score"].std(),
                "test_accuracy_mean": scores["test_score"].mean(),
                "test_accuracy_std": scores["test_score"].std(),
                "fit_time_mean": scores["fit_time"].mean(),
                "score_time_mean": scores["score_time"].mean(),
            }
        )

    return (
        pd.DataFrame(rows)
        .set_index("model")
        .sort_values("test_accuracy_mean", ascending=False)
    )
