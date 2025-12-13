import os
import sys

import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.model_evaluation import evaluate_models


def _load_sample_data():
    sample_path = Path(__file__).with_name("test_preprocessed_sample.csv")
    df = pd.read_csv(sample_path)
    X = df.drop(columns=["income"])
    y = df["income"]
    return X, y


def _load_empty_data():
    empty_path = Path(__file__).with_name("test_preprocessed_empty.csv")
    df = pd.read_csv(empty_path)
    X = df.drop(columns=["income"])
    y = df["income"]
    return X, y


def test_evaluate_models_returns_sorted_summary():
    X, y = _load_sample_data()
    models = {
        "Dummy": DummyClassifier(strategy="most_frequent"),
        "LogReg": LogisticRegression(max_iter=200, random_state=0),
    }

    summary = evaluate_models(models, X, y, cv_folds=3, n_jobs=1)

    expected_columns = {
        "train_accuracy_mean",
        "train_accuracy_std",
        "test_accuracy_mean",
        "test_accuracy_std",
        "fit_time_mean",
        "score_time_mean",
    }
    assert set(summary.columns) == expected_columns
    assert summary.index[0] == "LogReg"
    assert summary.loc["LogReg", "test_accuracy_mean"] >= summary.loc["Dummy", "test_accuracy_mean"]
    assert not summary.isnull().any().any()


def test_evaluate_models_requires_models():
    X, y = _load_sample_data()
    with pytest.raises(ValueError):
        evaluate_models({}, X, y, cv_folds=3, n_jobs=1)


def test_evaluate_models_checks_length_mismatch():
    X, y = _load_sample_data()
    with pytest.raises(ValueError):
        evaluate_models({"Dummy": DummyClassifier()}, X.iloc[:1], y, cv_folds=3, n_jobs=1)


def test_evaluate_models_requires_min_cv_folds():
    X, y = _load_sample_data()
    with pytest.raises(ValueError):
        evaluate_models({"Dummy": DummyClassifier()}, X, y, cv_folds=1, n_jobs=1)


def test_evaluate_models_rejects_non_estimators():
    X, y = _load_sample_data()
    with pytest.raises(TypeError):
        evaluate_models({"NotAModel": object()}, X, y, cv_folds=3, n_jobs=1)


def test_evaluate_models_rejects_empty_data():
    X_empty, y_empty = _load_empty_data()
    with pytest.raises(ValueError):
        evaluate_models({"Dummy": DummyClassifier()}, X_empty, y_empty, cv_folds=3, n_jobs=1)
