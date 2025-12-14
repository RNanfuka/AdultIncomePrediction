"""
Train income prediction models using preprocessed training features.

This script expects a CSV with the engineered training features and the income
label (from the preprocessing script) and writes cross-validation summaries,
hyperparameter search results, and simple coefficient plots to disk.

Outputs:
- Tables (CSVs) under `<artifacts_dir>/tables/`
- Figures (PNGs) under `<artifacts_dir>/figures/`
- Tuned models (pickles) under `<artifacts_dir>/models/`

Inputs:
- `--data-dir`: base data folder containing `processed/preprocessed_train.csv`
- `--artifacts-dir`: base folder where artifacts will be written
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

# Limit joblib worker discovery in constrained environments
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import click
import matplotlib.pyplot as plt
import joblib
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import loguniform
from src.model_evaluation import evaluate_models


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_training_data(train_path: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(train_path)
    if df.empty:
        raise ValueError(f"No rows found in {train_path}.")
    if "income" not in df.columns:
        raise ValueError("Preprocessed training data must include an 'income' column.")
    y = df["income"]
    X = df.drop(columns=["income"])
    return X, y


def build_models() -> Dict[str, object]:
    return {
        "Dummy-most_frequent": DummyClassifier(strategy="most_frequent"),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "KNN": KNeighborsClassifier(),
        "SVM-RBF": SVC(kernel="rbf", probability=False, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=500, random_state=42),
        "GaussianNB": GaussianNB(),
    }


def run_hyperparameter_search(
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int,
    n_jobs: int,
    log_reg_iter: int,
    svm_iter: int,
) -> tuple[pd.DataFrame, LogisticRegression, SVC]:
    log_reg_search = RandomizedSearchCV(
        LogisticRegression(max_iter=500, random_state=42),
        param_distributions={
            "C": loguniform(1e-3, 1e3),
            "penalty": ["l2"],
            "solver": ["lbfgs"],
        },
        n_iter=log_reg_iter,
        cv=cv_folds,
        scoring="accuracy",
        random_state=42,
        n_jobs=n_jobs,
    )

    svm_search = RandomizedSearchCV(
        SVC(kernel="rbf", random_state=42),
        param_distributions={
            "C": loguniform(1e-2, 1e2),
            "gamma": loguniform(1e-4, 1e0),
        },
        n_iter=svm_iter,
        cv=cv_folds,
        scoring="accuracy",
        random_state=42,
        n_jobs=n_jobs,
    )

    log_reg_search.fit(X, y)
    svm_search.fit(X, y)

    hpo_rows = [
        {
            "model": "LogisticRegression_tuned",
            "best_params": log_reg_search.best_params_,
            "best_score": log_reg_search.best_score_,
        },
        {
            "model": "SVM_RBF_tuned",
            "best_params": svm_search.best_params_,
            "best_score": svm_search.best_score_,
        },
    ]
    hpo_df = pd.DataFrame(hpo_rows).set_index("model")

    return hpo_df, log_reg_search.best_estimator_, svm_search.best_estimator_


def save_cv_chart_png(cv_summary: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = cv_summary.sort_values("test_accuracy_mean")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(
        ordered.index,
        ordered["test_accuracy_mean"],
        xerr=ordered["test_accuracy_std"],
        color="steelblue",
    )
    ax.set_xlabel("Cross-validated accuracy")
    ax.set_title("Model comparison (CV mean ± std)")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def save_coef_chart_png(coef_df: pd.DataFrame, top_n: int, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    top_positive = coef_df.nlargest(top_n, "coefficient")
    top_negative = coef_df.nsmallest(top_n, "coefficient")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=False)
    axes[0].barh(top_positive["feature"], top_positive["coefficient"], color="seagreen")
    axes[0].set_title(f"Top +{top_n} coefficients")
    axes[0].set_xlabel("Log-odds")

    axes[1].barh(top_negative["feature"], top_negative["coefficient"], color="firebrick")
    axes[1].set_title(f"Top -{top_n} coefficients")
    axes[1].set_xlabel("Log-odds")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def save_table_image(df: pd.DataFrame, output_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 0.5 + 0.4 * len(df)))
    ax.axis("off")
    table = ax.table(
        cellText=df.round(3).values,
        colLabels=df.columns,
        rowLabels=df.index if df.index.name is not None else None,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    fig.suptitle(title, fontsize=12, y=0.98)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)


@click.command()
@click.option(
    "--data-dir",
    default="data",
    show_default=True,
    type=click.Path(path_type=Path),
    help="Base data directory containing processed/preprocessed_train.csv",
)
@click.option(
    "--artifacts-dir",
    default="artifacts",
    show_default=True,
    type=click.Path(path_type=Path),
    help="Base directory where tables, figures, and models will be written.",
)
@click.option("--cv-folds", default=5, show_default=True, type=int, help="Cross-validation folds.")
@click.option("--top-n", default=10, show_default=True, type=int, help="Top coefficients to plot in each direction.")
@click.option(
    "--n-jobs",
    default=1,
    show_default=True,
    type=int,
    help="Parallel jobs for cross-validation (set >1 only if your environment allows).",
)
@click.option(
    "--log-reg-iter",
    default=1,
    show_default=True,
    type=int,
    help="Randomized search iterations for Logistic Regression.",
)
@click.option(
    "--svm-iter",
    default=30,
    show_default=True,
    type=int,
    help="Randomized search iterations for RBF SVM.",
)
@click.option(
    "--skip-hpo/--run-hpo",
    default=False,
    show_default=True,
    help="Skip hyperparameter optimization (defaults to running it).",
)
@click.option(
    "--save-pickles/--no-save-pickles",
    default=True,
    show_default=True,
    help="Save tuned models to pickle files.",
)
def main(
    data_dir: Path,
    artifacts_dir: Path,
    cv_folds: int,
    top_n: int,
    n_jobs: int,
    log_reg_iter: int,
    svm_iter: int,
    skip_hpo: bool,
    save_pickles: bool,
) -> None:
    data_dir = data_dir.resolve()
    artifacts_dir = ensure_dir(artifacts_dir)
    tables_dir = ensure_dir(artifacts_dir / "tables")
    figures_dir = ensure_dir(artifacts_dir / "figures")
    models_dir = ensure_dir(artifacts_dir / "models")

    train_path = data_dir / "processed" / "preprocessed_train.csv"
    X_train, y_train = load_training_data(str(train_path))

    models = build_models()
    cv_summary = evaluate_models(models, X_train, y_train, cv_folds, n_jobs)

    cv_table_path = tables_dir / "cv_summary.csv"
    coef_table_path = tables_dir / "log_reg_coefficients.csv"
    hpo_table_path = tables_dir / "hpo_results.csv"
    cv_fig_path = figures_dir / "cv_summary.png"
    coef_fig_path = figures_dir / "log_reg_coefficients.png"
    cv_table_img_path = figures_dir / "cv_summary_table.png"
    coef_table_img_path = figures_dir / "log_reg_coefficients_table.png"
    hpo_table_img_path = figures_dir / "hpo_results_table.png"

    cv_summary.to_csv(cv_table_path)
    save_cv_chart_png(cv_summary, cv_fig_path)
    save_table_image(cv_summary.reset_index(), cv_table_img_path, "CV summary table")

    tuned_log_reg = tuned_svm = None
    if not skip_hpo:
        hpo_df, tuned_log_reg, tuned_svm = run_hyperparameter_search(
            X_train, y_train, cv_folds, n_jobs, log_reg_iter, svm_iter
        )
        hpo_df.to_csv(hpo_table_path)
        save_table_image(hpo_df.reset_index(), hpo_table_img_path, "Hyperparameter search results")
        if save_pickles:
            joblib.dump(tuned_log_reg, models_dir / "log_reg_tuned.pkl")
            joblib.dump(tuned_svm, models_dir / "svm_rbf_tuned.pkl")
    else:
        log_reg_pickle = models_dir / "log_reg_tuned.pkl"
        if not log_reg_pickle.exists():
            raise FileNotFoundError(
                f"No tuned Logistic Regression pickle found in {models_dir}. "
                "Run with hyperparameter search enabled to generate it."
            )
        tuned_log_reg = joblib.load(log_reg_pickle)

    # Fit logistic regression (tuned if available) on full training set for interpretability artifacts.
    log_reg_model = tuned_log_reg or models["LogisticRegression"].fit(X_train, y_train)
    coef_df = pd.DataFrame(
        {
            "feature": X_train.columns,
            "coefficient": log_reg_model.coef_[0],
            "odds_ratio": np.exp(log_reg_model.coef_[0]),
        }
    )
    save_coef_chart_png(coef_df, top_n, coef_fig_path)
    save_table_image(coef_df, coef_table_img_path, "Logistic regression coefficients")
    coef_df.to_csv(coef_table_path, index=False)

    click.echo(f"Wrote CV summary table to {cv_table_path}")
    click.echo(f"Wrote CV summary figure to {cv_fig_path}")
    click.echo(f"Wrote CV summary table image to {cv_table_img_path}")
    if not skip_hpo:
        click.echo(f"Wrote hyperparameter search table to {hpo_table_path}")
        click.echo(f"Wrote hyperparameter search table image to {hpo_table_img_path}")
        if save_pickles:
            click.echo(f"Saved tuned Logistic Regression to {models_dir / 'log_reg_tuned.pkl'}")
            click.echo(f"Saved tuned RBF SVM to {models_dir / 'svm_rbf_tuned.pkl'}")
    click.echo(f"Wrote logistic regression coefficient table to {coef_table_path}")
    click.echo(f"Wrote logistic regression coefficient figure to {coef_fig_path}")
    click.echo(f"Wrote logistic regression coefficient table image to {coef_table_img_path}")


if __name__ == "__main__":
    main()
