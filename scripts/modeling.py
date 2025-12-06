"""
Train income prediction models using preprocessed training features.

This script expects a CSV with the engineered training features and the income
label (from the preprocessing script) and writes cross-validation summaries plus
simple coefficient plots to disk. Tables are written to a user-provided data
directory and figures to a user-provided image directory (PNG + HTML). It takes
three arguments that align with the data layout:

1. Path to the preprocessed training data CSV that includes the target column `income`
   (e.g., `data/processed/train_preprocessed.csv`).
2. Output directory for CSVs (e.g., `data/output`).
3. Output directory for images (e.g., `data/output/img`).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

# Limit joblib worker discovery in constrained environments
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import altair as alt
import numpy as np
import pandas as pd
import click
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


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


def evaluate_models(
    models: Dict[str, object],
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int,
    n_jobs: int,
) -> pd.DataFrame:
    rows = []
    for name, model in models.items():
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


def build_cv_chart(cv_summary: pd.DataFrame) -> alt.Chart:
    df = cv_summary.reset_index().copy()
    df["lower"] = df["test_accuracy_mean"] - df["test_accuracy_std"]
    df["upper"] = df["test_accuracy_mean"] + df["test_accuracy_std"]
    sort_order = df.sort_values("test_accuracy_mean")["model"].tolist()

    bars = (
        alt.Chart(df)
        .mark_bar(color="steelblue")
        .encode(
            y=alt.Y("model:N", sort=sort_order, title="Model"),
            x=alt.X("test_accuracy_mean:Q", title="CV accuracy"),
            tooltip=[
                alt.Tooltip("test_accuracy_mean:Q", format=".3f", title="Mean"),
                alt.Tooltip("test_accuracy_std:Q", format=".3f", title="Std"),
            ],
        )
    )

    errorbars = (
        alt.Chart(df)
        .mark_errorbar()
        .encode(
            y=alt.Y("model:N", sort=sort_order),
            x=alt.X("lower:Q"),
            x2=alt.X2("upper:Q"),
        )
    )

    return (bars + errorbars).properties(
        title="Model comparison (CV mean ± std)", width=520, height=240
    )


def build_log_reg_chart(coef_df: pd.DataFrame, top_n: int) -> alt.VConcatChart:
    top_positive = coef_df.nlargest(top_n, "coefficient").copy()
    top_negative = coef_df.nsmallest(top_n, "coefficient").copy()

    pos_order = top_positive.sort_values("coefficient")["feature"].tolist()
    neg_order = top_negative.sort_values("coefficient")["feature"].tolist()

    pos_chart = (
        alt.Chart(top_positive)
        .mark_bar(color="seagreen")
        .encode(
            y=alt.Y("feature:N", sort=pos_order, title="Feature"),
            x=alt.X("coefficient:Q", title="Log-odds"),
            tooltip=["feature", alt.Tooltip("coefficient:Q", format=".3f"), alt.Tooltip("odds_ratio:Q", format=".3f")],
        )
        .properties(title=f"Top +{top_n} coefficients", width=300, height=220)
    )

    neg_chart = (
        alt.Chart(top_negative)
        .mark_bar(color="firebrick")
        .encode(
            y=alt.Y("feature:N", sort=neg_order, title="Feature"),
            x=alt.X("coefficient:Q", title="Log-odds"),
            tooltip=["feature", alt.Tooltip("coefficient:Q", format=".3f"), alt.Tooltip("odds_ratio:Q", format=".3f")],
        )
        .properties(title=f"Top -{top_n} coefficients", width=300, height=220)
    )

    return alt.hconcat(pos_chart, neg_chart).resolve_scale(y="independent")


def save_altair_html(chart: alt.Chart, output_path: Path) -> None:
    output_path = output_path.with_suffix(".html")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    chart.save(output_path)


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
@click.argument(
    "train_path",
    default="data/processed/train_preprocessed.csv",
    required=False,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.argument(
    "output_dir",
    default="data/output",
    required=False,
    type=click.Path(path_type=Path),
)
@click.argument(
    "img_dir",
    default="data/output/img",
    required=False,
    type=click.Path(path_type=Path),
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
def main(train_path: Path, output_dir: Path, img_dir: Path, cv_folds: int, top_n: int, n_jobs: int) -> None:
    output_dir = ensure_dir(output_dir)
    img_dir = ensure_dir(img_dir)

    X_train, y_train = load_training_data(str(train_path))

    models = build_models()
    cv_summary = evaluate_models(models, X_train, y_train, cv_folds, n_jobs)

    cv_table_path = output_dir / "cv_summary.csv"
    coef_table_path = output_dir / "log_reg_coefficients.csv"
    cv_fig_path = img_dir / "cv_summary.png"
    coef_fig_path = img_dir / "log_reg_coefficients.png"
    cv_table_img_path = img_dir / "cv_summary_table.png"
    coef_table_img_path = img_dir / "log_reg_coefficients_table.png"

    cv_summary.to_csv(cv_table_path)
    save_altair_html(build_cv_chart(cv_summary), cv_fig_path.with_suffix(".html"))
    save_cv_chart_png(cv_summary, cv_fig_path)
    save_table_image(cv_summary.reset_index(), cv_table_img_path, "CV summary table")

    # Fit logistic regression on full training set for interpretability artifacts.
    log_reg = models["LogisticRegression"]
    log_reg.fit(X_train, y_train)
    coef_df = pd.DataFrame(
        {
            "feature": X_train.columns,
            "coefficient": log_reg.coef_[0],
            "odds_ratio": np.exp(log_reg.coef_[0]),
        }
    )
    save_altair_html(build_log_reg_chart(coef_df, top_n), coef_fig_path.with_suffix(".html"))
    save_coef_chart_png(coef_df, top_n, coef_fig_path)
    save_table_image(coef_df, coef_table_img_path, "Logistic regression coefficients")
    coef_df.to_csv(coef_table_path, index=False)

    click.echo(f"Wrote CV summary table to {cv_table_path}")
    click.echo(f"Wrote CV summary figure to {cv_fig_path}")
    click.echo(f"Wrote CV summary table image to {cv_table_img_path}")
    click.echo(f"Wrote logistic regression coefficient table to {coef_table_path}")
    click.echo(f"Wrote logistic regression coefficient figure to {coef_fig_path}")
    click.echo(f"Wrote logistic regression coefficient table image to {coef_table_img_path}")


if __name__ == "__main__":
    main()
