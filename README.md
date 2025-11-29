# Adult Income Prediction

## Introduction
This project uses the Adult Income dataset to explore general patterns in income inequality. Our aim is to understand, at a high level, how personal and work-related characteristics relate to income categories and to build a clear, reproducible starting point for further analysis.


## Contributors
- Chun-Mien Liu
- Rebecca Rosette Nanfuka
- Roganci Fontelera
- Yonas Gebre Marie

## Project overview

This project examines the Adult Income dataset to understand how demographic and socioeconomic factors influence whether an individual earns more or less than $50,000 per year. By analyzing variables such as age, education level, occupation, marital status, work hours, and race, the project uncovers broad patterns in income distribution and highlights which characteristics are most strongly associated with higher earnings. The goal is to provide a clear, high-level perspective on income inequality across different demographic groups while building a reproducible and transparent foundation for further data exploration and modeling.

## How to run the analysis
1. Clone the repository and move into it: `git clone <repo-url> && cd AdultIncomePrediction`.
2. Build the Conda environment defined in `environment.yml`: `conda env create -f environment.yml`.
3. Activate the environment: `conda activate adult-income-prediction`.
4. Launch Jupyter Lab from the project root: `jupyter lab`.
5. Open or create your analysis notebook (e.g., `reports/income-prediction.ipynb`), point it to `data/adult.csv`, and execute the cells to reproduce the charts and summary metrics.

## Data validation
Use the checklist-driven validator in `analysis/data_validation.py` to verify that the Adult Income dataset matches the expected schema and quality thresholds.

```bash
python analysis/data_validation.py data/adult.csv
```

Omit the file argument to automatically validate the default dataset at `data/adult.csv`.

Each row in the output corresponds to an item in the validation checklist (file format, column names, missingness, dtypes, duplicates, outliers, category levels, target distribution, and correlation screens). Any failing check prints the offending columns or metrics so you can take corrective action.

Pandera users: if the `pandera` package is installed in the active environment, the validator automatically runs an additional schema check that enforces the data types, ranges, and categorical domains defined for the Adult dataset.

## Dependencies
- Python 3.12+
- pandas
- jupyterlab
- See `environment.yml` for the authoritative set of packages and versions used in the project.

## License
This repository is distributed under the MIT License (see `LICENSE` for the full text).

## References

UCI Machine Learning Repository. (1996). Adult Dataset.
Retrieved from: https://archive.ics.uci.edu/dataset/2/adult
Kohavi, R. (1996). Scaling Up the Accuracy of Naive-Bayes Classifiers: A Decision-Tree Hybrid. Proceedings of the Second International Conference on Knowledge Discovery and Data Mining (KDD).
U.S. Census Bureau. Current Population Survey (CPS). https://www.census.gov/programs-surveys/cps.html
