# College Major Analysis

## Introduction
This project uses the Adult Income dataset to explore general patterns in income inequality. Our aim is to understand, at a high level, how personal and work-related characteristics relate to income categories and to build a clear, reproducible starting point for further analysis.


## Contributors
- Chun-Mien Liu
- Rebecca Rosette Nanfuka
- Roganci Fontelera
- Yonas Gebre Marie

## Project overview

This project explores the U.S. college majors dataset (`data/adult.csv`) to understand how fields of study influence employment rates, salaries, and demographic outcomes. The analysis is designed to surface broad labor-market patterns (e.g., which majors have the lowest unemployment or the highest earnings) so that educators and students can get a 10,000-foot view of the trade-offs among different disciplines.

## How to run the analysis
1. Clone the repository and move into it: `git clone <repo-url> && cd CollegeMajorAnalysis`.
2. Build the Conda environment defined in `environment.yml`: `conda env create -f environment.yml`.
3. Activate the environment: `conda activate college-major-analysis`.
4. Launch Jupyter Lab from the project root: `jupyter lab`.
5. Open or create your analysis notebook (e.g., `reports/college_major_analysis.ipynb`), point it to `data/recent-grads.csv`, and execute the cells to reproduce the charts and summary metrics.

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

