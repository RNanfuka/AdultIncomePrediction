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
1. Clone the repository and move into it: 
```
git clone <repo-url> && cd AdultIncomePrediction
```
2. Run the following command to start docker container:
```
docker compose up
```
3. In the terminal, look for a URL that starts with 
`http://127.0.0.1:8888/lab?token=` 
(for an example, see the highlighted text in the terminal below). 
Copy and paste that URL into your browser.
4. To run the analysis,
open a terminal and run the following commands:
```
make run-all-py
```
5. Navigate to the `reports` directory:
```
cd reports
```
6. Generate HTML:
```
quarto render income-prediction.qmd
```
5. Run `quarto preview`, and open up the URL mentioned in the terminal output to see the HTML:
```
quarto preview income-prediction.qmd
```

### Clean up
To shut down the container and clean up the resources, 
type `Ctrl` + `C` in the terminal
where you launched the container, and then type `docker compose rm`

## Dependencies
- Docker Desktop

## License
This repository is distributed under the MIT License (see `LICENSE` for the full text).

## References

UCI Machine Learning Repository. (1996). Adult Dataset.
Retrieved from: https://archive.ics.uci.edu/dataset/2/adult
Kohavi, R. (1996). Scaling Up the Accuracy of Naive-Bayes Classifiers: A Decision-Tree Hybrid. Proceedings of the Second International Conference on Knowledge Discovery and Data Mining (KDD).
U.S. Census Bureau. Current Population Survey (CPS). https://www.census.gov/programs-surveys/cps.html

