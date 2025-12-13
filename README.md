# Adult Income Prediction

This project explores the Adult Income dataset to understand how demographic and work-related factors relate to earning more or less than **$50,000 per year**. It provides a fully reproducible **Make + Docker + Quarto** workflow for data preparation, validation, exploratory data analysis (EDA), modeling, and report rendering.

## Contributors
- Chun-Mien Liu  
- Rebecca Rosette Nanfuka  
- Roganci Fontelera  
- Yonas Gebre Marie  

---
## Project Structure (high-level)

```text
.
├── data/
│   ├── raw/                 # original Adult dataset (downloaded)
│   └── processed/           # cleaned, split, and preprocessed datasets
│
├── artifacts/
│   ├── figures/             # EDA and model evaluation plots
│   ├── tables/              # metrics, coefficients, CV summaries
│   └── models/              # trained model artifacts (e.g., pickles)
│
├── src/                     # reusable pipeline code (cleaning, EDA, modeling)
│
├── test/                    # data and pipeline validation tests
│
├── reports/
│   └── income-prediction.qmd  # Quarto report source
│
├── docs/
│   └── index.html           # rendered Quarto report (final output)
│
├── _quarto.yml              # Quarto project configuration
├── environment.yml          # Conda environment & dependencies
├── conda-lock.yml           # locked, reproducible dependency versions
│
├── Dockerfile               # container definition
├── docker-compose.yml       # container orchestration
├── Makefile                 # reproducible pipeline and report commands
│
├── README.md                # project overview and usage instructions
├── CONTRIBUTING.md          # contribution guidelines
├── CODE_OF_CONDUCT.md       # community standards
├── LICENSE                  # project license
└── .gitignore               # ignored files and directories


```

---

## Requirements

On your machine:
- Docker Desktop (or Docker Engine)
- (Optional) Make

Inside the container:
- Python environment and dependencies defined in [`environment.yml`](environment.yml)
- Quarto (installed in the image / Conda environment)

All Python dependencies are declared in **[`environment.yml`](environment.yml)** and are installed during the Docker build to ensure full reproducibility.

---

## Quick start

```bash
# 1) Clone the repository
git clone <repo-url> IncomePrediction
cd IncomePrediction

# 2) Build the Docker image
make build
# or equivalently:
# docker build -t 522-milestone -f Dockerfile .

# 3) Start the container
make up
# or:
# docker-compose up -d

# 4) Run the full analysis pipeline and render the report
docker exec -it 522-milestone bash -lc 'eval "$(/opt/conda/bin/conda shell.bash hook)" && conda activate 522-milestone && make all'
```

After `make all` completes, the rendered report will be available at: <http://localhost:8889/>

---

## How to run (detailed)

### Build the image

```bash
make build
```

This builds the Docker image defined in `Dockerfile`, installing all dependencies from
[`environment.yml`](environment.yml).

Rebuild only if:

* `Dockerfile` changes
* `environment.yml` changes

---

### Start and stop the container

```bash
make up
make stop
```

---

### Run commands inside the container

All analysis and reporting commands are intended to run **inside the container**.

Open an interactive shell:

```bash
docker exec -it 522-milestone bash
conda activate 522-milestone
```

Run a single command directly:

```bash
docker exec -it 522-milestone bash -lc "<command>"
```

Examples:

```bash
docker exec -it 522-milestone bash -lc "make help"
docker exec -it 522-milestone bash -lc "make analysis"
docker exec -it 522-milestone bash -lc "make report"
```

Exit the container shell with `exit` or `Ctrl+D`.

---

## Make targets (what to run)

### Full pipeline + report

```bash
make all
```

Equivalent to:

```bash
make data validate split preprocess eda model report
```

---

### Analysis only (Python pipeline, no Quarto)

```bash
make analysis
```

This runs:

1. Download and clean raw data
2. Validate cleaned data
3. Split data into train/test
4. Run EDA and generate figures
5. Preprocess features
6. Train models with hyperparameter tuning
7. Generate tables and plots

---

### Individual steps

Download and clean data:

```bash
make data
```

Validate cleaned data (non-failing):

```bash
make validate
```

Split into train/test:

```bash
make split
```

Preprocess train and test data:

```bash
make preprocess
```

Run EDA:

```bash
make eda
```

Train models and generate artifacts:

```bash
make model
```

Render the Quarto report/site:

```bash
make report
```

---

## Outputs

### Data

* Raw data: `data/raw/adult.csv`
* Cleaned data: `data/processed/clean_adult.csv`
* Train/test split: `data/processed/train.csv`, `data/processed/test.csv`
* Preprocessed data:

  * `data/processed/preprocessed_train.csv`
  * `data/processed/preprocessed_test.csv`

### EDA figures

Saved to:

```text
artifacts/figures/
```

### Modeling artifacts

Saved to:

```text
artifacts/tables/
artifacts/figures/
```

Includes:

* Cross-validation summaries
* Model coefficients
* Hyperparameter search results
* Performance plots and table images

### Report

* Rendered output: `docs/index.html`

---

## Quarto notes

The Quarto source file is:

```text
reports/income-prediction.qmd
```

Rendering inside the container can be done with:

```bash
quarto render
```

The report expects all analysis artifacts (EDA + modeling outputs) to exist before rendering.

---

## Cleaning up

Remove processed data and figures (keeps directory structure):

```bash
make clean
```

Stop and remove containers:

```bash
docker-compose down
```

---

## Troubleshooting

**Container not found**

* Ensure it is running: `docker ps`
* Start it: `make up`

**Quarto not found inside container**

* Ensure Quarto is listed in [`environment.yml`](environment.yml) and rebuild the image

**Report missing plots or tables**

* Run the analysis first:

```bash
make analysis
make report
```

or simply:

```bash
make all
```

---

## License

Distributed under the MIT License (see [LICENSE](LICENSE)).

---

## References

* UCI Machine Learning Repository. (1996). Adult Dataset. [https://archive.ics.uci.edu/dataset/2/adult](https://archive.ics.uci.edu/dataset/2/adult)
* Kohavi, R. (1996). Scaling Up the Accuracy of Naive-Bayes Classifiers: A Decision-Tree Hybrid. KDD.
* U.S. Census Bureau. Current Population Survey (CPS). [https://www.census.gov/programs-surveys/cps.html](https://www.census.gov/programs-surveys/cps.html)

