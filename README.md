# CS-433 Project 1 — Heart disease (MICHD) prediction

Binary classification on the BRFSS 2015 survey: predict whether a respondent
has coronary heart disease (MICHD) from ~320 health and lifestyle answers.
Only Python, NumPy and matplotlib are used (no external ML libraries).

## Repository layout

| File | Content |
|---|---|
| `implementations.py` | The 6 required methods (MSE GD/SGD, least squares, ridge, logistic and regularized logistic regression) |
| `helpers.py` | Course-provided CSV loading and submission writing |
| `preprocessing.py` | Data loading with `.npy` cache, feature cleaning / one-hot encoding / standardization (`Preprocessor`), label conversion |
| `cross_validation.py` | F1 / accuracy, F1-optimal threshold, stratified k-fold cross-validation |
| `tune.py` | Hyperparameter search of all methods with 5-fold CV, results in `results/cv_results.csv` |
| `eda.ipynb` | Exploratory data analysis |
| `test_implementations.py`, `test_pipeline.py` | Unit tests (`pytest`) |
| `run.py` | *To do:* reproduces our best submission |

## Setup

Put `x_train.csv`, `y_train.csv` and `x_test.csv` in `data/` (they are
tracked with Git LFS: `git lfs pull`). The first data load parses the CSVs
(~4 min) and caches them in `data/cache/`; later loads take seconds.

```
python tune.py --features selected   # 21 hand-picked features
python tune.py --features all        # all non-metadata features
pytest test_implementations.py test_pipeline.py
```

## Pipeline

1. **Labels**: the data uses y in {-1, 1}; models are trained on {0, 1}
   (`to_01`) and predictions are mapped back with `to_pm1` for submission.
2. **Feature selection**: survey metadata (dates, sequence numbers, sampling
   weights) and columns with more than 80% missing values are dropped.
3. **Categorical features** (at most 15 distinct values) are one-hot encoded,
   with the most frequent value as reference level. "Don't know" / "Refused"
   codes and NaN become their own categories; categories seen in less than
   0.5% of the training samples are merged into the reference.
4. **Continuous features**: BRFSS codes are decoded (88 / 888 / 555 -> 0,
   77 / 99 / 777 / 999 / ... -> missing), missing values are imputed with the
   training median plus a missing indicator, values are clipped to the
   [0.1%, 99.9%] training quantiles.
5. Columns that are linear combinations of others are dropped (keeps
   `tx^T tx` invertible), all features are standardized with training
   statistics, and a bias column is added.
6. **Evaluation**: stratified 5-fold CV. Because only 8.8% of samples are
   positive, we report the F1 score, with the decision threshold chosen to
   maximize F1 on the out-of-fold scores.
