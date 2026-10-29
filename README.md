# CS-433 Project 1 - Heart disease prediction

We predict if a person has coronary heart disease (MICHD) from their answers
to the 2015 BRFSS health survey. Only Python and NumPy are used
(matplotlib only for the figure of the report).

## Files

- `implementations.py`: the 6 methods asked in the project
- `helpers.py`: functions given by the course (load the csv, write the submission)
- `preprocessing.py`: loads and cleans the data for the linear models
- `cross_validation.py`: F1 score, best threshold and 5-fold cross-validation
- `tune.py`: tries the 6 methods with different parameters
- `gbdt.py`: our boosted trees model
- `run.py`: makes our best submission (boosted trees)
- `run_linear.py`: makes a submission with the linear models
- `eda.ipynb`: first look at the data
- `test_implementations.py`, `test_pipeline.py`: tests
- `report/`: the report

## How to run

Put `x_train.csv`, `y_train.csv` and `x_test.csv` in the `data/` folder.
The first time, loading the csv files takes about 1 minute, then they are
saved in `data/cache/` and loading is fast.

```
python run.py          # writes submission.csv (about 6 minutes)
python run.py --cv     # 5-fold cross-validation of the boosted trees
python run_linear.py   # submission with regularized logistic regression
python tune.py         # compares the 6 methods
pytest test_implementations.py test_pipeline.py
```

## What we did

**Preprocessing (linear models)**, all statistics are computed on the training set:
1. remove the survey metadata and columns with more than 80% missing values
2. special codes: 88 / 888 / 555 mean "none" -> 0, 77 / 99 / 777 / 999 mean
   "don't know" or "refused" -> missing
3. columns with at most 15 values are one-hot encoded (missing is its own column)
4. other columns: missing values replaced by the median, plus a column saying
   if the value was missing, and extreme values are clipped
5. remove columns that are linear combinations of others, standardize, add a bias

**Boosted trees**: each column is cut into 64 bins, missing values get their
own bin. Trees of depth 3, learning rate 0.1, each tree uses half of the rows
and half of the columns. We stop adding trees when the loss on 10% of the
training set stops improving. The column HAREHAB1 is not used because it is
only asked to people who had a heart attack (it gives the label away).

**Threshold**: only 8.8% of the people are sick, so we choose the threshold
that gives the best F1 score instead of 0.5.

**Cross-validation**: stratified 5-fold cross-validation (each fold has 8.8%
of sick people), the same folds for every model.

## Results (5-fold cross-validation)

| Model | F1 | Accuracy |
|---|---|---|
| Regularized logistic regression | 0.4290 | 0.877 |
| Boosted trees (final) | 0.4357 | 0.872 |

All results of `tune.py` are in `results/cv_results.csv`.
