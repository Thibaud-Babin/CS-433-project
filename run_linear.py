"""Linear-model pipeline of the report (regularized logistic regression).

This script runs the whole pipeline end-to-end and writes the predictions for
the test set to a csv file in the format required by AIcrowd
(see data/sample_submission.csv):

1. load the BRFSS data (with the .npy cache from preprocessing.load_data),
2. preprocess the features (Preprocessor, all statistics learned on train),
3. pick the decision threshold by stratified k-fold cross-validation
   (maximizing the F1 score on the out-of-fold predictions),
4. retrain the model on the full training set,
5. predict on the test set and map the scores to {-1, +1}.

The pipeline is deterministic for a given --seed (the only random method,
SGD, is not used by default), so re-running it reproduces the same csv.

Usage:
    python run_linear.py                   # reg. logistic, all features
    python run_linear.py --features selected      # the 21 hand-picked features
    python run_linear.py --method least_squares   # pick another method
    python run_linear.py --help                   # all options

See README.md for the full pipeline description.
"""

import argparse

import numpy as np

from cross_validation import build_k_indices, cross_validate, train_model
from helpers import create_csv_submission
from preprocessing import SELECTED_FEATURES, Preprocessor, load_data, to_01, to_pm1

# Defaults of the best model found with `python tune.py`
# (see results/cv_results.csv):
#   reg_logistic(lambda=0.0001, gamma=0.5, iters=500) on all features -> F1 = 0.4290.
DEFAULT_LAMBDA = 1e-4
DEFAULT_GAMMA = 0.5
DEFAULT_MAX_ITERS = 500


def build_model(args):
    """Return the model dict understood by cross_validation.train_model."""
    model = {"name": args.method, "method": args.method}
    if args.method in ("ridge", "reg_logistic"):
        model["lambda"] = args.lambda_
    if args.method in ("mse_gd", "mse_sgd", "logistic", "reg_logistic"):
        model["gamma"] = args.gamma
        model["max_iters"] = args.max_iters
    return model


def run_pipeline(
    x_train, x_test, y_train, test_ids, columns, model, features, folds, seed
):
    """Train on x_train, predict on x_test and return the {-1, +1} labels.

    Returns a tuple (y_pred, cv_f1, threshold). cv_f1 and threshold are the
    out-of-fold F1 score and the threshold that maximizes it, both obtained
    with a stratified k-fold cross-validation.
    """
    # Learn every preprocessing statistic on the training set only.
    preprocessor = Preprocessor(features=features)
    tx_train = preprocessor.fit_transform(x_train, columns)
    tx_test = preprocessor.transform(x_test)
    y = to_01(y_train)  # the methods are trained on labels in {0, 1}

    # Threshold and honest F1 estimate from out-of-fold predictions.
    k_indices = build_k_indices(y, folds, seed)
    results = cross_validate(tx_train, y, k_indices, [model])
    if model["name"] not in results:
        raise RuntimeError(f"{model['name']} failed during cross-validation")
    threshold = results[model["name"]]["threshold"]
    cv_f1 = results[model["name"]]["f1"]

    # Retrain on the full training set with the same hyperparameters.
    w, _ = train_model(model, y, tx_train)

    scores = tx_test.dot(w)
    y_pred = to_pm1(scores >= threshold)
    return y_pred, cv_f1, threshold


def main():
    parser = argparse.ArgumentParser(
        description="Train a model and write the AIcrowd submission csv."
    )
    parser.add_argument(
        "--features",
        choices=["selected", "all"],
        default="all",
        help="all = every non-metadata feature, selected = the 21 "
        "hand-picked features (default: %(default)s)",
    )
    parser.add_argument(
        "--method",
        default="reg_logistic",
        choices=[
            "least_squares",
            "ridge",
            "mse_gd",
            "mse_sgd",
            "logistic",
            "reg_logistic",
        ],
    )
    parser.add_argument(
        "--lambda",
        dest="lambda_",
        type=float,
        default=DEFAULT_LAMBDA,
        help="regularization for ridge / reg_logistic (default: %(default)s)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=DEFAULT_GAMMA,
        help="step size for the gradient methods (default: %(default)s)",
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=DEFAULT_MAX_ITERS,
        help="iterations for the gradient methods (default: %(default)s)",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="cross-validation folds (default: %(default)s)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="random seed for reproducible folds (default: %(default)s)",
    )
    parser.add_argument(
        "--data_dir",
        default="data",
        help="folder with x_train.csv, y_train.csv, x_test.csv",
    )
    parser.add_argument(
        "--output",
        default="submission.csv",
        help="path of the generated csv (default: %(default)s)",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)  # only used by mean_squared_error_sgd

    x_train, x_test, y_train, _, test_ids, columns = load_data(args.data_dir)
    features = SELECTED_FEATURES if args.features == "selected" else None
    model = build_model(args)

    y_pred, cv_f1, threshold = run_pipeline(
        x_train,
        x_test,
        y_train,
        test_ids,
        columns,
        model,
        features,
        args.folds,
        args.seed,
    )

    create_csv_submission(test_ids, y_pred, args.output)

    print(f"Feature set        : {args.features}")
    print(f"Model              : {args.method}")
    print(f"Cross-val F1       : {cv_f1:.4f}")
    print(f"Decision threshold : {threshold:.4f}")
    print(f"Wrote {args.output} ({len(y_pred)} predictions)")


if __name__ == "__main__":
    main()
