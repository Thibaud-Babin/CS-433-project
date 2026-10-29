"""Hyperparameter search with stratified k-fold cross-validation.

Usage:
    python tune.py --features selected   # the 21 hand-picked features
    python tune.py --features all        # every non-metadata feature

One line per model is added to results/cv_results.csv.
"""

import argparse
import csv
import os
import time

import numpy as np

from cross_validation import build_k_indices, cross_validate
from preprocessing import SELECTED_FEATURES, Preprocessor, load_data, to_01

RESULTS_PATH = os.path.join("results", "cv_results.csv")


def largest_eigenvalue(tx):
    """Largest eigenvalue of tx^T tx / N, computed on every 10th sample
    to save time and memory."""
    sub = tx[::10].astype(np.float64)
    eigenvalues = np.linalg.eigvalsh(sub.T.dot(sub) / len(sub))
    # a Python float (not numpy float64), otherwise w becomes float64 and the
    # whole float32 matrix is converted at every gradient step
    return float(eigenvalues[-1])


def make_models(tx):
    """List of the models (method + hyperparameters) to evaluate.

    MSE gradient descent converges for gamma < 2 / L (L = largest eigenvalue),
    so we use gamma = 1 / L. For logistic regression on standardized
    features, GD diverged for gamma >= 1 and converged in ~500 iterations
    for gamma = 0.5 on one fold.
    """
    gamma_mse = 1 / largest_eigenvalue(tx)
    models = []

    models.append({"name": "least_squares", "method": "least_squares"})
    models.append(
        {
            "name": f"mse_gd(gamma={gamma_mse:.3f},iters=500)",
            "method": "mse_gd",
            "gamma": gamma_mse,
            "max_iters": 500,
        }
    )
    models.append(
        {
            "name": "mse_sgd(gamma=0.001,iters=20000)",
            "method": "mse_sgd",
            "gamma": 0.001,
            "max_iters": 20000,
        }
    )

    for lambda_ in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
        models.append(
            {"name": f"ridge(lambda={lambda_:g})", "method": "ridge", "lambda": lambda_}
        )

    for gamma in [0.25, 0.5]:
        models.append(
            {
                "name": f"logistic(gamma={gamma:g},iters=500)",
                "method": "logistic",
                "gamma": gamma,
                "max_iters": 500,
            }
        )

    for lambda_ in [1e-4, 1e-3, 1e-2]:
        models.append(
            {
                "name": f"reg_logistic(lambda={lambda_:g},gamma=0.5,iters=500)",
                "method": "reg_logistic",
                "lambda": lambda_,
                "gamma": 0.5,
                "max_iters": 500,
            }
        )
    return models


def save_results(results, features, n_columns):
    """Add one line per model to the results csv file."""
    os.makedirs("results", exist_ok=True)
    write_header = not os.path.exists(RESULTS_PATH)
    with open(RESULTS_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            header = ["features", "n_columns", "model", "f1", "f1_std"]
            header += ["accuracy", "threshold", "train_loss"]
            writer.writerow(header)
        for name in results:
            r = results[name]
            row = [features, n_columns, name]
            row.append(f"{r['f1']:.4f}")
            row.append(f"{r['fold_f1'].std():.4f}")
            row.append(f"{r['accuracy']:.4f}")
            row.append(f"{r['threshold']:.4f}")
            row.append(f"{r['train_loss']:.4f}")
            writer.writerow(row)


def print_results(results):
    """Print the models sorted from best to worst F1 score."""
    names = list(results.keys())
    names.sort(key=lambda name: results[name]["f1"], reverse=True)
    print(f"{'model':<50} {'F1':>7} {'+/- std':>7} {'acc':>7} {'thresh':>8}")
    for name in names:
        r = results[name]
        print(
            f"{name:<50} {r['f1']:7.4f} {r['fold_f1'].std():7.4f} "
            f"{r['accuracy']:7.4f} {r['threshold']:8.4f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter search")
    parser.add_argument("--features", choices=["selected", "all"], default="all")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    # the seed is also used by mean_squared_error_sgd
    np.random.seed(args.seed)

    x_train, x_test, y_train, train_ids, test_ids, columns = load_data()
    if args.features == "selected":
        features = SELECTED_FEATURES
    else:
        features = None
    preprocessor = Preprocessor(features=features)
    tx = preprocessor.fit_transform(x_train, columns)
    y = to_01(y_train)
    print(f"Feature set '{args.features}': tx has shape {tx.shape}")

    k_indices = build_k_indices(y, args.folds, args.seed)
    start = time.time()
    results = cross_validate(tx, y, k_indices, make_models(tx))
    print(f"Cross-validation took {time.time() - start:.0f}s\n")

    save_results(results, args.features, tx.shape[1])
    print_results(results)


if __name__ == "__main__":
    main()
