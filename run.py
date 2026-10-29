import sys
import time

import numpy as np

from cross_validation import accuracy, best_threshold, build_k_indices, f1_score
from gbdt import GBDT, apply_bins, fit_bin_edges
from helpers import create_csv_submission
from preprocessing import METADATA_COLUMNS, load_data, to_01, to_pm1

EXCLUDED_COLUMNS = METADATA_COLUMNS + ["HAREHAB1"]

PARAMS = {
    "n_trees": 1500,
    "learning_rate": 0.1,
    "max_depth": 3,
    "lam": 1.0,
    "min_child_hess": 5.0,
    "subsample": 0.5,
    "colsample": 0.5,
    "early_stopping": 50,
}


def stratified_split(y, frac, seed):
    rng = np.random.default_rng(seed)
    b = []
    for label in [0, 1]:
        indices = rng.permutation(np.where(y == label)[0])
        n = int(round(frac * len(indices)))
        b.extend(indices[:n])
    b = np.sort(np.array(b, dtype=int))
    is_a = np.ones(len(y), dtype=bool)
    is_a[b] = False
    a = np.where(is_a)[0]
    return a, b


def train(xb, y, seed):
    fit_idx, stop_idx = stratified_split(y, 0.1, seed)
    model = GBDT(
        n_trees=PARAMS["n_trees"],
        learning_rate=PARAMS["learning_rate"],
        max_depth=PARAMS["max_depth"],
        lam=PARAMS["lam"],
        min_child_hess=PARAMS["min_child_hess"],
        subsample=PARAMS["subsample"],
        colsample=PARAMS["colsample"],
        early_stopping=PARAMS["early_stopping"],
        seed=seed,
        verbose=100,
    )
    model.fit(xb[fit_idx], y[fit_idx], xb[stop_idx], y[stop_idx])
    scores = model.decision_function(xb[stop_idx])
    threshold, f1 = best_threshold(y[stop_idx], scores)
    print(f"  {len(model.trees)} trees, threshold {threshold:.3f}, F1 {f1:.4f}")
    return model, threshold


def cross_validation(xb, y):
    k_indices = build_k_indices(y, 5, 1)
    scores = np.zeros(len(y))
    for k in range(len(k_indices)):
        val_idx = k_indices[k]
        is_train = np.ones(len(y), dtype=bool)
        is_train[val_idx] = False
        train_idx = np.where(is_train)[0]
        print(f"fold {k + 1}/5")
        model, _ = train(xb[train_idx], y[train_idx], seed=100 + k)
        scores[val_idx] = model.decision_function(xb[val_idx])

    threshold, f1 = best_threshold(y, scores)
    y_pred = scores >= threshold
    fold_f1 = []
    for val_idx in k_indices:
        fold_f1.append(f1_score(y[val_idx], y_pred[val_idx]))
    print(f"CV F1 = {f1:.4f} +- {np.std(fold_f1):.4f}")
    print(f"CV accuracy = {accuracy(y, y_pred):.4f}")
    print("fold F1:", " ".join(f"{f:.4f}" for f in fold_f1))


def main():
    start = time.time()
    x_train, x_test, y_train, train_ids, test_ids, columns = load_data("data")
    y = to_01(y_train)

    used = []
    for j in range(len(columns)):
        if columns[j] not in EXCLUDED_COLUMNS:
            used.append(j)
    edges = fit_bin_edges(x_train, used)
    xb_train = apply_bins(x_train, used, edges)
    print(f"binned training set: {xb_train.shape} ({time.time() - start:.0f}s)")

    if "--cv" in sys.argv:
        cross_validation(xb_train, y)
    else:
        model, threshold = train(xb_train, y, seed=0)
        xb_test = apply_bins(x_test, used, edges)
        y_pred = model.decision_function(xb_test) >= threshold
        create_csv_submission(test_ids, to_pm1(y_pred), "submission.csv")
        print(f"predicted positive rate on the test set: {np.mean(y_pred):.3f}")

        counts = model.split_counts(len(used))
        top = np.argsort(-counts)[:10]
        names = []
        for k in top:
            names.append(f"{columns[used[k]]} ({int(counts[k])})")
        print("most used features:", ", ".join(names))
    print(f"total time {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
