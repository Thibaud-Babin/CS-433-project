import numpy as np

from implementations import (
    least_squares,
    logistic_regression,
    mean_squared_error_gd,
    mean_squared_error_sgd,
    reg_logistic_regression,
    ridge_regression,
)


def f1_score(y_true, y_pred):
    """F1 score of binary predictions."""
    y_true = np.array(y_true) == 1
    y_pred = np.array(y_pred) == 1
    tp = np.sum(y_true & y_pred)
    fp = np.sum(~y_true & y_pred)
    fn = np.sum(y_true & ~y_pred)
    if 2 * tp + fp + fn == 0:
        return 0.0
    return 2 * tp / (2 * tp + fp + fn)


def accuracy(y_true, y_pred):
    """Fraction of correct binary predictions."""
    y_true = np.array(y_true) == 1
    y_pred = np.array(y_pred) == 1
    return np.mean(y_true == y_pred)


def best_threshold(y_true, scores):
    """Find the threshold on the scores that gives the best F1 score."""
    order = np.argsort(-scores)
    sorted_scores = scores[order]
    sorted_y = y_true[order]
    n_positives = int(np.sum(y_true == 1))

    best_f1 = -1
    best_t = sorted_scores[0]
    tp = 0
    for i in range(len(sorted_scores)):

        if sorted_y[i] == 1:
            tp += 1

        if i + 1 < len(sorted_scores) and sorted_scores[i + 1] == sorted_scores[i]:
            continue
        n_predicted = i + 1
        f1 = 2 * tp / (n_predicted + n_positives)
        if f1 > best_f1:
            best_f1 = f1
            best_t = sorted_scores[i]
    return best_t, best_f1


def build_k_indices(y, k_fold, seed):
    """Split the sample indices into k folds with the same ratio of
    positives in each fold (stratified k-fold)."""
    rng = np.random.default_rng(seed)
    folds = []
    for k in range(k_fold):
        folds.append([])

    for label in [0, 1]:
        indices = np.where(y == label)[0]
        indices = rng.permutation(indices)
        parts = np.array_split(indices, k_fold)
        for k in range(k_fold):
            folds[k].extend(parts[k])

    k_indices = []
    for k in range(k_fold):
        k_indices.append(np.sort(np.array(folds[k], dtype=int)))
    return k_indices


def train_model(model, y, tx):
    """Train one model described by a dict (see tune.py) and return (w, loss)."""
    method = model["method"]
    n_features = tx.shape[1]
    initial_w = np.zeros(n_features, dtype=np.float32)

    if method == "least_squares":
        return least_squares(y, tx)
    elif method == "ridge":
        return ridge_regression(y, tx, model["lambda"])
    elif method == "mse_gd":
        return mean_squared_error_gd(
            y, tx, initial_w, model["max_iters"], model["gamma"]
        )
    elif method == "mse_sgd":
        return mean_squared_error_sgd(
            y, tx, initial_w, model["max_iters"], model["gamma"]
        )
    elif method == "logistic":
        return logistic_regression(y, tx, initial_w, model["max_iters"], model["gamma"])
    elif method == "reg_logistic":
        return reg_logistic_regression(
            y, tx, model["lambda"], initial_w, model["max_iters"], model["gamma"]
        )
    else:
        raise ValueError("Unknown method: " + method)


def cross_validate(tx, y, k_indices, models, verbose=True):
    """Run k-fold cross-validation for a list of models, all on the same folds.

    models is a list of dicts, each with a "name" and a "method" (plus the
    hyperparameters needed by that method, see train_model).

    Returns a dict {model name: results}. The results are computed on the
    out-of-fold scores: "threshold", "f1", "fold_f1" (one F1 per fold),
    "accuracy", "train_loss" (mean over folds) and "scores".
    """
    n = len(y)
    scores = {}
    losses = {}
    failed = []
    for model in models:
        scores[model["name"]] = np.zeros(n, dtype=np.float32)
        losses[model["name"]] = []

    for k in range(len(k_indices)):
        val_idx = k_indices[k]
        is_train = np.ones(n, dtype=bool)
        is_train[val_idx] = False
        tx_train = tx[is_train]
        y_train = y[is_train]
        tx_val = tx[val_idx]

        for model in models:
            name = model["name"]
            if name in failed:
                continue
            try:
                w, loss = train_model(model, y_train, tx_train)
            except np.linalg.LinAlgError as error:
                print("  " + name + " failed (" + str(error) + "), skipping it")
                failed.append(name)
                continue
            scores[name][val_idx] = tx_val.dot(w)
            losses[name].append(loss)
            if verbose:
                print(f"  fold {k + 1}/{len(k_indices)} {name}: loss={loss:.4f}")

    results = {}
    for model in models:
        name = model["name"]
        if name in failed:
            continue
        s = scores[name]
        threshold, f1 = best_threshold(y, s)
        y_pred = s >= threshold

        fold_f1 = []
        for val_idx in k_indices:
            fold_f1.append(f1_score(y[val_idx], y_pred[val_idx]))

        results[name] = {
            "threshold": threshold,
            "f1": f1,
            "fold_f1": np.array(fold_f1),
            "accuracy": accuracy(y, y_pred),
            "train_loss": np.mean(losses[name]),
            "scores": s,
        }
    return results
