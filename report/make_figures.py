import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from cross_validation import best_threshold, build_k_indices, cross_validate, f1_score
from preprocessing import Preprocessor, load_data, to_01

BEST_MODEL = {
    "name": "reg_logistic(lambda=1e-4,gamma=0.5,iters=500)",
    "method": "reg_logistic",
    "lambda": 1e-4,
    "gamma": 0.5,
    "max_iters": 500,
}

FOLDS = 5
SEED = 1


def f1_curve(y_true, scores):
    order = np.argsort(-scores)
    sorted_scores = scores[order]
    sorted_y = y_true[order]
    n_positives = int(np.sum(y_true == 1))

    thresholds = []
    f1s = []
    tp = 0
    for i in range(len(sorted_scores)):
        if sorted_y[i] == 1:
            tp += 1
        if i + 1 < len(sorted_scores) and sorted_scores[i + 1] == sorted_scores[i]:
            continue
        n_predicted = i + 1
        thresholds.append(sorted_scores[i])
        f1s.append(2 * tp / (n_predicted + n_positives))
    return np.array(thresholds), np.array(f1s)


def main():
    np.random.seed(SEED)

    x_train, _, y_train, _, _, columns = load_data()
    preprocessor = Preprocessor(features=None)
    tx = preprocessor.fit_transform(x_train, columns)
    y = to_01(y_train)
    print(f"tx shape: {tx.shape}, positive rate: {y.mean():.4f}")

    k_indices = build_k_indices(y, FOLDS, SEED)
    results = cross_validate(tx, y, k_indices, [BEST_MODEL], verbose=False)
    scores = results[BEST_MODEL["name"]]["scores"]

    threshold, best_f1 = best_threshold(y, scores)
    f1_at_0 = f1_score(y, scores >= 0.0)
    print(f"best threshold = {threshold:.4f}, F1 = {best_f1:.4f}")
    print(f"F1 at default threshold t=0 = {f1_at_0:.4f}")

    t, f1 = f1_curve(y, scores)

    step = max(1, len(t) // 2500)
    t_plot = t[::step]
    f1_plot = f1[::step]

    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.plot(t_plot, f1_plot, color="#1f77b4", lw=1.4, label="F1 vs. threshold")
    ax.axvline(
        0.0,
        color="0.55",
        ls="--",
        lw=1.0,
        label=f"default $t=0$ (F1={f1_at_0:.3f})",
    )
    ax.plot(
        [threshold],
        [best_f1],
        "o",
        color="#d62728",
        ms=6,
        zorder=5,
        label=f"optimum $t={threshold:.2f}$ (F1={best_f1:.4f})",
    )
    ax.set_xlabel("decision threshold $t$")
    ax.set_ylabel("F1 score")
    ax.set_title("F1 vs. decision threshold (regularized logistic, all features)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()

    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "fig_threshold.png"
    )
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
