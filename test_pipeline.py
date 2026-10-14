"""Tests for preprocessing.py and cross_validation.py (run with pytest)."""

import numpy as np
import pytest

from cross_validation import best_threshold, build_k_indices, cross_validate, f1_score
from implementations import least_squares
from preprocessing import Preprocessor, clean_sentinels, to_01, to_pm1


def test_label_conversions_are_inverse():
    y = np.array([-1, 1, 1, -1])
    np.testing.assert_array_equal(to_01(y), [0, 1, 1, 0])
    np.testing.assert_array_equal(to_pm1(to_01(y)), y)


def test_clean_sentinels_two_digit_field():
    # 88 = "none" -> 0, 77 / 99 = don't know / refused -> NaN
    col = np.array([1, 30, 88, 77, 99, np.nan, 55])
    np.testing.assert_array_equal(
        clean_sentinels(col), [1, 30, 0, np.nan, np.nan, np.nan, 55]
    )


def test_clean_sentinels_three_digit_field():
    col = np.array([101, 555, 888, 777, 999, 205])
    np.testing.assert_array_equal(
        clean_sentinels(col), [101, 0, 0, np.nan, np.nan, 205]
    )


def test_clean_sentinels_leaves_regular_values():
    col = np.array([12.5, 30.1, 97.6])
    np.testing.assert_array_equal(clean_sentinels(col), col)


@pytest.fixture
def raw_data():
    rng = np.random.default_rng(0)
    n = 2000
    cat = rng.choice([1.0, 2.0, 3.0, 9.0], size=n)
    cat[:100] = np.nan
    cont = rng.uniform(1, 30, size=n).round()
    cont[rng.random(n) < 0.1] = 88  # "none"
    cont[rng.random(n) < 0.05] = 99  # refused
    copy = cat.copy()  # duplicated column -> collinear features
    return np.column_stack([cat, cont, copy]), ["CAT", "CONT", "COPY"]


def test_preprocessor_output(raw_data):
    x, columns = raw_data
    p = Preprocessor(features=columns, max_categories=5)
    tx = p.fit_transform(x, columns)
    assert tx.dtype == np.float32
    assert not np.isnan(tx).any()
    np.testing.assert_array_equal(tx[:, 0], 1)  # bias
    np.testing.assert_allclose(tx[:, 1:].mean(axis=0), 0, atol=1e-5)
    np.testing.assert_allclose(tx[:, 1:].std(axis=0), 1, atol=1e-4)
    # The one-hot columns of COPY duplicate those of CAT and are dropped
    assert not any(name.startswith("COPY") for name in p.feature_names_)
    assert tx.shape[1] == len(p.feature_names_)
    # Full column rank, so least squares can be solved
    assert np.linalg.matrix_rank(tx.astype(np.float64)) == tx.shape[1]
    least_squares(np.zeros(len(tx)), tx)


def test_preprocessor_uses_train_statistics(raw_data):
    x, columns = raw_data
    p = Preprocessor(features=columns, max_categories=5).fit(x[:1000], columns)
    # Transforming the training part again gives the same rows
    np.testing.assert_allclose(p.transform(x[:1000])[:10], p.transform(x[:10]))
    assert p.transform(x[1000:]).shape == (1000, len(p.feature_names_))


def test_f1_score():
    y_true = np.array([1, 1, 0, 0, 1])
    y_pred = np.array([1, 0, 1, 0, 1])
    # tp = 2, fp = 1, fn = 1
    assert f1_score(y_true, y_pred) == pytest.approx(2 * 2 / (2 * 2 + 1 + 1))


def test_best_threshold_matches_brute_force():
    rng = np.random.default_rng(1)
    y = (rng.random(300) < 0.2).astype(float)
    scores = (y + rng.normal(size=300)).round(1)  # rounding creates ties
    threshold, f1 = best_threshold(y, scores)
    brute = max(f1_score(y, scores >= t) for t in np.unique(scores))
    assert f1 == pytest.approx(brute)
    assert f1_score(y, scores >= threshold) == pytest.approx(f1)


def test_build_k_indices_is_a_stratified_partition():
    y = np.r_[np.zeros(90), np.ones(10)]
    folds = build_k_indices(y, 5, seed=0)
    np.testing.assert_array_equal(np.sort(np.concatenate(folds)), np.arange(100))
    for fold in folds:
        assert len(fold) == 20 and y[fold].sum() == 2


def test_cross_validate_skips_failing_model():
    rng = np.random.default_rng(2)
    tx = np.column_stack([np.ones(200), rng.normal(size=200)])
    tx = np.column_stack([tx, tx[:, 1]])  # singular for least squares
    y = (tx[:, 1] > 0).astype(float)
    models = [
        {"name": "ls", "method": "least_squares"},
        {"name": "ridge", "method": "ridge", "lambda": 1e-3},
    ]
    results = cross_validate(tx, y, build_k_indices(y, 4, 0), models, verbose=False)
    # least squares fails on the singular matrix, ridge still works
    assert "ls" not in results
    assert results["ridge"]["f1"] == pytest.approx(1.0)
