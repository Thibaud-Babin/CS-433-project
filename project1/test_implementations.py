"""Tests for the linear regression methods in implementations.py.

Run from the project1 folder with:  python -m pytest test_implementations.py -v

The first group reproduces the expected values from the public grading tests
(grading_tests/test_project1_public.py). The second group checks mathematical
properties on a larger random problem.
"""

import numpy as np
import pytest

import implementations as impl

# Same tolerances, step size and iteration count as the grading tests
RTOL = 1e-4
ATOL = 1e-8
MAX_ITERS = 2
GAMMA = 0.1


@pytest.fixture()
def y():
    return np.array([0.1, 0.3, 0.5])


@pytest.fixture()
def tx():
    return np.array([[2.3, 3.2], [1.0, 0.1], [1.4, 2.3]])


@pytest.fixture()
def initial_w():
    return np.array([0.5, 1.0])


@pytest.fixture()
def random_problem():
    """Well-conditioned linear problem with N=200 samples and D=5 features."""
    rng = np.random.default_rng(0)
    tx = np.c_[np.ones(200), rng.normal(size=(200, 4))]
    true_w = rng.normal(size=5)
    y = tx @ true_w + 0.1 * rng.normal(size=200)
    return y, tx


def check_output(w, loss, expected_w, expected_loss):
    """Check values and the required output format (1D w, scalar loss)."""
    np.testing.assert_allclose(w, expected_w, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(loss, expected_loss, rtol=RTOL, atol=ATOL)
    assert np.ndim(loss) == 0
    assert w.shape == np.shape(expected_w)


# ---------------------------------------------------------------------------
# Expected values from the public grading tests
# ---------------------------------------------------------------------------


def test_compute_mse(y, tx):
    w = np.array([0.5, 1.0])
    e = y - tx @ w
    assert np.isclose(impl.compute_mse(y, tx, w), 0.5 * np.mean(e**2))


def test_mean_squared_error_gd_0_step(y, tx):
    w0 = np.array([0.413044, 0.875757])
    w, loss = impl.mean_squared_error_gd(y, tx, w0, 0, GAMMA)
    check_output(w, loss, w0, 2.959836)


def test_mean_squared_error_gd(y, tx, initial_w):
    w, loss = impl.mean_squared_error_gd(y, tx, initial_w, MAX_ITERS, GAMMA)
    check_output(w, loss, [-0.050586, 0.203718], 0.051534)


def test_mean_squared_error_sgd(y, tx, initial_w):
    # A single sample makes SGD deterministic
    w, loss = impl.mean_squared_error_sgd(y[:1], tx[:1], initial_w, MAX_ITERS, GAMMA)
    check_output(w, loss, [0.063058, 0.39208], 0.844595)


def test_mean_squared_error_sgd_0_step(y, tx, initial_w):
    w, loss = impl.mean_squared_error_sgd(y, tx, initial_w, 0, GAMMA)
    check_output(w, loss, initial_w, impl.compute_mse(y, tx, initial_w))


def test_least_squares(y, tx):
    w, loss = impl.least_squares(y, tx)
    check_output(w, loss, [0.218786, -0.053837], 0.026942)


def test_ridge_regression_lambda0(y, tx):
    w, loss = impl.ridge_regression(y, tx, 0.0)
    check_output(w, loss, [0.218786, -0.053837], 0.026942)


def test_ridge_regression_lambda1(y, tx):
    w, loss = impl.ridge_regression(y, tx, 1.0)
    check_output(w, loss, [0.054303, 0.042713], 0.03175)


# ---------------------------------------------------------------------------
# Mathematical properties on a larger problem
# ---------------------------------------------------------------------------


def test_least_squares_matches_lstsq(random_problem):
    # lstsq is only forbidden inside implementations.py, not as a reference
    y, tx = random_problem
    w, loss = impl.least_squares(y, tx)
    w_ref = np.linalg.lstsq(tx, y, rcond=None)[0]
    np.testing.assert_allclose(w, w_ref, rtol=1e-8, atol=1e-10)
    assert np.isclose(loss, impl.compute_mse(y, tx, w_ref))


def test_least_squares_gradient_is_zero(random_problem):
    y, tx = random_problem
    w, _ = impl.least_squares(y, tx)
    np.testing.assert_allclose(impl.compute_mse_gradient(y, tx, w), 0, atol=1e-10)


def test_gd_converges_to_least_squares(random_problem):
    y, tx = random_problem
    w_ls, loss_ls = impl.least_squares(y, tx)
    w, loss = impl.mean_squared_error_gd(y, tx, np.zeros(5), 1000, 0.5)
    np.testing.assert_allclose(w, w_ls, atol=1e-6)
    assert np.isclose(loss, loss_ls)


def test_gd_loss_decreases(random_problem):
    y, tx = random_problem
    w0 = np.zeros(5)
    losses = [impl.mean_squared_error_gd(y, tx, w0, k, 0.1)[1] for k in range(10)]
    assert all(b < a for a, b in zip(losses, losses[1:]))


def test_sgd_approaches_least_squares(random_problem):
    np.random.seed(0)
    y, tx = random_problem
    _, loss_ls = impl.least_squares(y, tx)
    w0 = np.zeros(5)
    _, loss_start = impl.mean_squared_error_sgd(y, tx, w0, 0, 0.01)
    _, loss = impl.mean_squared_error_sgd(y, tx, w0, 5000, 0.01)
    # SGD with a constant step only gets close to the optimum
    assert loss < loss_start
    assert loss < 1.5 * loss_ls


def test_sgd_is_reproducible_with_seed(random_problem):
    y, tx = random_problem
    np.random.seed(42)
    w1, _ = impl.mean_squared_error_sgd(y, tx, np.zeros(5), 50, 0.01)
    np.random.seed(42)
    w2, _ = impl.mean_squared_error_sgd(y, tx, np.zeros(5), 50, 0.01)
    np.testing.assert_array_equal(w1, w2)


def test_ridge_minimizes_penalized_objective(random_problem):
    y, tx = random_problem
    lambda_ = 0.1
    w, _ = impl.ridge_regression(y, tx, lambda_)
    # Gradient of MSE(w) + lambda * ||w||^2 must vanish at the optimum
    grad = impl.compute_mse_gradient(y, tx, w) + 2 * lambda_ * w
    np.testing.assert_allclose(grad, 0, atol=1e-10)


def test_ridge_loss_excludes_penalty(random_problem):
    y, tx = random_problem
    w, loss = impl.ridge_regression(y, tx, 0.5)
    assert np.isclose(loss, impl.compute_mse(y, tx, w))


def test_ridge_shrinks_weights(random_problem):
    y, tx = random_problem
    norms = [
        np.linalg.norm(impl.ridge_regression(y, tx, lam)[0])
        for lam in [0.0, 0.01, 0.1, 1.0, 10.0]
    ]
    assert all(b < a for a, b in zip(norms, norms[1:]))


@pytest.mark.parametrize(
    "method, extra_args",
    [
        (impl.mean_squared_error_gd, (np.zeros(5), 10, 0.1)),
        (impl.mean_squared_error_sgd, (np.zeros(5), 10, 0.1)),
        (impl.least_squares, ()),
        (impl.ridge_regression, (0.1,)),
    ],
)
def test_inputs_not_modified(random_problem, method, extra_args):
    y, tx = random_problem
    copies = [a.copy() for a in (y, tx) + extra_args if isinstance(a, np.ndarray)]
    method(y, tx, *extra_args)
    originals = [a for a in (y, tx) + extra_args if isinstance(a, np.ndarray)]
    for before, after in zip(copies, originals):
        np.testing.assert_array_equal(before, after)
