import numpy as np
import pytest

import implementations as impl

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
    rng = np.random.default_rng(0)
    tx = np.c_[np.ones(200), rng.normal(size=(200, 4))]
    true_w = rng.normal(size=5)
    y = tx @ true_w + 0.1 * rng.normal(size=200)
    return y, tx


def check_output(w, loss, expected_w, expected_loss):
    np.testing.assert_allclose(w, expected_w, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(loss, expected_loss, rtol=RTOL, atol=ATOL)
    assert np.ndim(loss) == 0
    assert w.shape == np.shape(expected_w)


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


def test_least_squares_matches_lstsq(random_problem):
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
    losses = []
    for k in range(10):
        w, loss = impl.mean_squared_error_gd(y, tx, w0, k, 0.1)
        losses.append(loss)
    for k in range(1, 10):
        assert losses[k] < losses[k - 1]


def test_sgd_approaches_least_squares(random_problem):
    np.random.seed(0)
    y, tx = random_problem
    _, loss_ls = impl.least_squares(y, tx)
    w0 = np.zeros(5)
    _, loss_start = impl.mean_squared_error_sgd(y, tx, w0, 0, 0.01)
    _, loss = impl.mean_squared_error_sgd(y, tx, w0, 5000, 0.01)
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
    grad = impl.compute_mse_gradient(y, tx, w) + 2 * lambda_ * w
    np.testing.assert_allclose(grad, 0, atol=1e-10)


def test_ridge_loss_excludes_penalty(random_problem):
    y, tx = random_problem
    w, loss = impl.ridge_regression(y, tx, 0.5)
    assert np.isclose(loss, impl.compute_mse(y, tx, w))


def test_ridge_shrinks_weights(random_problem):
    y, tx = random_problem
    norms = []
    for lam in [0.0, 0.01, 0.1, 1.0, 10.0]:
        w, loss = impl.ridge_regression(y, tx, lam)
        norms.append(np.linalg.norm(w))
    for k in range(1, len(norms)):
        assert norms[k] < norms[k - 1]


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
    arrays = [y, tx]
    for a in extra_args:
        if isinstance(a, np.ndarray):
            arrays.append(a)
    copies = []
    for a in arrays:
        copies.append(a.copy())
    method(y, tx, *extra_args)
    for i in range(len(arrays)):
        np.testing.assert_array_equal(copies[i], arrays[i])


REQUIRED_FUNCTIONS = [
    "mean_squared_error_gd",
    "mean_squared_error_sgd",
    "least_squares",
    "ridge_regression",
    "logistic_regression",
    "reg_logistic_regression",
]

needs_reg_logistic = pytest.mark.skipif(
    not hasattr(impl, "reg_logistic_regression"),
    reason="reg_logistic_regression is not implemented yet",
)


@pytest.fixture()
def y_binary(y):
    return (y > 0.2) * 1.0


@pytest.fixture()
def random_classification():
    rng = np.random.default_rng(1)
    tx = np.c_[np.ones(300), rng.normal(size=(300, 3))]
    true_w = np.array([0.3, 1.0, -2.0, 0.5])
    p = 1 / (1 + np.exp(-tx @ true_w))
    y = (rng.random(300) < p) * 1.0
    return y, tx


def reference_log_loss(y, tx, w):
    z = tx @ w
    return np.mean(np.logaddexp(0, z) - y * z)


@pytest.mark.parametrize("function_name", REQUIRED_FUNCTIONS)
def test_required_function_exists_with_docstring(function_name):
    assert hasattr(impl, function_name), f"{function_name} is missing"
    assert getattr(impl, function_name).__doc__, f"{function_name} has no docstring"


def test_compute_probabilities():
    tx = np.array([[0.0], [1.0], [-1.0]])
    p = impl.compute_probabilities(tx, np.array([2.0]))
    np.testing.assert_allclose(p, [0.5, 1 / (1 + np.exp(-2)), 1 / (1 + np.exp(2))])


def test_log_loss_at_zero_weights(random_classification):
    y, tx = random_classification
    assert np.isclose(impl.log_loss(y, tx, np.zeros(4)), np.log(2))


def test_log_loss_matches_reference(random_classification):
    y, tx = random_classification
    w = np.array([0.1, -0.4, 0.7, 0.2])
    assert np.isclose(impl.log_loss(y, tx, w), reference_log_loss(y, tx, w))


def test_log_loss_gradient_matches_finite_differences(random_classification):
    y, tx = random_classification
    w = np.array([0.1, -0.4, 0.7, 0.2])
    eps = 1e-6
    numerical = np.array(
        [
            (
                reference_log_loss(y, tx, w + eps * e)
                - reference_log_loss(y, tx, w - eps * e)
            )
            / (2 * eps)
            for e in np.eye(4)
        ]
    )
    np.testing.assert_allclose(impl.log_loss_gradient(y, tx, w), numerical, atol=1e-7)


def test_logistic_regression_0_step(y_binary, tx):
    w0 = np.array([0.463156, 0.939874])
    w, loss = impl.logistic_regression(y_binary, tx, w0, 0, GAMMA)
    check_output(w, loss, w0, 1.533694)


def test_logistic_regression(y_binary, tx, initial_w):
    w, loss = impl.logistic_regression(y_binary, tx, initial_w, MAX_ITERS, GAMMA)
    check_output(w, loss, [0.378561, 0.801131], 1.348358)


def test_logistic_regression_loss_decreases(random_classification):
    y, tx = random_classification
    w0 = np.zeros(4)
    losses = []
    for k in range(10):
        w, loss = impl.logistic_regression(y, tx, w0, k, 0.5)
        losses.append(loss)
    for k in range(1, 10):
        assert losses[k] < losses[k - 1]


def test_logistic_regression_converges(random_classification):
    y, tx = random_classification
    w, loss = impl.logistic_regression(y, tx, np.zeros(4), 5000, 1.0)
    np.testing.assert_allclose(impl.log_loss_gradient(y, tx, w), 0, atol=1e-6)
    assert np.isclose(loss, reference_log_loss(y, tx, w))


def test_log_loss_is_finite_for_large_scores():
    tx = np.array([[1000.0], [-1000.0]])
    y = np.array([0.0, 1.0])
    w = np.array([1.0])
    with np.errstate(all="ignore"):
        loss = impl.log_loss(y, tx, w)
    assert np.isfinite(loss)
    assert np.isclose(loss, reference_log_loss(y, tx, w))


@needs_reg_logistic
def test_reg_logistic_regression_0_step(y_binary, tx):
    w0 = np.array([0.409111, 0.843996])
    w, loss = impl.reg_logistic_regression(y_binary, tx, 1.0, w0, 0, GAMMA)
    check_output(w, loss, w0, 1.407327)


@needs_reg_logistic
def test_reg_logistic_regression(y_binary, tx, initial_w):
    w, loss = impl.reg_logistic_regression(
        y_binary, tx, 1.0, initial_w, MAX_ITERS, GAMMA
    )
    check_output(w, loss, [0.216062, 0.467747], 0.972165)


@needs_reg_logistic
def test_reg_logistic_regression_lambda0_equals_logistic(random_classification):
    y, tx = random_classification
    w_reg, loss_reg = impl.reg_logistic_regression(y, tx, 0.0, np.zeros(4), 50, 0.5)
    w, loss = impl.logistic_regression(y, tx, np.zeros(4), 50, 0.5)
    np.testing.assert_allclose(w_reg, w)
    assert np.isclose(loss_reg, loss)


@needs_reg_logistic
def test_reg_logistic_regression_optimum_and_loss(random_classification):
    y, tx = random_classification
    lambda_ = 0.1
    w, loss = impl.reg_logistic_regression(y, tx, lambda_, np.zeros(4), 5000, 0.5)
    grad = impl.log_loss_gradient(y, tx, w) + 2 * lambda_ * w
    np.testing.assert_allclose(grad, 0, atol=1e-6)
    assert np.isclose(loss, reference_log_loss(y, tx, w))
