import numpy as np


def compute_mse(y, tx, w):
    """Mean squared error loss (with the 1/2 factor)."""
    e = y - tx.dot(w)
    return e.dot(e) / (2 * len(e))


def compute_mse_gradient(y, tx, w):
    """Gradient of the MSE loss at w."""
    e = y - tx.dot(w)
    return -tx.T.dot(e) / len(e)


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent.

    Returns the last weights w and the MSE loss at w.
    """
    w = initial_w
    for i in range(max_iters):
        grad = compute_mse_gradient(y, tx, w)
        w = w - gamma * grad
    loss = compute_mse(y, tx, w)
    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent (batch size 1).

    Returns the last weights w and the MSE loss at w (on the full data).
    """
    w = initial_w
    for i in range(max_iters):
        n = np.random.randint(len(y))
        y_n = y[n : n + 1]
        tx_n = tx[n : n + 1]
        grad = compute_mse_gradient(y_n, tx_n, w)
        w = w - gamma * grad
    loss = compute_mse(y, tx, w)
    return w, loss


def least_squares(y, tx):
    """Least squares regression using the normal equations.

    Returns the optimal weights w and the MSE loss at w.
    """
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_mse(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """Ridge regression using the normal equations.

    Returns the optimal weights w and the MSE loss at w (without the penalty).
    """
    n = tx.shape[0]
    d = tx.shape[1]
    a = tx.T.dot(tx) + 2 * n * lambda_ * np.identity(d)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_mse(y, tx, w)
    return w, loss


def compute_utilities(tx, w):
    """Score U = tx @ w of each sample."""
    return tx.dot(w)


def compute_probabilities(tx, w):
    """Sigmoid of the scores, P(y = 1 | x)."""
    U = compute_utilities(tx, w)
    U = np.clip(U, -30, 30)
    return 1 / (1 + np.exp(-U))


def log_loss(y, tx, w):
    """Logistic loss (mean negative log-likelihood)."""
    U = compute_utilities(tx, w)
    return np.mean(np.logaddexp(0, U) - y * U)


def log_loss_gradient(y, tx, w):
    """Gradient of the logistic loss at w."""
    N = len(y)
    P = compute_probabilities(tx, w)
    return tx.T.dot(P - y) / N


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent (y in {0, 1}).

    Returns the last weights w and the logistic loss at w.
    """
    w = initial_w
    for i in range(max_iters):
        grad = log_loss_gradient(y, tx, w)
        w = w - gamma * grad
    loss = log_loss(y, tx, w)
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent (y in {0, 1}).

    Returns the last weights w and the logistic loss at w (without the
    penalty lambda_ * ||w||^2).
    """
    w = initial_w
    for i in range(max_iters):
        grad = log_loss_gradient(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * grad
    loss = log_loss(y, tx, w)
    return w, loss
