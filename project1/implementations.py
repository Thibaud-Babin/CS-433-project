"""Implementations of the ML methods required for Project 1.

All vectors are 1D numpy arrays of shape (N,) or (D,).
Every function returns (w, loss): the last weight vector and its loss.
"""

import numpy as np


# Helpers repris depuis les labs
# les descriptions générées par IA pour faire plus clean
# il manque logistic_regression and reg_logistic_regression à faire


def compute_mse(y, tx, w):
    """Compute the mean squared error loss, with the 1/2 factor.

    Args:
        y: numpy array of shape (N,), the targets.
        tx: numpy array of shape (N, D), the features.
        w: numpy array of shape (D,), the model weights.

    Returns:
        mse: scalar, (1 / 2N) * sum((y - tx @ w) ** 2).
    """
    e = y - tx.dot(w)
    return e.dot(e) / (2 * len(e))


def compute_mse_gradient(y, tx, w):
    """Compute the gradient of the MSE loss at w.

    Args:
        y: numpy array of shape (N,), the targets.
        tx: numpy array of shape (N, D), the features.
        w: numpy array of shape (D,), the model weights.

    Returns:
        grad: numpy array of shape (D,), the gradient of the MSE at w.
    """
    e = y - tx.dot(w)
    return -tx.T.dot(e) / len(e)


# Linear regression


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent.

    Args:
        y: numpy array of shape (N,), the targets.
        tx: numpy array of shape (N, D), the features.
        initial_w: numpy array of shape (D,), the initial weights.
        max_iters: int, the number of gradient steps.
        gamma: float, the step size.

    Returns:
        w: numpy array of shape (D,), the last weight vector.
        loss: scalar, the MSE of the last weight vector.
    """
    w = initial_w
    for _ in range(max_iters):
        # Full-batch gradient step
        w = w - gamma * compute_mse_gradient(y, tx, w)
    # Loss is computed on the final w (also works when max_iters == 0)
    return w, compute_mse(y, tx, w)


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent (mini-batch size 1).

    Args:
        y: numpy array of shape (N,), the targets.
        tx: numpy array of shape (N, D), the features.
        initial_w: numpy array of shape (D,), the initial weights.
        max_iters: int, the number of stochastic gradient steps.
        gamma: float, the step size.

    Returns:
        w: numpy array of shape (D,), the last weight vector.
        loss: scalar, the MSE of the last weight vector on the full dataset.
    """
    w = initial_w
    for _ in range(max_iters):
        n = np.random.randint(len(y))
        w = w - gamma * compute_mse_gradient(y[n : n + 1], tx[n : n + 1], w)
    return w, compute_mse(y, tx, w)


def least_squares(y, tx):
    """Least squares regression using the normal equations.

    Solves (tx^T tx) w = tx^T y with np.linalg.solve (more stable than
    explicitly inverting the Gram matrix).

    Args:
        y: numpy array of shape (N,), the targets.
        tx: numpy array of shape (N, D), the features.

    Returns:
        w: numpy array of shape (D,), the optimal weights.
        loss: scalar, the MSE of the optimal weights.
    """
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    return w, compute_mse(y, tx, w)


def ridge_regression(y, tx, lambda_):
    """Ridge regression using the normal equations.

    Solves (tx^T tx + 2N * lambda_ * I) w = tx^T y, which minimizes
    MSE(w) + lambda_ * ||w||^2.

    Args:
        y: numpy array of shape (N,), the targets.
        tx: numpy array of shape (N, D), the features.
        lambda_: float, the regularization parameter.

    Returns:
        w: numpy array of shape (D,), the optimal weights.
        loss: scalar, the MSE of the optimal weights (without penalty term).
    """
    n, d = tx.shape
    a = tx.T.dot(tx) + 2 * n * lambda_ * np.identity(d)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    return w, compute_mse(y, tx, w)

# Logistic regression

def compute_utilities(tx, w):
    # tx contains the features for all observations (N, D)
    # w contains the model parameters (D,)
    # Compute the utility/score U = Xw for each observation
    return tx.dot(w)


def compute_probabilities(tx, w):
    # Compute the utility for each observation
    U = compute_utilities(tx, w)
    
    # Convert the utility into a probability using the logistic function
    # P = 1 / (1 + exp(-U))
    return 1 / (1 + np.exp(-U))


def log_loss(y, tx, w):
    # Compute the predicted probabilities
    P = compute_probabilities(tx, w)
    
    # Number of observations
    N = len(y)
    
    # Compute the average binary cross-entropy (logistic) loss
    # This measures the difference between the true labels y and probabilities P
    return -1/N * np.sum(y * np.log(P) + (1-y) * np.log(1-P))


def log_loss_gradient(y, tx, w):
    # Number of observations
    N = len(y)
    
    # Compute the gradient of the log-loss with respect to w
    # Gradient = (1/N) * X^T * (P - y)
    return 1/N * tx.T.dot(compute_probabilities(tx, w) - y)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    # Initialize the model parameters with the given initial values
    w = initial_w
    
    # Repeat the gradient descent update for the specified number of iterations
    for i in range(max_iters):
        # Compute the gradient of the log-loss
        gradient = log_loss_gradient(y, tx, w)
        
        # Update the parameters in the direction that decreases the loss
        w = w - gamma * gradient

    # Return the optimized parameters and the final loss
    return w, log_loss(y, tx, w)