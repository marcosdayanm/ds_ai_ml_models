import sys
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt


def read_dataset_csv(fileroute: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(fileroute, delimiter=",")
    X = data[:, 0:2]
    y = data[:, 2].astype(int) #GitHub copilot gave me this parsing method to parse the numbers as integers instead of floats
    return X, y

def mapeoCaracterísticas(X, degree = 6) -> np.ndarray:
    """
    This function does a two variable feature mapping to a polynomial with a chosen degree, it defaults to 6

    I found the pattern of the polynomial form using ChatGPT5
    """
    # Example [1, x1, x2, x1^2, x1 x2, x2^2, ..., x1 x2^5, x2^6]
    features = []
    for x1, x2 in X:
        row = []
        for i in range(degree + 1):
            for j in range(i + 1):
                row.append((x1 ** (i - j)) * (x2 ** j))
        features.append(row)
    return np.array(features)


def sigmoidal(z: np.ndarray):
    """Sigmoid acgtivation function"""
    return 1.0/(1.0 + np.exp(-z))


def funcionCostoReg(theta: np.ndarray, X: np.ndarray, y: np.ndarray, l: float) -> Tuple[float, np.ndarray]:
    """
    This function computes the cost and gradient for regularized logistic regression.
    """
    m: int = len(y)
    h: np.ndarray = sigmoidal(X @ theta)  # h0, in linear regression it is only the result of X @ theta, but in this case we are passing through an activation function
    J = (-1/m) * (y @ np.log(h) + (1 - y) @ np.log(1 - h)) # logarithm error function
    J += (l/(2*m)) * np.sum(theta[1:]**2) # adding the lambda for regularization (avoiding overfitting, but if the value is too high it can underfit the model)
    grad = (1/m) * (X.T @ (h - y))
    grad[1:] += (l/m) * theta[1:]
    return J, grad


def aprende(theta: np.ndarray, X: np.ndarray, y: np.ndarray, alpha: float, lambd: float, iteraciones: int) -> Tuple[np.ndarray, List[float]]:
    """This function performs the gradient descent, iterating through 'iteraciones' epochs to add theta the new gradient minimized by the learning rate.
    This function performs matrix operations with numpy, being so much efficient.

    """
    J_history: List[float] = []
    for _ in range(iteraciones):
        J, grad = funcionCostoReg(theta, X, y, lambd)
        theta -= alpha * grad
        J_history.append(J)
    return theta, J_history # returning J history to plot error


def predice(theta: np.ndarray, X: np.ndarray):
    return (sigmoidal(X @ theta) >= 0.5).astype(int)

def plot_points(X: np.ndarray, y: np.ndarray):
    plt.figure()
    pos = y == 1
    neg = y == 0
    plt.scatter(X[pos, 0], X[pos, 1], marker='x', color='k', label='Positivo')
    plt.scatter(X[neg, 0], X[neg, 1], marker='o', color='gray', label='Negativo')
    plt.xlabel('Variable 1')
    plt.ylabel('Variable 2')
    plt.legend(loc='best')
    plt.show()


def graficaDatos(X: np.ndarray, y: np.ndarray, theta: np.ndarray, lambd: float | None = None, is_plot_together = True):
    """
    Plot the decision boundary and data points.

    Used GitHub Copilot agent mode to assist with the scatter settings, and the z contour plot.
    """
    pos = y == 1
    neg = y == 0
    plt.figure()
    plt.scatter(X[pos, 0], X[pos, 1], marker='x', label='Accepted')
    plt.scatter(X[neg, 0], X[neg, 1], marker='o', label='Rejected')
    plt.xlabel('Variable 1')
    plt.ylabel('Variable 2')

    # decision boundary via z=0 contour
    u = np.linspace(X[:,0].min()-1, X[:,0].max()+1, 200)
    v = np.linspace(X[:,1].min()-1, X[:,1].max()+1, 200)
    uu, vv = np.meshgrid(u, v)
    grid = np.c_[uu.ravel(), vv.ravel()]
    mapped = mapeoCaracterísticas(grid)

    z = mapped @ theta
    z = z.reshape(uu.shape)
    plt.contour(uu, vv, z, levels=[0], linewidths=2)

    if lambd:
        plt.title(f'Decision boundary (lambda={lambd})')

    if not is_plot_together:
        plt.show()



def plot_error(J_hist, lambd=None):
    """Plot cost history J_hist"""
    plt.figure()
    plt.plot(J_hist)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    if lambd is None:
        plt.title('error')
    else:
        plt.title(f'error (lambda={lambd})')


def full_test_case(X_original: np.ndarray, X: np.ndarray, y: np.ndarray, theta: np.ndarray, alpha: float, lambd: float, epochs: int, is_plot_together = True, is_plot_error = False):
    theta, J_hist = aprende(theta, X, y, alpha, lambd, iteraciones=epochs)
    acc: float = (predice(theta, X) == y).mean() * 100.0
    print(f"[lambda={lambd}] final cost={J_hist[-1]:.6f} | train acc={acc:.3f}%")
    graficaDatos(X_original, y, theta, lambd)

    if is_plot_error:
        plot_error(J_hist, lambd)


def test_cases_with_lambda_variation(X_original: np.ndarray, X6: np.ndarray, y: np.ndarray, theta: np.ndarray, alpha: float, lambdas: List[float], epochs: int, is_plot_together = True, is_plot_error = False):
    """Orchestrates different test cases each one with a different lambda value """
    for lambd in lambdas:
        full_test_case(X_original, X6, y, theta, alpha, lambd, epochs, is_plot_together, is_plot_error)

    if is_plot_together:
        plt.show()


def main():
    args = [a.lower() for a in sys.argv[1:]]
    # enable error-plotting when the script is called with the token 'plot_error'
    is_plot_error = 'plot_error' in args

    X, y = read_dataset_csv("ex2data2.csv")

    # plot_points(X, y)

    X6 = mapeoCaracterísticas(X)

    initial_theta = np.zeros(X6.shape[1])
    alpha = 0.3

    J0, _ = funcionCostoReg(initial_theta, X6, y, l=1.0)
    print(f"Initial cost (lambda=1): {J0:.6f}")


    test_case_lambdas = [
        0.0, # Overfitting
        100.0, # Underfitting
        1.0 # Good fit
    ]
    test_cases_with_lambda_variation(X, X6, y, initial_theta.copy(), alpha, lambdas=test_case_lambdas, epochs=4000, is_plot_error=is_plot_error)


if __name__ == "__main__":
    main()
