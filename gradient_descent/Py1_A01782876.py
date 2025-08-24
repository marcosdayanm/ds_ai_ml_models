import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from typing import List


"""
Wrapper functions for grading
"""
def gradienteDescendente(X: List[float], y: List[float], theta: List[float] = [0.0, 0.0], alpha: float = 0.01, iteraciones: int = 1500) -> List[float]:
    print("Initial cost:", calculaCosto(X, y, theta))
    theta = linear_gradient_descent(X, y, theta, alpha, iteraciones)
    print("Final cost:", calculaCosto(X, y, theta))
    return theta


def calculaCosto(X: List[float], y: List[float], theta: List[float]) -> float:
    return calculate_error(X, y, theta)


def graficaDatos(X: List[float], y: List[float], theta: List[float]) -> None:
    __data_graph(X, y, theta)


"""
Helper functions for gradient descent algorithm
"""
def __read_dataset_from_txt(fileroute):
    X = []
    y = []
    with open(fileroute, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                try:
                    X.append(float(parts[0]))
                    y.append(float(parts[1]))
                except (ValueError, UnicodeDecodeError, Exception): # used GitHub copilot completions to determine the possible exception types
                    continue
    return X, y


def __print_points_for_desmos(X: List[float], y: List[float]) -> None:
    for i, j in zip(X, y):
        print(f"({i}, {j})")


def __testCases(theta: List[float]) -> None:
    print(f"Prediction f(3.5): {__linear_estimate(theta[0], theta[1], 3.5)}")
    print(f"Prediction f(7): {__linear_estimate(theta[0], theta[1], 7)}")


def __data_graph(X: List[float], y: List[float], theta: List[float]) -> None:
    """
    Used GitHub Copilot code completions to create this scatter plot with the trained theta regression line
    """
    plt.scatter(X, y, color='blue', label='Food carts')
    plt.plot(X, [__linear_estimate(theta[0], theta[1], x) for x in X], color='red', label='Linear regression by Gradient Descent')
    plt.xlabel('Population')
    plt.ylabel('Earnings')
    plt.title('Linear Regression')
    plt.legend()
    plt.show()


def __linear_estimate(t0: float, t1: float, x: float) -> float:
    return t0 + t1 * x


"""
Linear gradient descent functions
"""
def linear_gradient_descent(X: List[float], y: List[float], theta: List[float] = [0.0, 0.0], alpha: float = 0.01, learning_rate: int = 1500) -> List[float]:
    m = len(X)
    for _ in range(learning_rate):
        theta0 = theta[0] - alpha * (1/m) * sum([__linear_estimate(theta[0], theta[1], X[i]) - y[i] for i in range(m)])  # As in the final algorithm of slide number 32 in supervised learning linear regression
        theta1 = theta[1] - alpha * (1/m) * sum([(__linear_estimate(theta[0], theta[1], X[i]) - y[i]) * X[i] for i in range(m)])
        theta = [theta0, theta1]
    return theta


def calculate_error(X: List[float], y: List[float], theta: List[float]) -> float:
    """
    This function is equal to doing the operations with matrixes, but instead is using a for loop

    Operation: J = (1/m)*(X*theta - y)**2
    """
    m = len(X)
    J = 0.0
    for i in range(m):
        h = theta[0] + theta[1] * X[i]
        J += (h - y[i]) ** 2 # J = (h-y)**2

    return J/(m)


def main():
    X, y = __read_dataset_from_txt("ex1data1.txt")

    trained_theta = linear_gradient_descent(X, y)

    __testCases(trained_theta)

    __data_graph(X, y, trained_theta)  # Plot with the trained theta


if __name__ == '__main__':
    main()