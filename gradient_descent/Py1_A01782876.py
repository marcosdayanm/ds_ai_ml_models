import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from typing import List


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
                except (ValueError, UnicodeDecodeError, Exception):
                    continue
    return X, y


def __print_points_for_desmos(X: List[float], y: List[float]) -> None:
    for i, j in zip(X, y):
        print(f"({i}, {j})")


def __h0(t0: float, t1: float, x: float) -> float:
    return t0 + t1 * x


def gradienteDescendente(X: List[float], y: List[float], theta: List[float] = [0.0, 0.0], alpha: float = 0.01, iteraciones: int = 1500) -> List[float]:
    print("Initial cost:", calculaCosto(X, y, theta))

    m = len(X)

    for _ in range(iteraciones):
        theta0 = theta[0] - alpha * (1/m) * sum([__h0(theta[0], theta[1], X[i]) - y[i] for i in range(m)])  # As in the final formula of slide number 32 in supervised learning linear regression
        theta1 = theta[1] - alpha * (1/m) * sum([(__h0(theta[0], theta[1], X[i]) - y[i]) * X[i] for i in range(m)])
        theta = [theta0, theta1]

    print("Final cost:", calculaCosto(X, y, theta))
    return theta


def calculaCosto(X: List[float], y: List[float], theta: List[float]) -> float:
    """
    This function calculates the cost for linear regression using the mean squared error function (J)
    Used ChatGPT5 for the numpy methods usage
    """
    m = len(y)
    X_mat: NDArray = np.c_[np.ones(m), X] # adding a 1's column on the left of the X vector to turn it into a matrix
    h: NDArray = X_mat @ theta # The @ operator in numpy equals a matrix multiplication (dot product)
    return (1/m) * np.sum((h - y) ** 2)


def graficaDatos(X: List[float], y: List[float], theta: List[float]) -> None:
    plt.scatter(X, y, color='blue', label='Food carts')
    plt.plot(X, [__h0(theta[0], theta[1], x) for x in X], color='red', label='Linear regression by Gradient Descent')
    plt.xlabel('Population')
    plt.ylabel('Earnings')
    plt.title('Linear Regression')
    plt.legend()
    plt.show()


def __calculaCostoForLoop(X: List[float], y: List[float], theta: List[float]) -> float:
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

    # graficaDatos(X, y, [0.0, 0.0])  # Initial plot with theta = [0, 0]

    # __print_points_for_desmos(X, y)
    trained_theta = gradienteDescendente(X, y)

    graficaDatos(X, y, trained_theta)  # Plot with the trained theta


if __name__ == '__main__':
    main()