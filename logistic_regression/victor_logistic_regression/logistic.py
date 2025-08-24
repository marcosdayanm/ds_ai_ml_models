from math import e as euler, log
from typing import List, Tuple


def read_dataset_csv(fileroute: str) -> Tuple[List[List[float]], List[float]]:
    with open(fileroute, "r") as f:
        data = f.readlines()
        X: List[List[float]] = []
        y: List[float] = []
        for l in data:
            line = l.strip().split(",")
            X.append([float(line[0]), float(line[1])])
            y.append(float(line[2]))

    return X, y



def linear_estimate_2v(t0: float, t1: float, x1: float, t2: float, x2: float) -> float:
    return t0 + t1 * x1 + t2 * x2

def sigmoid(z: float) -> float:
    return 1 / (1 + pow(euler, -z))



def logistic_regression(X: List[List[float]], y: List[float], theta: List[float] = [0.0, 0.0, 0.0], alpha: float = 0.001, learning_rate: int = 100000) -> Tuple[List[float], List[float]]:
    m = len(X)
    J = []
    for _ in range(learning_rate):

        t0sum, t1sum, t2sum = 0, 0, 0

        for i in range(m):
            z = linear_estimate_2v(theta[0], theta[1], X[i][0], theta[2], X[i][1])
            h0 = sigmoid(z)
            t0sum += (h0 - y[i])
            t1sum += (h0 - y[i]) * X[i][0]
            t2sum += (h0 - y[i]) * X[i][1]

        theta[0] -= alpha * (1/m) * t0sum
        theta[1] -= alpha * (1/m) * t1sum
        theta[2] -= alpha * (1/m) * t2sum

        J.append(calculate_error(X, y, theta))

    return theta, J



def calculate_error(X: List[List[float]], y: List[float], theta: List[float]) -> float:
    m = len(X)
    J = 0.0
    for i in range(m):
        h = sigmoid(linear_estimate_2v(theta[0], theta[1], X[i][0], theta[2], X[i][1]))
        # Clamp h to avoid math domain errors
        h = max(min(h, 1 - 1e-15), 1e-15) # made with ChatGPT 4o to avoid math domain errors in log function
        J += -y[i] * log(h) - (1 - y[i]) * log(1 - h)
    return J / m


def plot_error(J: List[float]):
    import matplotlib.pyplot as plt

    plt.plot(J)
    plt.xlabel("Iteration")
    plt.ylabel("Error")
    plt.title("Error over iterations")
    plt.show()


def test_model(x1, x2, y, theta):
    z = linear_estimate_2v(theta[0], theta[1], x1, theta[2], x2)
    h = sigmoid(z)
    prediction = 1 if h >= 0.5 else 0
    print(f"Test data: ({x1}, {x2}), Actual: {y}, Predicted: {prediction}")


def main():
    # X, y = read_dataset_csv("ex2data1.csv")
    X, y = read_dataset_csv("training.csv")
    error = calculate_error(X, y, [0.0, 0.0, 0.0])
    print("Initial error:", error)

    theta, J = logistic_regression(X, y)
    print("Trained parameters:", theta)
    error = calculate_error(X, y, theta)
    print("Training error:", error)

    Xtest, ytest = read_dataset_csv("test.csv")
    for i in range(len(Xtest)):
        test_model(Xtest[i][0], Xtest[i][1], ytest[i], theta)

    test_model(45, 85, 1, theta)  # Example test case

    plot_error(J)


if __name__ == '__main__':
    main()