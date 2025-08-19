from typing import List, Tuple

def step_function(x: float):
    return 1 if x >= 1 else 0


def perceptron_classification(X: List[List[float]], y: List[int], learning_rate: float = 0.01, epochs: int = 1500) -> Tuple[List[float], int]:

    W = [0.1] * len(X[0])  # Initializing weights to zero
    streak = 0
    e = 0
    while e < epochs and streak < len(X):
        weighted_sum = 0
        for i in range(len(X[streak])):
            weighted_sum += X[streak][i] * W[i]
        prediction = step_function(weighted_sum)
        error = y[streak] - prediction
        if error != 0:
            W = __adjust_weights(W, X[streak], weighted_sum, learning_rate)
            streak = 0
        else:
            streak += 1

        e += 1

    return W, e


def __adjust_weights(W: List[float], xi: List[float], e: float, learning_rate: float = 0.01) -> List[float]:
    for i in range(len(xi)):
        W[i] += (learning_rate * e * xi[i])
    return W


def read_data_from_csv(fileroute: str) -> Tuple[List[List[float]], List[int]]:
    with open(fileroute, "r") as f:
        lines = f.readlines()[1:]  # Skip header line
    X = []
    y = []
    for line in lines:
        values = line.strip().split(",")
        X.append([float(v) for v in values[:-1]])
        y.append(int(values[-1]))
    return X, y


def classify(x: List[float], y: int, W: List[float]) -> int:
    weighted_sum = sum(xi * wi for xi, wi in zip(x, W))
    return step_function(weighted_sum)


def main():
    X, y = read_data_from_csv("training_data.csv")
    W, e = perceptron_classification(X, y)
    print("Weights:", W)
    print("Epochs:", e)

    X_test, y_test = read_data_from_csv("test_data.csv")
    for i in range(len(X_test)):
        result = classify(X_test[i], y_test[i], W)
        print(f"Test {i+1}. variables:{X_test[i]} y:{y_test[i]} got:{result}")


if __name__ == '__main__':
    main()