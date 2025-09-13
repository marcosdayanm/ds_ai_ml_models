

from typing import List, Tuple
import numpy as np


class ALM:
    def __init__(self, input_vectors: List[List[float]], output_vectors: List[List[float]]):
        self.W = None
        self.bias = None
        self._train(input_vectors, output_vectors)

    def _train(self, input_vectors: List[List[float]], output_vectors: List[List[float]]):
        W = np.zeros((len(output_vectors[0]), len(input_vectors[0])))

        # create matrix
        for j in range(len(W)):
            for i in range(len(W[0])):
                for a, b in zip(input_vectors, output_vectors):
                    W[j][i] += self._vector_individual_sum(a[i], b[j])

        bias = self._compute_bias(W)
        self.W = W
        self.bias = bias.reshape(-1, 1)

    def predict(self, input_predict: np.ndarray) -> np.ndarray:
        if self.W is None or self.bias is None:
            raise ValueError("Model is not trained yet.")
        result = self.W @ input_predict + self.bias
        return self._asymetric_activation(result)
    
    def _vector_individual_sum(self, a: float, b: float) -> float:
        return (2*a - 1) * (2*b - 1)

    def _compute_bias(self, W: np.ndarray) -> np.ndarray:
        return np.array([(-1/2)*sum(row) for row in W])

    def _asymetric_activation(self, x: np.ndarray) -> np.ndarray:
        for i in range(len(x)):
            if x[i] <= 0:
                x[i] = 0
            else:
                x[i] = 1
        return x

    def print_matrix(self) -> None:
        if self.W is None:
            raise ValueError("Weight matrix is not initialized.")
        for row in self.W:
            print(row)





if __name__ == "__main__":
    # input_vectors = [
    #     [0.0, 1.0, 0.0],
    #     [0.0, 0.0, 0.0],
    #     [1.0, 1.0, 1.0],
    # ]

    # output_vectors = [
    #     [1.0, 0.0, 1.0],
    #     [0.0, 0.0, 0.0],
    #     [1.0, 1.0, 1.0],
    # ]

    # test_input = np.array([0.0, 1.0, 0.0]).reshape(-1, 1)

    # input_vectors = [
    #     [0.0, 1.0, 0.0, 1.0],
    #     [0.0, 0.0, 0.0, 0.0],
    #     [1.0, 1.0, 1.0, 1.0],
    # ]

    # output_vectors = [
    #     [1.0, 0.0, 1.0],
    #     [0.0, 0.0, 0.0],
    #     [1.0, 1.0, 1.0],
    # ]

    # test_input = np.array([0.0, 1.0, 0.0, 1.0]).reshape(-1, 1)

    # input_vectors = [
    #     [0.0, 1.0, 0.0],
    #     [0.0, 0.0, 0.0],
    #     [1.0, 1.0, 1.0],
    # ]

    # output_vectors = [
    #     [1.0, 0.0, 1.0],
    #     [0.0, 0.0, 0.0],
    #     [1.0, 1.0, 1.0],
    # ]

    input_vectors = [
        [1.0, 1.0, 1.0, 1.0],
        [-1.0,-1.0,-1.0,-1.0],
        [-1.0, 1.0, -1.0, 1.0],
    ]

    output_vectors = [
        [1.0, 1.0, 1.0, 1.0],
        [-1.0,-1.0,-1.0,-1.0],
        [-1.0, 1.0, -1.0, 1.0],
    ]

    test_input = np.array([-1, -1, -1, -1]).reshape(-1, 1)

    alm = ALM(input_vectors, output_vectors)

    alm.print_matrix()

    # alm.print_matrix()

    print(alm.predict(test_input))