from typing import List

import numpy as np


class BAM:
    def __init__(self, input: List[np.ndarray], output: List[np.ndarray]):
        if len(input) != len(output) or input[0].size != output[0].size:
            raise ValueError("Input and output must have the same length.")
        self.input = input
        self.output = output
        self.weights = self._train()

    def _train(self):
        # Initialize weights to zero
        weights = np.zeros((self.input[0].size, self.output[0].size))
        for x, y in zip(self.input, self.output):
            weights += np.outer(x, y)
            # print("WEIGHTS:", weights)
        return weights

    def predict(self, x: np.ndarray) -> np.ndarray:
        if x.size != self.input[0].size:
            raise ValueError("Input size must match training data.")
        return np.sign(self.weights @ x)




if __name__ == "__main__":
    # Example usage
    # input_patterns = [
    #     np.array([-1, 1, -1]), 
    #     np.array([-1, -1, -1]),
    #     np.array([1, 1, 1])
    # ]

    # output_patterns = [
    #     np.array([1, -1, 1]),
    #     np.array([-1, -1, -1]),
    #     np.array([1, 1, 1])
    # ]

    input_patterns = [
        np.array([1, 1, 1, 1]),
        np.array([-1, -1, -1, -1]),
        np.array([-1, 1, -1, 1]),
    ]

    output_patterns = [
        np.array([1, 1, 1, 1]),
        np.array([-1, -1, -1, -1]),
        np.array([-1, 1, -1, 1]),
    ]

    bam = BAM(input_patterns, output_patterns)

    print(bam.weights)

    test_input = np.array([-1, 1, -1, 1]).reshape(-1, 1)
    prediction = bam.predict(test_input)
    print(prediction)