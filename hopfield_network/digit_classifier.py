from typing import List, Tuple
import numpy as np
import random
from data import training_data


class HopfieldNetwork:
    def __init__(self, patterns: List[List[int]]):
        self.patterns = np.array(patterns)
        self.n_neurons = len(self.patterns[0])  # neuron number (100 for 10*10 matrixes)
        self.n_patterns = len(self.patterns)    # pattern number (10 digits in this case)
        
        self.weights = self._compute_weights()
        
    def _compute_weights(self):
        """
        Computes the weight matrix
        W_ij = (1/N) * sum(c_i^s*c_j^s) for i ≠ j
        W_ij = 0 for i = j
        """
        W = np.zeros((self.n_neurons, self.n_neurons))
        
        for i in range(self.n_neurons):
            for j in range(i+1, self.n_neurons):
                for pattern in self.patterns:
                    W[i][j] += pattern[i] * pattern[j]
                
                W[i][j] /= self.n_neurons
                W[j][i] = W[i][j]  # symmetric

                    
        return W

    def _vector_symmetric_activation(self, v) -> np.ndarray:
        return np.where(v >= 0, 1, -1)

    def _matrix_convergence(self, pattern: List[int], max_iterations = 100) -> Tuple[List[int], int, bool]:
        """Test the network with a given pattern"""
        if len(pattern) != self.n_neurons:
            raise ValueError(f"Input pattern must have {self.n_neurons} elements")

        input_pattern = self._vector_symmetric_activation(np.array(pattern.copy()))

        for i in range(max_iterations):
            output_pattern = self._vector_symmetric_activation(self.weights @ input_pattern)
            if np.array_equal(input_pattern, output_pattern):  # convergence
                return input_pattern.tolist(), i, True
            input_pattern = output_pattern

        return input_pattern.tolist(), max_iterations, False


    def classify_digit(self, pattern, expected_result):
        recovered_pattern, iterations, is_converged = self._matrix_convergence(pattern)
        if not is_converged:
            print(f"Unconverged after {iterations} iterations")

        similarities = []
        for i, stored_pattern in enumerate(self.patterns):
            similarity = np.dot(recovered_pattern, stored_pattern) / self.n_neurons
            similarities.append((i, similarity))

        best_digit: Tuple[int, float] = max(similarities, key=lambda x: x[1]) # (digit, similarity)

        self.visualize_pattern(recovered_pattern, "Recovered pattern")
        self.visualize_pattern(self.patterns[best_digit[0]], "Similar pattern")
        print(f"Predicted digit: {best_digit[0]} (similarity: {best_digit[1]:.2f}), Expected: {expected_result}")

        return best_digit, recovered_pattern, iterations


    def visualize_pattern(self, pattern, title="Pattern"):
        """
        Visualizes pattern in terminal
        """
        print(f"\n{title}:")
        pattern_2d = np.array(pattern).reshape(10, 10)
        
        for row in pattern_2d:
            line = ""
            for val in row:
                line += "██" if val == -1 else "  "
            print(line)
        print()


hopfield_net = HopfieldNetwork(training_data)


if __name__ == "__main__":
    hopfield_net = HopfieldNetwork(training_data)

    # for pattern in training_data:
    #     hopfield_net.visualize_pattern(pattern, "")

    num = 9
    hopfield_net.classify_digit(training_data[num], num)