from typing import List, Tuple
import numpy as np
from collections import defaultdict
from img_to_vector import load_labeled_vectors_hopfield, plot_matrix


class HopfieldNetwork:
    def __init__(self, patterns: List[np.ndarray], labels=None):
        if not patterns:
            raise ValueError("Se requiere al menos un patrón para entrenar.")
        sizes = {p.size for p in patterns}
        if len(sizes) != 1:
            raise ValueError("Todos los patrones deben tener la misma longitud.")
        self.patterns = [p.astype(np.int8).ravel() for p in patterns]
        self.labels = labels if labels is not None else list(range(len(self.patterns)))
        self.n_neurons = self.patterns[0].size
        self.weights = self._compute_weights()

    def _compute_weights(self) -> np.ndarray:
        N = self.n_neurons
        W = np.zeros((N, N), dtype=np.float32)
        for p in self.patterns:
            v = p.astype(np.float32)
            W += np.outer(v, v)
        np.fill_diagonal(W, 0.0)
        W /= N
        return W

    @staticmethod
    def _sign(v: np.ndarray) -> np.ndarray:
        return np.where(v >= 0, 1, -1).astype(np.int8)

    def _converge(self, pattern: np.ndarray, max_iterations: int = 50) -> Tuple[np.ndarray, int, bool]:
        if pattern.size != self.n_neurons:
            raise ValueError(f"El patrón debe tener {self.n_neurons} elementos.")
        x = pattern.astype(np.int8).copy()
        rng = np.random.default_rng(0)
        for it in range(max_iterations):
            changed = 0
            for j in rng.permutation(self.n_neurons):
                s = 1 if (self.weights[j] @ x) >= 0 else -1
                if s != x[j]:
                    x[j] = s
                    changed += 1
            if changed == 0:
                return x, it, True
        return x, max_iterations, False

    def classify_digit(self, pattern: np.ndarray, expected_result: int):
        plot_matrix(pattern.reshape(int(np.sqrt(self.n_neurons)), -1))
        recovered, iterations, converged = self._converge(pattern)
        if not converged:
            print(f"No convergió tras {iterations} iteraciones")

        den = (np.linalg.norm(recovered) + 1e-8)
        sims = [float(recovered @ p) / (den * (np.linalg.norm(p) + 1e-8)) for p in self.patterns]
        sims_abs = [abs(s) for s in sims]
        pred_idx = int(np.argmax(sims_abs))
        pred_label = self.labels[pred_idx]
        

        plot_matrix(recovered.reshape(int(np.sqrt(self.n_neurons)), -1))

        return pred_label, recovered, iterations


if __name__ == "__main__":
    n = 28
    threshold = 160
    dataset_folder = "number_dataset"
    patterns, labels = load_labeled_vectors_hopfield(root=dataset_folder, img_side_size=n, color_threshold=threshold, per_label_max=1)
    print("Dataset loaded")

    test_patterns, test_labels = load_labeled_vectors_hopfield(root=dataset_folder, img_side_size=n, color_threshold=threshold, per_label_max=1)

    # init network
    hopfield_net = HopfieldNetwork(patterns, labels)
    print("Hopfield Network initialized")

    # predict and results
    print("Predictions:")

    idxs = list(range(len(test_patterns)))

    correct = 0
    for i in idxs:
        pred, _, _ = hopfield_net.classify_digit(test_patterns[i], expected_result=test_labels[i])
        if pred == test_labels[i]:
            correct += 1
        print(f"Pred: {pred}, Esperado: {test_labels[i]}")
    accuracy = (correct / len(idxs))*100
    print(f"Accuracy ({len(idxs)} muestras): {accuracy:.2f}%")
