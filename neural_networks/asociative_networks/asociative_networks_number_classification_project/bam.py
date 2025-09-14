from typing import List
import numpy as np
from img_to_vector import load_labeled_vectors_bam


class BAM:
    def __init__(self, input: List[np.ndarray], output: List[np.ndarray]):
        if len(input) != len(output):
            raise ValueError("Input y output deben tener el mismo número de patrones.")
        
        self.input = input
        self.output = output
        self.n_in = input[0].size
        self.n_out = output[0].size
        self._train()

    def _train(self):
        # Inicializar matriz de pesos a cero
        weights = np.zeros((self.n_in, self.n_out), dtype=float)
        for x, y in zip(self.input, self.output):
            xv = np.asarray(x, dtype=float).ravel()
            yv = np.asarray(y, dtype=float).ravel()
            weights += np.outer(xv, yv)

        self.weights = weights

    def predict(self, x: np.ndarray) -> np.ndarray:
        if x.size != self.input[0].size:
            raise ValueError("Input size must match training data.")
        xv = np.asarray(x, dtype=float).ravel().reshape(-1, 1)
        return self.weights.T @ xv


def bam_predict_label(bam, x_vec, label_names):
    y = bam.predict(x_vec)
    y = np.asarray(y).ravel()
    idx = int(np.argmax(y))
    return y, label_names[idx]



if __name__ == "__main__":
    X_train, Y_train, _, label_to_index, label_names = load_labeled_vectors_bam(
        root="number_dataset", img_side_size=28, color_threshold=160, per_label_max=3
    )
    X_test, Y_test, test_labels, _, _ = load_labeled_vectors_bam(
        root="number_dataset", img_side_size=28, color_threshold=160, per_label_max=2
    )

    # Entrenar BAM
    bam = BAM(X_train, Y_train)

    # Evaluar
    correct = 0
    for x, lbl in zip(X_test, test_labels):
        y, pred_lbl = bam_predict_label(bam, x, label_names)
        print(f"Esperado: {lbl}, Predicho: {pred_lbl}")
        correct += int(pred_lbl == lbl)

    print(f"Accuracy: {correct/len(X_test):.3f}")