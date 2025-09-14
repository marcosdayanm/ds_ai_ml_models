

from typing import List, Tuple
import numpy as np

from img_to_vector import load_labeled_vectors_alm


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

        counts = np.array(output_vectors, dtype=float).sum(axis=0)  # shape: (num_clases,)
        counts[counts == 0] = 1.0
        W = W / counts.reshape(-1, 1)
        self.W = W
        self.bias = np.zeros((W.shape[0], 1), dtype=float)

    def predict(self, input_predict: np.ndarray) -> np.ndarray:
        if self.W is None or self.bias is None:
            raise ValueError("Model is not trained yet.")
        return self.W @ input_predict + self.bias
    
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


def alm_predict_label_onehot(alm: ALM, x01: np.ndarray, label_names: List[str]) -> Tuple[np.ndarray, str]:
    """
    x01: vector columna (N,1) con bits 0/1 de la imagen
    Devuelve la etiqueta (string) según argmax del vector de salida.
    """
    y = alm.predict(x01.copy())  # tu predict hace umbral y regresa 0/1
    idx = int(np.argmax(y))
    return y, label_names[idx]


if __name__ == "__main__":
    X_train, Y_train, train_labels, label_to_index = load_labeled_vectors_alm(
        root="number_dataset",
        img_side_size=28,
        color_threshold=160,
        per_label_max=3,
    )
    label_names = [lbl for lbl, _ in sorted(label_to_index.items(), key=lambda t: t[1])]

    X_test, Y_test, test_labels, _ = load_labeled_vectors_alm(
        root="number_dataset",
        img_side_size=28,
        color_threshold=160,
        per_label_max=2,
    )

    # Entrenar ALM (tu clase espera listas de listas)
    alm = ALM(X_train, Y_train)

    # Evaluar
    correct = 0
    for x, lbl in zip(X_test, test_labels):
        x_col = np.array(x, dtype=float).reshape(-1, 1)  # (N,1)
        y, pred_lbl = alm_predict_label_onehot(alm, x_col, label_names)
        print(f"Esperado: {lbl}, Predicho: {pred_lbl}")
        correct += int(pred_lbl == lbl)

    print(f"Accuracy: {correct/len(X_test):.3f}")