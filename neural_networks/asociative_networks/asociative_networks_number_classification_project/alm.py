

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


def alm_predict_label(alm, x01: np.ndarray, label_names: List[str]) -> str:
    """
    x01: vector columna (N,1) con bits 0/1 de la imagen
    Devuelve la etiqueta (string) según argmax del vector de salida.
    """
    y = alm.predict(x01.copy())  # tu predict hace umbral y regresa 0/1
    idx = int(np.argmax(y))
    return label_names[idx]



if __name__ == "__main__":


    # input_vectors = [
    #     [1.0, 1.0, 1.0, 1.0],
    #     [-1.0,-1.0,-1.0,-1.0],
    #     [-1.0, 1.0, -1.0, 1.0],
    # ]

    # output_vectors = [
    #     [1.0, 1.0, 1.0, 1.0],
    #     [-1.0,-1.0,-1.0,-1.0],
    #     [-1.0, 1.0, -1.0, 1.0],
    # ]

    # test_input = np.array([-1, -1, -1, -1]).reshape(-1, 1)

    # alm = ALM(input_vectors, output_vectors)

    # alm.print_matrix()

    # # alm.print_matrix()

    # print(alm.predict(test_input))

    rng_train = np.random.default_rng(123)
    rng_test  = np.random.default_rng(456)

    X_train, Y_train, train_labels, label_to_index = load_labeled_vectors_alm(
        root="number_dataset",
        img_side_size=28,
        color_threshold=160,
        per_label_max=3,
        target_mode="onehot",
        rng=rng_train
    )
    label_names = [lbl for lbl, _ in sorted(label_to_index.items(), key=lambda t: t[1])]

    X_test, Y_test, test_labels, _ = load_labeled_vectors_alm(
        root="number_dataset",
        img_side_size=28,
        color_threshold=160,
        per_label_max=1,
        target_mode="onehot",   # el test usa el mismo esquema de salida
        rng=rng_test
    )

    # Entrenar ALM (tu clase espera listas de listas)
    alm = ALM(X_train, Y_train)

    # Evaluar
    correct = 0
    for x, lbl in zip(X_test, test_labels):
        x_col = np.array(x, dtype=float).reshape(-1, 1)  # (N,1)
        pred_lbl = alm_predict_label(alm, x_col, label_names)
        correct += int(pred_lbl == lbl)

    print(f"Accuracy: {correct/len(X_test):.3f}")