import os
from typing import List, Tuple
import numpy as np
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from collections import defaultdict

def img_to_vector(input_path: str, n: int = 100, threshold: int = 200) -> np.ndarray:
    """
    Carga una imagen, compone sobre fondo blanco si trae alfa, engrosa el trazo,
    redimensiona a nxn (por defecto 100x100), binariza con 'threshold' y devuelve
    un vector 1D en {-1, 1} con negro=-1 y blanco=1.
    """
    img = Image.open(input_path)

    # 1) Componer sobre blanco si tiene canal alfa
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img.convert("RGBA")).convert("RGB")

    # 2) Escala de grises
    img = img.convert("L")

    # 3) Engrosar trazo para que sobreviva al downscale
    img = img.filter(ImageFilter.MinFilter(size=3))

    # 4) Redimensionar a n×n
    img = img.resize((n, n), resample=Image.LANCZOS)

    # 5) Binarizar con umbral (más alto que 128 para no perder trazos delgados)
    arr = np.array(img, dtype=np.uint8)
    # 1 donde es “tinta” (oscuro), 0 donde es fondo (claro)
    ink = (arr < threshold).astype(np.int8)
    if ink.mean() > 0.5:
        ink = 1 - ink

    # 6) Mapear a {-1, 1}: negro=-1, blanco=1
    vec = np.where(ink == 1, -1, 1).astype(np.int8).ravel()
    return vec


def load_vectors(folder: str, n: int = 100, threshold: int = 200) -> List[np.ndarray]:
    vectors = []
    for filename in sorted(os.listdir(folder)):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(folder, filename)
            vec = img_to_vector(path, n=n, threshold=threshold)
            vectors.append(vec)
    if not vectors:
        raise ValueError(f"No se encontraron imágenes en: {folder}")
    return vectors


def load_labeled_vectors(root: str, n: int = 28, threshold: int = 200, per_label_max: int | None = None):
    patterns, labels = [], []
    rng = np.random.default_rng(42)
    for lbl in sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]):
        folder = os.path.join(root, lbl)
        files = [f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        if not files:
            continue
        if per_label_max is not None:
            k = min(per_label_max, len(files))
            files = list(rng.choice(files, size=k, replace=False))
        for fname in files:
            path = os.path.join(folder, fname)
            vec = img_to_vector(path, n=n, threshold=threshold)
            patterns.append(vec)
            labels.append(lbl)
    if not patterns:
        raise ValueError(f"No se encontraron imágenes en: {root}")
    return patterns, labels


def plot_matrix(matrix: np.ndarray):
    # Convertimos los -1 en 0 para que matplotlib los pinte como negro
    img = (matrix + 1) // 2  # -1 → 0, 1 → 1

    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.show()



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
            W += np.outer(p, p)
        W /= N
        np.fill_diagonal(W, 0.0)
        return W

    @staticmethod
    def _sign(v: np.ndarray) -> np.ndarray:
        return np.where(v >= 0, 1, -1).astype(np.int8)

    def _converge(self, pattern: np.ndarray, max_iterations: int = 50) -> Tuple[np.ndarray, int, bool]:
        if pattern.size != self.n_neurons:
            raise ValueError(f"El patrón debe tener {self.n_neurons} elementos.")
        x = self._sign(pattern.astype(np.int8))
        rng = np.random.default_rng(0)
        for it in range(max_iterations):
            changed = 0
            for j in rng.permutation(self.n_neurons):
                s = 1 if (self.weights[j] @ x) >= 0 else -1
                if s != x[j]:
                    x[j] = s
                    changed += 1
            print("CONVERGE",x)
            if changed == 0:
                return x, it, True
        return x, max_iterations, False

    def classify_digit(self, pattern: np.ndarray, expected_result: int):
        recovered, iterations, converged = self._converge(pattern)
        if not converged:
            print(f"⚠️ No convergió tras {iterations} iteraciones")

        sims = [float(np.dot(recovered, p)) / self.n_neurons for p in self.patterns]
        agg = defaultdict(list)
        for s, lbl in zip(sims, self.labels):
            agg[lbl].append(s)
        pred_label, best_score = max(((lbl, float(np.mean(v))) for lbl, v in agg.items()),
                                    key=lambda t: t[1])
        
        plot_matrix(recovered.reshape(int(np.sqrt(self.n_neurons)), -1))

        return pred_label, recovered, iterations


if __name__ == "__main__":
    
    # load number dataset
    # n = 28
    # threshold = 200
    # dataset_folder = "exclude_dataset"
    # patterns, labels = load_labeled_vectors(dataset_folder, n, threshold, per_label_max=1)
    print("Dataset loaded")


    # patterns = [
    #     np.array([1,1,1,1,1,1,1,1], dtype=np.int8),
    #     np.array([1,1,1,1,-1,-1,-1,-1], dtype=np.int8),
    #     np.array([1,-1,1,-1,1,-1,1,-1], dtype=np.int8),
    # ]


    test_patterns = [
        np.array([-1, 1, -1, 1], dtype=np.int8),
        np.array([-1, 1, 1, 1], dtype=np.int8),
        np.array([1, -1, 1, -1], dtype=np.int8),
        np.array([1, 1, 1, -1], dtype=np.int8),
        np.array([-1, -1, 1, 1], dtype=np.int8),
        np.array([-1, 1, 1, -1], dtype=np.int8),
        np.array([1, -1, -1, 1], dtype=np.int8),
        np.array([1, 1, -1, -1], dtype=np.int8),
    ] 

    patterns = [
        np.array([-1, 1, -1, 1], dtype=np.int8),
        np.array([1, -1, 1, -1], dtype=np.int8),
    ]

    # init labels
    labels = [0, 1, 2, 3, 4, 5, 6, 7]

    # init network
    hopfield_net = HopfieldNetwork(patterns, labels)
    print("Hopfield Network initialized")

    print("PESOS")
    print(hopfield_net.weights)

    # predict and results
    print("Predicciones:")
    # rng = np.random.default_rng(0)
    # k = min(200, len(patterns))
    # idxs = rng.choice(len(patterns), size=k, replace=False)

    idxs = list(range(len(test_patterns)))

    correct = 0
    # print(patterns[idxs[50]])
    # plot_matrix(patterns[idxs[72]].reshape(28, 28))
    for i in idxs:
        pred, _, _ = hopfield_net.classify_digit(test_patterns[i], expected_result=labels[i])
        if pred == labels[i]:
            correct += 1
        print(f"Pred: {pred}, Esperado: {labels[i]}")
    accuracy = correct / len(idxs)
    print(f"Precisión aleatoria ({len(idxs)} muestras): {accuracy:.3f}")
