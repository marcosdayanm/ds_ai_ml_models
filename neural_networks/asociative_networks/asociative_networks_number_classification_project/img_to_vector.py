import os
from typing import Dict, List, Literal, Tuple

from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import numpy as np


def img_to_vector(input_path: str, n: int = 100, threshold: int = 160) -> np.ndarray:
    img = Image.open(input_path)
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img.convert("RGBA")).convert("RGB")
    img = img.convert("L")

    arr = np.array(img, dtype=np.uint8)
    ink = (arr < threshold).astype(np.uint8)       # 1 = tinta, 0 = fondo

    # Engrosa para mostrar trazos finos
    pil_bin = Image.fromarray((ink * 255).astype(np.uint8))
    pil_bin = pil_bin.filter(ImageFilter.MaxFilter(size=3)) 
    pil_bin = pil_bin.resize((n, n), resample=Image.NEAREST)

    ink = (np.array(pil_bin) > 0).astype(np.uint8)

    # Sin auto-invertir: negro = -1, blanco = +1
    vec = np.where(ink == 1, -1, 1).astype(np.int8).ravel()
    return vec


def load_labeled_vectors_hopfield(root: str, img_side_size: int = 28, color_threshold: int = 200, per_label_max: int | None = None):
    patterns, labels = [], []
    rng = np.random.default_rng()
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
            vec = img_to_vector(path, n=img_side_size, threshold=color_threshold)
            patterns.append(vec)
            labels.append(lbl)
    if not patterns:
        raise ValueError(f"No se encontraron imágenes en: {root}")
    return patterns, labels


def plot_matrix(matrix: np.ndarray):
    img = (matrix + 1) // 2  # -1 → 0, 1 → 1

    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.show()



TargetMode = Literal["onehot", "auto"]  # "auto" = salida = entrada (autoasociación)

def load_labeled_vectors_alm(
    root: str,
    img_side_size: int = 28,
    color_threshold: int = 160,
    per_label_max: int | None = None,
    target_mode: TargetMode = "onehot",
    rng: np.random.Generator | None = None,
) -> Tuple[List[List[float]], List[List[float]], List[str], Dict[str, int]]:
    """
    Carga imágenes desde `root/<label>/*.png|jpg|jpeg` y devuelve:
      - X: lista de vectores de entrada (0/1) tamaño img_side_size^2
      - Y: lista de vectores de salida (0/1):
            - "onehot": vector de longitud = #labels, con 1 en la clase
            - "auto"  : la misma imagen (autoasociación)
      - labels: lista paralela de etiquetas (str) de cada ejemplo
      - label_to_index: mapeo etiqueta -> índice en el one-hot
    """
    if rng is None:
        rng = np.random.default_rng()

    # orden fijo de etiquetas según carpetas
    label_names = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    if not label_names:
        raise ValueError(f"No se encontraron subcarpetas de etiquetas en: {root}")
    label_to_index = {lbl: i for i, lbl in enumerate(label_names)}

    X: List[List[float]] = []
    Y: List[List[float]] = []
    labels_out: List[str] = []

    for lbl in label_names:
        folder = os.path.join(root, lbl)
        files = [f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        if not files:
            continue
        if per_label_max is not None:
            k = min(per_label_max, len(files))
            files = list(rng.choice(files, size=k, replace=False))

        for fname in files:
            path = os.path.join(folder, fname)
            # img_to_vector devuelve {-1, +1} con negro=-1
            v_pm1 = img_to_vector(path, n=img_side_size, threshold=color_threshold)
            # a {0,1} con 1 = tinta (negro)
            v01 = (v_pm1 < 0).astype(np.uint8)  # True->1 para pixeles negros
            X.append(v01.astype(float).tolist())

            if target_mode == "onehot":
                y = np.zeros(len(label_names), dtype=np.uint8)
                y[label_to_index[lbl]] = 1
                Y.append(y.astype(float).tolist())
            elif target_mode == "auto":
                Y.append(v01.astype(float).tolist())
            else:
                raise ValueError(f"target_mode desconocido: {target_mode}")

            labels_out.append(lbl)

    if not X:
        raise ValueError(f"No se encontraron imágenes en: {root}")

    return X, Y, labels_out, label_to_index

