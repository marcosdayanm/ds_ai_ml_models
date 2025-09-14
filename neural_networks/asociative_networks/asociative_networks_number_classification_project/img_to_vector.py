import os
from typing import Dict, List, Tuple

from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import numpy as np


def plot_matrix(matrix: np.ndarray):
    """Muestra una matriz 2D como imagen en escala de grises."""
    img = (matrix + 1) // 2  # -1 → 0, 1 → 1
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.show()


def img_to_vector(input_path: str, n: int = 100, threshold: int = 160) -> np.ndarray:
    """Convierte una imagen en un vector de características."""
    img = Image.open(input_path)
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img.convert("RGBA")).convert("RGB")
    img = img.convert("L")

    arr = np.array(img, dtype=np.uint8)
    ink = (arr < threshold).astype(np.uint8)

    pil_bin = Image.fromarray((ink * 255).astype(np.uint8))
    pil_bin = pil_bin.filter(ImageFilter.MaxFilter(size=1))
    pil_bin = pil_bin.resize((n, n), resample=Image.Resampling.NEAREST)

    ink = (np.array(pil_bin) > 0).astype(np.uint8)

    vec = np.where(ink == 1, 1, -1).astype(np.int8).ravel()
    return vec


def load_labeled_vectors_hopfield(root: str, img_side_size: int = 28, color_threshold: int = 200, per_label_max: int | None = None) -> Tuple[List[np.ndarray], List[int]]:
    """
    Carga imágenes desde el directorio "root" y devuelve una lista de vectores de características y etiquetas en forma de integers."""
    patterns, labels = [], []
    for lbl in sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]):
        folder = os.path.join(root, lbl)
        files = [f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        if not files:
            continue
        if per_label_max is not None:
            k = min(per_label_max, len(files))
            files = files[:k]
        for fname in files:
            path = os.path.join(folder, fname)
            vec = img_to_vector(path, n=img_side_size, threshold=color_threshold)
            patterns.append(vec)
            labels.append(lbl)
    if not patterns:
        raise ValueError(f"No se encontraron imágenes en: {root}")
    return patterns, labels


def load_labeled_vectors_alm(
    root: str,
    img_side_size: int = 28,
    color_threshold: int = 160,
    per_label_max: int | None = None,
) -> Tuple[List[List[float]], List[List[float]], List[str], Dict[str, int]]:
    """
    Carga imágenes desde "root" y devuelve lista de vectores de entrada, y para la salida, un vector del tamaño de los diferentes labels, con one hot encoding, además de las etiquetas y un diccionario label a índice.
    """
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
            files = files[:k]

        for fname in files:
            path = os.path.join(folder, fname)
            v_pm1 = img_to_vector(path, n=img_side_size, threshold=color_threshold)
            v01 = (v_pm1 > 0).astype(np.uint8)
            X.append(v01.astype(float).tolist())

            y = np.zeros(len(label_names), dtype=np.uint8)
            y[label_to_index[lbl]] = 1
            Y.append(y.astype(float).tolist())

            labels_out.append(lbl)

    if not X:
        raise ValueError(f"No se encontraron imágenes en: {root}")

    return X, Y, labels_out, label_to_index



def load_labeled_vectors_bam(
    root: str,
    img_side_size: int = 28,
    color_threshold: int = 160,
    per_label_max: int | None = None,
):
    """
    Carga imágenes desde "root" y las convierte en vectores de características y devuelve una lista de vectores de entrada, una lista de vectores de salida con one hot encoding, las etiquetas, un diccionario label a índice y una lista de nombres de etiquetas.
    """
    import os
    import numpy as np

    label_names = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    if not label_names:
        raise ValueError(f"No se encontraron subcarpetas de etiquetas en: {root}")
    label_to_index = {lbl: i for i, lbl in enumerate(label_names)}

    X, Y, labels_out = [], [], []

    for lbl in label_names:
        folder = os.path.join(root, lbl)
        files = sorted([f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
        if not files:
            continue
        if per_label_max is not None:
            files = files[:min(per_label_max, len(files))]

        for fname in files:
            path = os.path.join(folder, fname)
            v_pm1 = img_to_vector(path, n=img_side_size, threshold=color_threshold)  # ±1
            X.append(v_pm1.astype(np.int8))  # 1D

            y = -np.ones(len(label_names), dtype=np.int8)
            y[label_to_index[lbl]] = 1
            Y.append(y)  # 1D

            labels_out.append(lbl)

    if not X:
        raise ValueError(f"No se encontraron imágenes en: {root}")

    return X, Y, labels_out, label_to_index, label_names
