from __future__ import annotations
import argparse
import math
import os
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
from PIL import Image

import matplotlib.pyplot as plt


def findClosestCentroids(X: np.ndarray, initial_centroids: np.ndarray) -> np.ndarray:
    """
    Calcula el centroide más cercano para cada punto y regresa un array con el índice de cada centroide más cercano por cada punto.
    """
    # Ésta era mi implementación, usé ChatGPT5 para optimizar las operaciones con matrices que de esta forma era muy ineficiente
    # idx = np.zeros(X.shape[0], dtype=int)
    # for point_idx, point in enumerate(X):
    #     closest = initial_centroids[0]
    #     distance = _euclidean_distance(point, closest)
    #     for c_idx, c in enumerate(initial_centroids):
    #         temp_distance = _euclidean_distance(point, c)
    #         if temp_distance < distance:
    #             closest = c
    #             distance = temp_distance
    #             idx[point_idx] = c_idx

    diffs = X[:, None, :] - initial_centroids[None, :, :] # se resta cada punto con cada centroide. los None en las posiciones 1 y 0 agregan a las matrices para operar con ellas
    dists = np.sum(diffs ** 2, axis=2)  # se elevan todas las operaciones al cuadrado y se suman las componentes x,y,z
    idx = np.argmin(dists, axis=1)      # de la matriz (m,k), se toma en cada fila m, k más bajo que ese sería el centroide mas cercano
    return idx


def computeCentroids(X: np.ndarray, idx: np.ndarray, K: int) -> np.ndarray:
    """
    Recalcula los centroides como la media de los puntos asignados a cada cluster.
    Si algún cluster queda vacío, se posiciona en un lugar random.
    """
    m, n = X.shape
    centroids = np.zeros((K, n), dtype=X.dtype)
    for k in range(K):
        puntos_k = X[idx == k]
        if puntos_k.size == 0:
            centroids[k] = X[np.random.randint(0, m)] # si un centroide no tiene puntos, se posiciona en un lugar random de nuevo
        else:
            centroids[k] = np.mean(puntos_k, axis=0)
    return centroids


def kMeansInitCentroids(X: np.ndarray, K: int, random_state: Optional[int] = None) -> np.ndarray:
    """Inicializa los centroides seleccionando K ejemplos aleatorios de X."""
    if random_state is not None:
        np.random.seed(random_state) # esto es para que en debugging siempre sean los mismos números random

    m = X.shape[0] 
    perm = np.random.permutation(m)[:K] # crea un array de 0 a m-1 en orden aleatorio y lo corta a longitud K
    centroids = X[perm].copy() # toma los índices de perm dentro de X como centroides
    return centroids


def runkMeans(
    X: np.ndarray,
    initial_centroids: np.ndarray,
    max_iters: int,
    plot_error: bool = False
) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Ejecuta el algoritmo K-means por max_iters épocas.
    Si plot_error=True se grafica el error.
    """
    centroids = initial_centroids.copy()
    K = centroids.shape[0]
    error_history = []

    for _ in range(max_iters):
        idx = findClosestCentroids(X, centroids)
        centroids = computeCentroids(X, idx, K)

        if plot_error:
            error_history.append(_compute_error(X, centroids, idx))

    return centroids, idx, error_history


def _load_image_as_array(image_path: str) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Carga una imagen y la convierte en un arreglo (m,3) en los rangos [0,1].
    Uso de ChatGPT5 para interactuar con PIL.
    """
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0  # normaliza a [0,1]
    h, w, _ = arr.shape
    plain_img_array = arr.reshape(-1, 3)  # de (h,w,3) a (m, 3)
    return plain_img_array, (h, w)


def _array_to_image(arr_01: np.ndarray, shape_hw: Tuple[int, int]) -> Image.Image:
    """
    Convierte un arreglo (m,3) en [0,1] a imagen PIL RGB con shape dado.
    Uso de ChatGPT5 para interactuar con PIL.
    """
    h, w = shape_hw
    arr = np.clip(arr_01.reshape(h, w, 3) * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError("a and b dimensionalities are not equal")
    return np.sqrt(np.sum((a - b) ** 2))


def _compute_error(X: np.ndarray, centroids: np.ndarray, idx: np.ndarray) -> float:
    """
    Calcula el error cuadrático medio entre cada punto y su centroide asignado.
    """
    error = 0.0
    for i in range(X.shape[0]):
        c = centroids[idx[i]]
        error += np.sum((X[i] - c) ** 2)
    return error / X.shape[0]


def _print_compression_metrics(m: int, K: int, h: int, w: int):
    """Imprime métricas de compresión aproximadas.
    Usé ChatGPT5 para clacular éstas métricas.
    """
    original_bits = 24 * m
    bits_por_pixel_comprimida = math.ceil(math.log2(K))  # 16 -> 4 bits, un apximado
    compressed_bits_aprox = bits_por_pixel_comprimida * m
    ratio = original_bits / max(compressed_bits_aprox, 1)

    print(f"Tamaño pixeles: {m} ({h} x {w})")
    print(f"K: {K}")
    print(f"Bits originales (24 bpp): {original_bits:,}")
    print(f"Bits comprimidos aprox. ({bits_por_pixel_comprimida} bpp): {compressed_bits_aprox:,}")
    print(f"Razón aproximada de compresión: {ratio:,.2f} : 1")


def _plot_error_history(error_history: list):
    if not error_history:
        return
    plt.plot(range(1, len(error_history) + 1), error_history, marker='o')
    plt.xlabel('Iteración')
    plt.ylabel('Error cuadrático medio')
    plt.title('Error en cada Iteración')
    plt.grid()
    plt.show()
    

def compress_image_with_kmeans(
    image_path: str,
    K: int = 16,
    max_iters: int = 10,
    output_path: Optional[str] = None,
    random_state: Optional[int] = 42,
    plot_error: bool = True
) -> Tuple[str, np.ndarray, list]:
    """
    Wrapper de toda la funcionalidad del algoritmo K-means para compresión de imágenes.
    """
    X, shape_hw = _load_image_as_array(image_path)  # (m,3) en un rango [0,1]
    m = X.shape[0] # cantidad de pixeles totales en la imagen

    centroids0 = kMeansInitCentroids(X, K, random_state=random_state)
    centroids, idx, error_history = runkMeans(X, centroids0, max_iters=max_iters, plot_error=plot_error)

    final_idx = findClosestCentroids(X, centroids)
    X_compressed = centroids[final_idx]

    if output_path is None:
        base, _ = os.path.splitext(image_path)
        output_path = f"{base}_compressed_k{K}.png"

    out_img = _array_to_image(X_compressed, shape_hw)
    out_img.save(output_path, format="PNG", optimize=True)

    print(f"Imagen: {image_path}")
    _print_compression_metrics(m, K, shape_hw[0], shape_hw[1])

    return output_path, centroids, error_history


def main():
    image_path = "input.png"
    K = 16
    max_iters = 10
    output_path = None
    random_state = 1

    output_path, centroids, error_history = compress_image_with_kmeans(
        image_path=image_path,
        K=K,
        max_iters=max_iters,
        output_path=output_path,
        random_state=random_state,
    )

    print(f"Imagen comprimida guardada en: {output_path}")

    if error_history:
        _plot_error_history(error_history)


if __name__ == "__main__":
    main()
