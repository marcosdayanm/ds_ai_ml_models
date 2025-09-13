import math
import random
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def euclidean_distance(a: List[float], b: List[float]):
    if len(a) != len(b):
        raise ValueError("a and b dimensionality aren't equal")
    return math.sqrt(sum((i-j)**2 for i, j in zip(a, b)))


def select_centroids(data: List[Tuple[float, float]], k: int):
    centroids = []

    while len(centroids) < k:
        i = random.randint(0, len(data))
        if data[i] not in centroids:
            centroids.append(data[i])
    
    return centroids


def compute_centroid_mean(centroid_points: List[Tuple[float, float]]) -> Tuple[float, float]:
    cx, cy = 0, 0

    for x, y in centroid_points:
        cx += x
        cy += y

    return cx/len(centroid_points), cy/len(centroid_points)


def init_centr_points(centroids: List[Tuple[float, float]]) -> Dict[Tuple[float, float], List]:
    centr_points: Dict[Tuple[float, float], List] = dict()
    for c in centroids:
        centr_points[c] = []
    return centr_points


def k_means(data: List[Tuple[float, float]], k: int, epochs: int = 100) -> Tuple[List[Tuple[float, float]], Dict[Tuple[float, float], List]]:
    centroids = select_centroids(data, k)

    centr_points = init_centr_points(centroids)

    for _ in range(epochs):
        centr_points = init_centr_points(centroids)
        for point in data:
            closest = centroids[0]
            distance = euclidean_distance(list(point), list(closest))

            for c in centroids:
                temp_distance = euclidean_distance(list(point), list(c))
                if temp_distance < distance:
                    closest = c
                    distance = temp_distance
            centr_points[closest].append(point)
        
        new_centroids = []

        for c in centroids:
            new_centroids.append(compute_centroid_mean(centr_points[c]))

        centroids = new_centroids

    return centroids, centr_points



def plot_points_centroids(data: List[Tuple[float, float]], centroids: List[Tuple[float, float]]):
    plt.scatter(*zip(*data), c='blue', label='Data Points')
    plt.scatter(*zip(*centroids), c='red', label='Centroids')
    plt.legend()
    plt.show()


def plot_centr_points(centr_points: Dict[Tuple[float, float], List[Tuple[float, float]]]):
    colors = ['red', 'green', 'blue', 'purple', 'orange']
    for i, (centroid, points) in enumerate(centr_points.items()):
        color = colors[i % len(colors)]
        # Plot centroid with the same color as its points, but different marker
        plt.scatter(*centroid, c=color, marker='x', s=100)
        if points:
            xs, ys = zip(*points)
            plt.scatter(xs, ys, c=color, s=5)
    plt.show()


def read_data(file_path: str) -> List[Tuple[float, float]]:
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            x, y = map(float, line.strip().split(' '))
            data.append((x, y))
    return data


if __name__ == "__main__":
   data = read_data("ex7data2.txt")
   k = 3
   centroids, centr_points = k_means(data, k)
   # plot_points_centroids(data, centroids)
   plot_centr_points(centr_points)
