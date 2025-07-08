import numpy as np

def generate_data(num_points, dim=2, seed=None):
    """
    Genera puntos aleatorios en el espacio de dimensión = dim.
    Los pesos también son aleatorios.
    Los puntos están en el rango [0, 100) y los pesos en [0, 10).
    """
    if seed:
        np.random.seed(seed)
    points = np.random.rand(num_points, dim) * 100
    weights = np.random.rand(num_points) * 10
    return points, weights
