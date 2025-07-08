import numpy as np
import matplotlib.pyplot as plt
from src.utils.plotting import plot_results

def generate_uniform_data(num_points, dim=2, seed=None):
    """Distribución uniforme."""
    if seed:
        np.random.seed(seed)
    points = np.random.rand(num_points, dim) * 100
    weights = np.random.rand(num_points) * 10
    return points, weights

def generate_clustered_data(num_points, dim=2, num_clusters=5, seed=None):
    """Distribución por clusters distribuidos uniformemente."""
    if seed:
        np.random.seed(seed)
    points = []
    weights = []
    cluster_centers = np.random.rand(num_clusters, dim) * 100
    points_per_cluster = num_points // num_clusters
    
    for i in range(num_clusters):
        cluster_points = np.random.randn(points_per_cluster, dim) * 10 + cluster_centers[i]
        cluster_weights = np.random.rand(points_per_cluster) * 10
        points.append(cluster_points)
        weights.append(cluster_weights)
        
    remaining_points = num_points % num_clusters
    if remaining_points > 0:
        cluster_points = np.random.randn(remaining_points, dim) * 10 + cluster_centers[0]
        cluster_weights = np.random.rand(remaining_points) * 10
        points.append(cluster_points)
        weights.append(cluster_weights)

    return np.vstack(points), np.hstack(weights)

def generate_linear_data(num_points, dim=2, seed=None):
    """Puntos distribuidos linealmente en una dirección (con un ruido)."""
    if seed:
        np.random.seed(seed)
    t = np.linspace(0, 100, num_points)
    points = np.zeros((num_points, dim))
    direction = np.random.rand(dim)
    direction /= np.linalg.norm(direction)
    
    for i in range(dim):
        points[:, i] = t * direction[i]
        
    # Ruido 
    noise = np.random.randn(num_points, dim) * 5
    points += noise
    weights = np.random.rand(num_points) * 10
    return points, weights

def generate_circular_data(num_points, seed=None):
    """Puntos sobre un cítculo."""
    if seed:
        np.random.seed(seed)
    radius = 50
    angles = np.linspace(0, 2 * np.pi, num_points)
    x = radius * np.cos(angles) + 50 + np.random.randn(num_points) * 5
    y = radius * np.sin(angles) + 50 + np.random.randn(num_points) * 5
    points = np.vstack([x, y]).T
    weights = np.random.rand(num_points) * 10
    return points, weights

def generate_scenario_data(scenario, num_points, dim=2, seed=None):

    if scenario == 'uniforme':
        return generate_uniform_data(num_points, dim, seed)
    elif scenario == 'clusterizado':
        return generate_clustered_data(num_points, dim, seed=seed)
    elif scenario == 'lineal':
        return generate_linear_data(num_points, dim, seed)
    elif scenario == 'circular':
        if dim != 2:
            raise ValueError("Solo implementamos el circular para R^2.")
        return generate_circular_data(num_points, seed)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

if __name__ == '__main__':
    scenarios_to_plot = ['uniform', 'clustered', 'linear', 'circular']
    num_points_demo = 150
    seed_demo = 42

    for scenario in scenarios_to_plot:
        print(f"--- Escenario: {scenario.capitalize()} ---")
        try:
            points, weights = generate_scenario_data(scenario, num_points_demo, seed=seed_demo)
            title = f'Escenario de prueba: {scenario.capitalize()}'
            save_path = f'escenarioPrueba_{scenario}.png'
            plot_results(points, weights, None, title, save_path=save_path)
        except ValueError as e:
            print(f"No se pudo generar '{scenario}': {e}")
    