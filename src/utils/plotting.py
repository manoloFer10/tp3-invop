import matplotlib.pyplot as plt
import numpy as np

def plot_results(points, weights, optimal_point, title, save_path=None):

    points = np.array(points)
    plt.figure(figsize=(8, 8))
    plt.scatter(points[:, 0], points[:, 1], s=weights*20, c='b', alpha=0.5, label='Puntos')
    if optimal_point is not None:
        plt.scatter(optimal_point[0], optimal_point[1], c='r', s=100, marker='*', label='Punto Óptimo')
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    
    #plt.show()
