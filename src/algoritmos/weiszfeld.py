import numpy as np

def weiszfeld(points, weights, epsilon=1e-6, max_iter=100):
    points = np.array(points)
    weights = np.array(weights)
    x = np.sum(points * weights[:, np.newaxis], axis=0) / np.sum(weights)
    
    for i in range(max_iter):
        distances = np.linalg.norm(points - x, axis=1)
        
        if np.any(distances < epsilon):
            return x
            
        T = np.sum(weights[:, np.newaxis] * points / distances[:, np.newaxis], axis=0) / np.sum(weights / distances)
        
        if np.linalg.norm(x - T) < epsilon:
            return T
            
        x = T
        
    return x
