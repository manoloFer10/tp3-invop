import numpy as np

def objective_function(x, points, weights):
    """
    Función distancia euclídea con pesos.
    """
    distances = np.linalg.norm(points - x, axis=1)
    return np.sum(weights * distances)

def gradient(x, points, weights):
    """
    Gradiente analítico de la función distancia pesada.
    """
    diff = points - x
    distances = np.linalg.norm(diff, axis=1)

    # Evitar división por cero para puntos coincidentes con x
    mask = distances > 1e-6
    
    # Si no hay puntos con distancia > epsilon, el gradiente es cero
    if not np.any(mask):
        return np.zeros_like(x)

    # El gradiente es la suma de wi * (x - pi) / ||x - pi||
    # (x - pi) es -diff
    grad = np.sum(weights[mask, np.newaxis] * (-diff[mask]) / distances[mask, np.newaxis], axis=0)
    
    return grad

def gradient_descent(points, weights, initial_x, epsilon=1e-6, max_iter=1000):
    points = np.array(points)
    weights = np.array(weights)
    x = np.array(initial_x)
    c1 = 0.25
    
    for i in range(max_iter):
        # Si x_k coincide con un p_i, el gradiente es indefinido.
        # devolvemos p_i en ese caso.
        distances = np.linalg.norm(points - x, axis=1)
        if np.any(distances < 1e-9):
            return x

        grad = gradient(x, points, weights)
        
        if np.linalg.norm(grad) < epsilon:
            return x
        
        alpha = 1.0
        
        current_obj_val = objective_function(x, points, weights)
        grad_dot_grad = np.dot(grad, grad)

        while objective_function(x - alpha * grad, points, weights) >= current_obj_val - c1 * alpha * grad_dot_grad:
            alpha *= 0.5
            # Evitamos que alpha se vuelva demasiado pequeño y el while no temrine nunca.
            if alpha < 1e-12:
                break
        
        x_new = x - alpha * grad
        
        if np.linalg.norm(x_new - x) < epsilon:
            return x_new
            
        x = x_new
        
    return x
