import numpy as np
from scipy.optimize import minimize

def objective_function(x, points, weights):
    """
    Función distancia euclídea con pesos.
    """
    distances = np.linalg.norm(points - x, axis=1)
    return np.sum(weights * distances)

def coordinate_descent(points, weights, initial_x, step_size=0.1, epsilon=1e-6, max_iter=100):

    points = np.array(points)
    weights = np.array(weights)
    x = np.array(initial_x)
    
    for i in range(max_iter):
        x_prev = x.copy()
        
        for j in range(len(x)):
            def f(coord):
                x_temp = x.copy()
                x_temp[j] = coord
                return objective_function(x_temp, points, weights)
            
            # Usar 'Nelder-Mead' para minimizar la función unidimensional, es bueno 
            # para funciones no diferenciables.
            res = minimize(f, x[j], method='Nelder-Mead')
            x[j] = res.x
            
        if np.linalg.norm(x - x_prev) < epsilon:
            return x
            
    return x
