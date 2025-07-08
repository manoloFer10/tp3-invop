from src.algoritmos.weiszfeld import weiszfeld
from src.algoritmos.coordinate_descent import coordinate_descent
from src.algoritmos.gradient_descent import gradient_descent
from src.utils.data_generator import generate_data
from src.utils.plotting import plot_results
import numpy as np
import time
import csv

def main():
    seed = 42
    
    # Solo para mostrar
    num_points_plot = 50
    points_plot, weights_plot = generate_data(num_points_plot, seed=seed)
    plot_results(points_plot, weights_plot, None, 'Datos de Entrada', 'datos_entrada.png')
    initial_x_plot = np.mean(points_plot, axis=0)

    # Weiszfeld
    optimal_point_w = weiszfeld(points_plot, weights_plot)
    plot_results(points_plot, weights_plot, optimal_point_w, 'Weiszfeld', 'weiszfeld.png')

    # Descenso Coordinado
    optimal_point_cd = coordinate_descent(points_plot, weights_plot, initial_x_plot)
    plot_results(points_plot, weights_plot, optimal_point_cd, 'Descenso Coordinado', 'descenso_coordinado.png')

    # Descenso por Gradiente
    optimal_point_gd = gradient_descent(points_plot, weights_plot, initial_x_plot)
    plot_results(points_plot, weights_plot, optimal_point_gd, 'Descenso por Gradiente', 'descenso_gradiente.png')

    # Testeos para tiempos
    point_counts = [5000, 10000, 20000, 50000, 100000]
    results = []

    for count in point_counts:
        points, weights = generate_data(count, seed=seed)
        initial_x = np.mean(points, axis=0)

        #Weiszfeld
        start_time = time.time()
        opt_weis = weiszfeld(points, weights)
        weiszfeld_time = time.time() - start_time

        # Descenso Coordinado
        start_time = time.time()
        opt_coord = coordinate_descent(points, weights, initial_x)
        cd_time = time.time() - start_time

        # Descenso por Gradiente
        start_time = time.time()
        opt_grad = gradient_descent(points, weights, initial_x)
        gd_time = time.time() - start_time
        
        results.append([count, weiszfeld_time, opt_weis, cd_time, opt_coord, gd_time, opt_grad])
        print(f"{count} puntos terminado.")
        print(time.strftime("%H:%M:%S", time.gmtime(time.time())))

    with open('res_temporales_2d.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['#puntos', 'weiszfeld_tiempo', 'weiszfeld_optimo', 'coordinate_descent_tiempo', 'coordinate_descent_optimo', 'gradient_descent_tiempo', 'gradient_descent_optimo'])
        writer.writerows(results)

    # Experimentos con 3 y 6 coordenadas
    dims = [3, 6]
    for dim in dims:
        results = []

        for count in point_counts:
            points, weights = generate_data(count, dim=dim, seed=seed)
            initial_x = np.mean(points, axis=0)

            # Weiszfeld
            start_time = time.time()
            opt_weis = weiszfeld(points, weights)
            weiszfeld_time = time.time() - start_time

            # Descenso Coordinado
            start_time = time.time()
            opt_coord = coordinate_descent(points, weights, initial_x)
            cd_time = time.time() - start_time

            # Descenso por Gradiente
            start_time = time.time()
            opt_grad = gradient_descent(points, weights, initial_x)
            gd_time = time.time() - start_time
            
            results.append([count, weiszfeld_time, opt_weis, cd_time, opt_coord, gd_time, opt_grad])
            print(f"{count} puntos con {dim} coordenadas terminado.")
            print(time.strftime("%H:%M:%S", time.gmtime(time.time())))


        with open(f'res_temporales_{dim}d.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['#puntos', 'weiszfeld_tiempo', 'weiszfeld_optimo', 'coordinate_descent_tiempo', 'coordinate_descent_optimo', 'gradient_descent_tiempo', 'gradient_descent_optimo'])
            writer.writerows(results)

if __name__ == '__main__':
    main()
