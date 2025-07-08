from src.algoritmos.weiszfeld import weiszfeld
from src.algoritmos.coordinate_descent import coordinate_descent
from src.algoritmos.gradient_descent import gradient_descent
from src.utils.data_generator import generate_data
from src.utils.plotting import plot_results
from src.utils.scenario_generator import generate_scenario_data
import numpy as np
import time
import csv
import os

def main():
    seed = 42
    scenarios = ['uniforme', 'clusterizado', 'lineal', 'circular']
    
    # Ploteo los datos en R^2
    for scenario in scenarios:
        output_dir = f"resultados_{scenario}"
        os.makedirs(output_dir, exist_ok=True)

        num_points_plot = 50
        points_plot, weights_plot = generate_scenario_data(scenario, num_points_plot, dim=2, seed=seed)
        plot_results(points_plot, weights_plot, None, f'Datos de Entrada - {scenario.capitalize()}', os.path.join(output_dir, f'datos_entrada_{scenario}.png'))
        initial_x_plot = np.mean(points_plot, axis=0)

        # Weiszfeld
        optimal_point_w = weiszfeld(points_plot, weights_plot)
        plot_results(points_plot, weights_plot, optimal_point_w, f'Weiszfeld - {scenario.capitalize()}', os.path.join(output_dir, f'weiszfeld_{scenario}.png'))

        # Descenso Coordinado
        optimal_point_cd = coordinate_descent(points_plot, weights_plot, initial_x_plot)
        plot_results(points_plot, weights_plot, optimal_point_cd, f'Descenso Coordinado - {scenario.capitalize()}', os.path.join(output_dir, f'descenso_coordinado_{scenario}.png'))

        # Descenso por Gradiente
        optimal_point_gd = gradient_descent(points_plot, weights_plot, initial_x_plot)
        plot_results(points_plot, weights_plot, optimal_point_gd, f'Descenso por Gradiente - {scenario.capitalize()}', os.path.join(output_dir, f'descenso_gradiente_{scenario}.png'))

    point_counts = [5000, 10000, 20000, 50000, 100000]
    
    # Dimensiones en R^2
    for scenario in scenarios:
        output_dir = f"resultados_{scenario}"
        os.makedirs(output_dir, exist_ok=True)
        results = []
        for count in point_counts:
            points, weights = generate_scenario_data(scenario, count, dim=2, seed=seed)
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
            print(f"{count} puntos para el escenario '{scenario}' terminado.")
            print(time.strftime("%H:%M:%S", time.gmtime(time.time())))

        with open(os.path.join(output_dir, f'res_temporales_2d.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['#puntos', 'weiszfeld_tiempo', 'weiszfeld_optimo', 'coordinate_descent_tiempo', 'coordinate_descent_optimo', 'gradient_descent_tiempo', 'gradient_descent_optimo'])
            writer.writerows(results)

    # Dimensiones R^3 y R^6 (sin circular)
    dims = [3, 6]
    scenarios_3d_6d = ['uniforme', 'clusterizado', 'lineal']
    for dim in dims:
        for scenario in scenarios_3d_6d:
            output_dir = f"resultados_{scenario}"
            os.makedirs(output_dir, exist_ok=True)
            results = []
            for count in point_counts:
                points, weights = generate_scenario_data(scenario, count, dim=dim, seed=seed)
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
                print(f"{count} puntos con {dim} coordenadas para el escenario '{scenario}' terminado.")
                print(time.strftime("%H:%M:%S", time.gmtime(time.time())))

            with open(os.path.join(output_dir, f'res_temporales_{dim}d.csv'), 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['#puntos', 'weiszfeld_tiempo', 'weiszfeld_optimo', 'coordinate_descent_tiempo', 'coordinate_descent_optimo', 'gradient_descent_tiempo', 'gradient_descent_optimo'])
                writer.writerows(results)

if __name__ == '__main__':
    main()
