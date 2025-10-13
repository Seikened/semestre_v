import numpy as np
import concurrent.futures
import time
from typing import Tuple, List
import os

def fcn(i: int, j: int) -> float:
    """Función computacionalmente intensiva para evaluar en cada elemento"""
    result = 0.0
    for k in range(100):
        result += np.sin(i * 0.1 + j * 0.05 + k * 0.01) * np.cos(i * 0.05 - j * 0.1)
    return result

def process_column_range(args: Tuple[int, int, int]) -> Tuple[int, np.ndarray]:
    """Procesa un rango de columnas y retorna los resultados"""
    start_col, end_col, n = args
    col_count = end_col - start_col + 1
    submatrix = np.zeros((n, col_count))
    
    print(f"Procesando columnas {start_col}-{end_col} (PID: {os.getpid()})")
    
    for local_col in range(col_count):
        global_col = start_col + local_col
        for i in range(n):
            submatrix[i, local_col] = fcn(i, global_col)
    
    return start_col, submatrix

def main_concurrent_futures():
    # Parámetros de la matriz
    n = 1000  # Tamaño de la matriz n x n
    num_workers = os.cpu_count()
    
    print(f"Procesando matriz {n}x{n} con {num_workers} workers")
    print("Esquema de distribución: bloques por columnas")
    
    start_time = time.time()
    
    # Distribuir el trabajo por columnas
    columns_per_worker = n // num_workers
    remainder = n % num_workers
    
    # Preparar los rangos de columnas para cada worker
    tasks = []
    start_col = 0
    
    for i in range(num_workers):
        count = columns_per_worker
        if i < remainder:
            count += 1
        
        end_col = start_col + count - 1
        tasks.append((start_col, end_col, n))
        start_col += count
    
    # Crear la matriz principal
    matrix = np.zeros((n, n))
    
    # Usar ProcessPoolExecutor para procesamiento paralelo
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Enviar todas las tareas
        future_to_task = {
            executor.submit(process_column_range, task): task 
            for task in tasks
        }
        
        # Recolectar resultados conforme se completan
        for future in concurrent.futures.as_completed(future_to_task):
            try:
                start_col, submatrix = future.result()
                col_count = submatrix.shape[1]
                
                # Colocar los resultados en la matriz principal
                for local_col in range(col_count):
                    global_col = start_col + local_col
                    matrix[:, global_col] = submatrix[:, local_col]
                    
            except Exception as e:
                print(f"Error procesando tarea: {e}")
    
    end_time = time.time()
    print(f"Tiempo total de ejecución: {end_time - start_time:.4f} segundos")
    
    # Verificación rápida
    print("Verificación rápida de resultados...")
    test_points = [(0, 0), (n//2, n//2), (n-1, n-1)]
    for i, j in test_points:
        expected = fcn(i, j)
        actual = matrix[i, j]
        error = abs(expected - actual)
        print(f"Elemento ({i},{j}): error={error:.2e}")
    
    return matrix

# Versión alternativa con ThreadPoolExecutor (para operaciones I/O bound)
def process_element(args: Tuple[int, int]) -> Tuple[int, int, float]:
    """Procesa un solo elemento - útil para ThreadPoolExecutor"""
    i, j = args
    return i, j, fcn(i, j)

def main_threaded():
    """Versión que usa threads para demostración (no recomendada para CPU-bound)"""
    n = 500  # Matriz más pequeña para threads
    num_workers = os.cpu_count() * 2  # Más threads que CPUs
    
    print(f"Procesando matriz {n}x{n} con {num_workers} threads")
    
    start_time = time.time()
    matrix = np.zeros((n, n))
    
    # Generar todas las coordenadas
    coordinates = [(i, j) for i in range(n) for j in range(n)]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Procesar cada elemento individualmente
        future_to_coord = {
            executor.submit(process_element, coord): coord 
            for coord in coordinates
        }
        
        for future in concurrent.futures.as_completed(future_to_coord):
            try:
                i, j, value = future.result()
                matrix[i, j] = value
            except Exception as e:
                print(f"Error procesando elemento: {e}")
    
    end_time = time.time()
    print(f"Tiempo threaded: {end_time - start_time:.4f} segundos")
    return matrix

if __name__ == "__main__":   
    print("\n=== Versión con concurrent.futures (ProcessPool) ===")
    result2 = main_concurrent_futures()