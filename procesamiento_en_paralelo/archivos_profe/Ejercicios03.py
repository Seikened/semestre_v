import numpy as np
import multiprocessing as mp
import time
from typing import Tuple

def fcn(i: int, j: int) -> float:
    """Función computacionalmente intensiva para evaluar en cada elemento"""
    result = 0.0
    for k in range(100):
        result += np.sin(i * 0.1 + j * 0.05 + k * 0.01) * np.cos(i * 0.05 - j * 0.1)
    return result

def process_column_chunk(args: Tuple[int, int, int]) -> Tuple[int, np.ndarray]:
    """Procesa un chunk de columnas y retorna los resultados"""
    start_col, col_count, n = args
    submatrix = np.zeros((n, col_count))
    
    print(f"Procesando columnas {start_col}-{start_col + col_count - 1}")
    
    for local_col in range(col_count):
        global_col = start_col + local_col
        for i in range(n):
            submatrix[i, local_col] = fcn(i, global_col)
    
    return start_col, submatrix

def main_multiprocessing():
    # Parámetros de la matriz
    n = 1000  # Tamaño de la matriz n x n
    num_processes = mp.cpu_count()
    
    print(f"Procesando matriz {n}x{n} con {num_processes} procesos")
    print("Esquema de distribución: bloques por columnas")
    
    start_time = time.time()
    
    # Distribuir el trabajo por columnas
    columns_per_process = n // num_processes
    remainder = n % num_processes
    
    # Preparar los argumentos para cada proceso
    tasks = []
    start_col = 0
    
    for i in range(num_processes):
        count = columns_per_process
        if i < remainder:
            count += 1
        
        tasks.append((start_col, count, n))
        start_col += count
    
    # Crear la matriz principal
    matrix = np.zeros((n, n))
    
    # Usar Pool para procesamiento paralelo
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(process_column_chunk, tasks)
    
    # Recolectar y organizar resultados
    for start_col, submatrix in results:
        col_count = submatrix.shape[1]
        for local_col in range(col_count):
            global_col = start_col + local_col
            matrix[:, global_col] = submatrix[:, local_col]
    
    end_time = time.time()
    print(f"Tiempo total de ejecución: {end_time - start_time:.4f} segundos")
    
    # Verificación
    print("Verificando algunos elementos...")
    for i in range(0, min(5, n), max(1, n//10)):
        for j in range(0, min(5, n), max(1, n//10)):
            expected = fcn(i, j)
            actual = matrix[i, j]
            if abs(expected - actual) > 1e-10:
                print(f"Error en ({i},{j}): esperado={expected}, actual={actual}")
    
    return matrix

if __name__ == "__main__":
    main_multiprocessing()