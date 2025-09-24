# versión_simplificada.py
import numpy as np
import concurrent.futures
import time


def multiplicacion_simple_secuencial(A, B):
    """Versión muy simple para enseñanza"""
    m, n = A.shape
    n, p = B.shape
    C = np.zeros((m, p))

    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]

    return C


# --- Helper top-level (picklable) ---
def _calcular_filas_chunk(args):
    """
    args: (A_chunk, B, start, end)
    Retorna: (start, end, chunk_result)
    """
    A_chunk, B, start, end = args
    # Para docencia podrías hacerlo con bucles; aquí usamos producto matricial directo
    chunk_result = A_chunk @ B
    return (start, end, chunk_result)


def multiplicacion_simple_multiprocessing(A, B, num_workers=2):
    """Versión simple con multiprocessing para enseñanza"""
    m, n = A.shape
    n, p = B.shape
    C = np.zeros((m, p))

    # Evitar workers vacíos
    workers = max(1, min(num_workers, m))

    # Partir filas en bloques contiguos
    # Ej: m=10, workers=3 -> límites [0,3,7,10]
    bounds = np.linspace(0, m, workers + 1, dtype=int)

    tasks = []
    for i in range(workers):
        start, end = bounds[i], bounds[i + 1]
        if start < end:
            tasks.append((A[start:end], B, start, end))

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        for start, end, chunk_result in executor.map(_calcular_filas_chunk, tasks):
            C[start:end, :] = chunk_result

    return C


# Demo rápida
if __name__ == "__main__":
    # Matrices pequeñas para demo
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])

    print("Matriz A:")
    print(A)
    print("\nMatriz B:")
    print(B)

    # Secuencial
    C_sec = multiplicacion_simple_secuencial(A, B)
    print("\nResultado secuencial:")
    print(C_sec)

    # Multiprocessing
    C_mp = multiplicacion_simple_multiprocessing(A, B, num_workers=2)
    print("\nResultado multiprocessing:")
    print(C_mp)

    # Verificación
    print(f"\n¿Resultados iguales? {np.array_equal(C_sec, C_mp)}")