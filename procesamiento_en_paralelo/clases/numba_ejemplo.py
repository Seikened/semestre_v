from time import perf_counter
import numpy as np
import math
import os
from numba import jit, prange, get_num_threads, set_num_threads

USE_FLOAT32 = True
DTYPE = np.float64 if USE_FLOAT32 else np.float64

# === Configura hilos de Numba aquí ===
# Prueba primero con 1, luego con 8 (o los que tengas)
nucleos = 8
print(f"Nucleos usados: {nucleos} / {os.cpu_count()}")
set_num_threads(nucleos)   # <-- pon 1 para ver Numba en single-thread

# set_num_threads(8) # <-- cambia a 8 para ver Numba multi-thread

print("Jit(nopython=True, parallel=True, cache=True, fastmath=True)")
print(f"Numba threads: {get_num_threads()} | dtype: {DTYPE}")

@jit(nopython=True, parallel=True, cache=True, fastmath=True)
def calcular_senios_paralelos(angulos, out):
    n = angulos.size
    for i in prange(n):
        out[i] = math.sin(angulos[i])

def warm_up():
    ang_small = np.linspace(0, 2 * np.pi, 10_000, dtype=DTYPE)
    out_small = np.empty_like(ang_small)
    calcular_senios_paralelos(ang_small, out_small)

def calcular_senios_numba(angulos, out):
    inicio = perf_counter()
    calcular_senios_paralelos(angulos, out)
    fin = perf_counter()
    t = fin - inicio
    print(f"Tiempo de ejecución (Numba prange): {t:.6f} s")
    return t

def calcular_senios_numpy(angulos, out):
    inicio = perf_counter()
    np.sin(angulos, out=out)
    fin = perf_counter()
    t = fin - inicio
    print(f"Tiempo de ejecución (NumPy out=): {t:.6f} s")
    return t

def senos(numeros, angulos, out, impl="numba"):
    tiempos = []
    for _ in range(numeros):
        if impl == "numba":
            tiempos.append(calcular_senios_numba(angulos, out))
        else:
            tiempos.append(calcular_senios_numpy(angulos, out))
    return tiempos

# Calentamiento (compilación JIT)
inicio = perf_counter()
warm_up()
fin = perf_counter()
print(f"Tiempo de compilación de calentamiento: {fin - inicio:.6f} s")

# Datos grandes
N = 400_000_000   # baja de 400M si no cabe en RAM
angulos = np.linspace(0, 2 * np.pi, N, dtype=DTYPE)
out = np.empty_like(angulos)
repeticiones = 10
# Corre Numba
resultados_numba = senos(repeticiones, angulos, out, impl="numba")
print(f"Tiempo total Numba: {sum(resultados_numba):.6f} s")

# Corre NumPy
resultados_numpy = senos(repeticiones, angulos, out, impl="numpy")
print(f"Tiempo total NumPy: {sum(resultados_numpy):.6f} s")