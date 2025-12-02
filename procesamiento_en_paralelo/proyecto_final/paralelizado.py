import math
import random
import numpy as np
from time import perf_counter
from numba import njit, prange


# ============================================================
# 2. Función objetivo (Michalewicz general en d dimensiones)
#    Versión para evaluar TODA la población en paralelo
# ============================================================


@njit(parallel=True)
def michalewicz_poblacion(pop, m=10):
    """Evalúa la función de Michalewicz para toda la población.

    pop: arreglo de shape (N, d)
    regresa: arreglo de shape (N,) con los valores de fitness
    """
    N, d = pop.shape
    fitness = np.empty(N)
    for idx in prange(N):  # paralelizamos sobre individuos
        s = 0.0
        for j in range(d):
            xj = pop[idx, j]
            i = j + 1
            s += math.sin(xj) * (math.sin(i * xj * xj / math.pi) ** (2 * m))
        fitness[idx] = -s
    return fitness


# ============================================================
# 3. Utilidades para Evolución Diferencial
# ============================================================


def ini_pob(tam, d, xmin, xmax):
    """Genera población inicial como un arreglo NumPy (tam, d)."""
    return np.random.uniform(xmin, xmax, size=(tam, d))


def mutacion(poblacion, F, i):
    """Construye vector mutante para el individuo i.

    Nota: poblacion es un arreglo NumPy, pero aquí lo tratamos
    por índices; regresamos una lista que luego NumPy castea a fila.
    """
    n = len(poblacion)
    indices = list(range(n))
    indices.remove(i)
    r1, r2, r3 = random.sample(indices, 3)

    d = poblacion.shape[1]
    return [
        poblacion[r1][j] + F * (poblacion[r2][j] - poblacion[r3][j])
        for j in range(d)
    ]


def cruza(ind, mut, CR):
    """Construye el vector de prueba (trial vector)."""
    d = len(ind)
    j_rand = random.randint(0, d - 1)

    return [
        mut[j] if (random.random() < CR or j == j_rand) else ind[j]
        for j in range(d)
    ]


# ============================================================
# 4. Implementación de Evolución Diferencial
# ============================================================

# Parámetros del problema
d = 5
tam_pob = 50
xmin = 0.0
xmax = np.pi

# Parámetros del algoritmo DE
F = 0.8
CR = 0.9
generaciones = 10_000

# Población inicial como matriz NumPy
poblacion = ini_pob(tam_pob, d, xmin, xmax)

# Warmup: compilar Numba ANTES de medir el tiempo
_ = michalewicz_poblacion(poblacion[:5])

inicio = perf_counter()

# Bucle principal de DE
for g in range(generaciones):
    # Construimos todos los trial vectors primero
    trial_pop = np.empty_like(poblacion)

    for i in range(tam_pob):
        mut = mutacion(poblacion, F, i)
        vec_prueba = cruza(poblacion[i], mut, CR)
        trial_pop[i] = vec_prueba

    # Evaluamos población actual y trial en bloque usando Numba
    f_pob = michalewicz_poblacion(poblacion)
    f_trial = michalewicz_poblacion(trial_pop)

    # Selección vectorizada: si trial es mejor, lo sustituye
    mask = f_trial < f_pob
    poblacion[mask] = trial_pop[mask]

fin = perf_counter()

# ============================================================
# 5. Resultados
# ============================================================

fitness = michalewicz_poblacion(poblacion)
best_idx = np.argmin(fitness)
best_de = poblacion[best_idx]
best_de_f = fitness[best_idx]

print(f"Tiempo transcurrido [Paralelizado]: {fin - inicio:.4f} segundos")
print("\n===== MEJOR SOLUCIÓN DE =====")
print("x =", best_de)
print("f(x) =", best_de_f)
