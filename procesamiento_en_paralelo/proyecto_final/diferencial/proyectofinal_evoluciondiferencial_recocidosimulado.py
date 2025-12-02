import math
import random
import numpy as np
from time import perf_counter
import os


# ============================================================
# 2. Función objetivo (Michalewicz general en d dimensiones)
# ============================================================

def michalewicz(x, m=10):
    x = np.array(x)
    d = len(x)
    i = np.arange(1, d + 1)
    return -np.sum(np.sin(x) * (np.sin((i * x**2) / np.pi))**(2 * m))


# ============================================================
# 3. Utilidades para Evolución Diferencial
# ============================================================

def ini_pob(tam, d, xmin, xmax):
    """Genera población inicial."""
    return [list(np.random.uniform(xmin, xmax, d)) for _ in range(tam)]


def mutacion(poblacion, F, i):
    """Construye vector mutante para el individuo i."""
    n = len(poblacion)
    indices = list(range(n))
    indices.remove(i)
    r1, r2, r3 = random.sample(indices, 3)

    return [
        poblacion[r1][j] + F * (poblacion[r2][j] - poblacion[r3][j])
        for j in range(len(poblacion[i]))
    ]


def cruza(ind, mut, CR):
    """Construye el vector de prueba (trial vector)."""
    d = len(ind)
    j_rand = random.randint(0, d - 1)

    return [
        mut[j] if (random.random() < CR or j == j_rand) else ind[j]
        for j in range(d)
    ]


def seleccion(ind, trial):
    """Selecciona entre individuo original y de prueba."""
    f1 = michalewicz(ind)
    f2 = michalewicz(trial)
    return trial if f2 < f1 else ind


# ============================================================
# 4. Implementación de Evolución Diferencial
# ============================================================
def main_no_paralelizado(n_gen=10_000)-> tuple[float, list[float]]:
    """Evolución Diferencial no paralelizada."""
    inicio = perf_counter()

    # Parámetros del problema
    d = 5
    tam_pob = 50
    xmin = 0.0
    xmax = np.pi

    # Parámetros del algoritmo DE
    F = 0.8
    CR = 0.9
    generaciones = n_gen

    # Población inicial
    poblacion = ini_pob(tam_pob, d, xmin, xmax)
    lista_tiempos = []
    # Bucle principal de DE
    for g in range(generaciones):
        tiempo_generacion_inicio = perf_counter()
        nueva_pob = []
        for i in range(tam_pob):
            mut = mutacion(poblacion, F, i)
            vec_prueba = cruza(poblacion[i], mut, CR)
            ind_nuevo = seleccion(poblacion[i], vec_prueba)
            nueva_pob.append(ind_nuevo)
        poblacion = nueva_pob
        lista_tiempos.append(perf_counter() - tiempo_generacion_inicio)

    fin = perf_counter()
    # ============================================================
    # 5. Resultados
    # ============================================================

    fitness = [michalewicz(p) for p in poblacion]
    best_idx = np.argmin(fitness)
    best_de = np.array(poblacion[best_idx])
    best_de_f = fitness[best_idx]

    print("============[ ❌ No Paralelizado ]============")
    print(f"Tiempo transcurrido : {fin - inicio:.4f} segundos")

    print("\n===== MEJOR SOLUCIÓN DE =====")
    print("x =", best_de)
    print("f(x) =", best_de_f)

    return fin - inicio, lista_tiempos


if __name__ == "__main__":
    main_no_paralelizado()