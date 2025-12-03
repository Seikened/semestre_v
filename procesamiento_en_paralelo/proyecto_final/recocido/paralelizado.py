import numpy as np
import math
from numba import njit, prange
from time import perf_counter

# ============================= FUNCIÓN OBJETIVO =============================



@njit
def fun_objetivo_jit(x, m=10):
    d = len(x)
    total = 0.0
    for i in prange(d):
        idx = i + 1
        xi = x[i]
        s1 = math.sin(xi)
        s2 = math.sin((idx * xi * xi) / math.pi)
        total += s1 * (s2 ** (2 * m))
    return -total

@njit
def solucion_vecina_jit(x, perturbacion, xmin, xmax):
    x_new = np.empty_like(x)
    d = len(x)
    for i in prange(d):
        paso = np.random.uniform(-perturbacion, perturbacion)
        val = x[i] + paso
        if val < xmin:
            val = xmin
        elif val > xmax:
            val = xmax
        x_new[i] = val
    return x_new


def recocido_simulado_core_jit(x_inicial,
                               T_inicial,
                               alpha,
                               iteraciones,
                               perturbacion,
                               xmin,
                               xmax,
                               T_min=1e-8):
    x_actual = x_inicial.copy()
    f_actual = fun_objetivo_jit(x_actual)
    mejor_x = x_actual.copy()
    mejor_f = f_actual
    T = T_inicial
    pasos_efectivos = 0

    tiempos = []
    for _ in prange(iteraciones):
        inicio = perf_counter()
        
        if T <= T_min:
            break

        x_vecina = solucion_vecina_jit(x_actual, perturbacion, xmin, xmax)
        f_vecina = fun_objetivo_jit(x_vecina)
        delta = f_vecina - f_actual

        if delta < 0:
            x_actual = x_vecina
            f_actual = f_vecina
        else:
            prob = math.exp(-delta / T)
            if np.random.rand() < prob:
                x_actual = x_vecina
                f_actual = f_vecina

        if f_actual < mejor_f:
            mejor_x = x_actual
            mejor_f = f_actual

        T *= alpha
        pasos_efectivos += 1
        
        fin = perf_counter()
        tiempos.append(fin - inicio)

    return mejor_x, mejor_f, pasos_efectivos, tiempos



# ============================= EJECUCIÓN (PARALELIZADO) =============================

def main_paralelizado(tem=1_000_000):
    """
    Versión base paralelizada
    Aquí medimos el tiempo total de la llamada al recocido.
    """
    # Aumentamos la dimensión para que haya más carga computacional
    d = 50                # antes 5, ahora 50 para que se note más el coste
    xmin = 0.0
    xmax = math.pi
    x0 = np.random.uniform(xmin, xmax, d)

    temperatura = tem
    alpha = 0.9995        # enfriamiento lento → más iteraciones efectivas
    iteraciones = 300_000 # muchas iteraciones para que se note la diferencia con Numba
    perturbacion = 0.2

    print("Punto inicial SA =", x0)
    print("f(x) inicial =", fun_objetivo_jit(x0))


    mejor_x, mejor_f, pasos, tiempos = recocido_simulado_core_jit(
        x0,
        temperatura,
        alpha,
        iteraciones,
        perturbacion,
        xmin,
        xmax
    )

    tiempo_total = sum(tiempos)

    print("============[ ✅ Paralelizado ]============")
    print(f"Pasos efectivos      : {pasos}")
    print(f"Tiempo transcurrido  : {tiempo_total:.4f} segundos")

    print("\n===== RESULTADOS SA =====")
    print("x =", mejor_x)
    print("f(x) =", mejor_f)
    return tiempo_total, tiempos


if __name__ == "__main__":
    temp = 100_000
    main_paralelizado(temp)
