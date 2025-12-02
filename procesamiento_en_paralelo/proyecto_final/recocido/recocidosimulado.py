import numpy as np
import math
from time import perf_counter

# ============================= FUNCIÓN OBJETIVO =============================

def fun_objetivo(x, m=10):
    """
    Función objetivo tipo Rastrigin/oscillante.
    Esta versión usa loops explícitos para que sea más atractivo para Numba.
    """
    d = len(x)
    total = 0.0
    for i in range(d):
        idx = i + 1  # porque en tu versión usabas i desde 1 hasta d
        xi = x[i]
        s1 = math.sin(xi)
        s2 = math.sin((idx * xi * xi) / math.pi)
        total += s1 * (s2 ** (2 * m))
    return -total


# ============================= VECINO =============================

def solucion_vecina(x, perturbacion, xmin, xmax):
    """
    Genera una solución vecina con ruido uniforme y la recorta a [xmin, xmax].
    Esta función también es jitteable con Numba después.
    """
    # Copia para no modificar el vector original
    x_new = np.empty_like(x)
    d = len(x)
    for i in range(d):
        paso = np.random.uniform(-perturbacion, perturbacion)
        val = x[i] + paso
        # clip manual (más friendly para Numba que np.clip dentro de njit)
        if val < xmin:
            val = xmin
        elif val > xmax:
            val = xmax
        x_new[i] = val
    return x_new


# ============================= NÚCLEO DEL RECOCIDO =============================

def recocido_simulado_core(fun_objetivo,
                           x_inicial,
                           T_inicial,
                           alpha,
                           iteraciones,
                           perturbacion,
                           xmin,
                           xmax,
                           T_min=1e-8):
    """
    Núcleo del recocido simulado SIN medir tiempos.
    Esta función es la candidata perfecta para meterle @njit después.

    - No usa perf_counter
    - No usa random.random (solo np.random, que Numba soporta mejor)
    """
    x_actual = x_inicial.copy()
    f_actual = fun_objetivo(x_actual)
    mejor_x = x_actual.copy()
    mejor_f = f_actual
    T = T_inicial

    pasos_efectivos = 0
    tiempos = []
    for i in range(iteraciones):
        inicio = perf_counter()
        if T <= T_min:
            break

        # Generar vecina
        x_vecina = solucion_vecina(x_actual, perturbacion, xmin, xmax)
        f_vecina = fun_objetivo(x_vecina)
        delta = f_vecina - f_actual

        # Regla de aceptación
        if delta < 0:
            x_actual = x_vecina
            f_actual = f_vecina
        else:
            prob = math.exp(-delta / T)
            if np.random.rand() < prob:
                x_actual = x_vecina
                f_actual = f_vecina

        # Actualizar mejor solución
        if f_actual < mejor_f:
            mejor_x = x_actual
            mejor_f = f_actual

        # Enfriamiento
        T *= alpha
        pasos_efectivos += 1
        fin = perf_counter()
        tiempos.append(fin - inicio)

    return mejor_x, mejor_f, pasos_efectivos, tiempos


# ============================= EJECUCIÓN (NO PARALELIZADO) =============================

def main_no_paralelizado(tem=1_000_000):
    """
    Versión base no paralelizada y sin Numba.
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
    print("f(x) inicial =", fun_objetivo(x0))

    mejor_x, mejor_f, pasos, tiempos = recocido_simulado_core(
        fun_objetivo,
        x0,
        temperatura,
        alpha,
        iteraciones,
        perturbacion,
        xmin,
        xmax
    )

    tiempo_total = sum(tiempos)

    print("============[ ❌ No Paralelizado | Sin Numba ]============")
    print(f"Pasos efectivos      : {pasos}")
    print(f"Tiempo transcurrido  : {tiempo_total:.4f} segundos")

    print("\n===== RESULTADOS SA =====")
    print("x =", mejor_x)
    print("f(x) =", mejor_f)
    return tiempo_total, tiempos


if __name__ == "__main__":
    temp = 100_000
    main_no_paralelizado(temp)
