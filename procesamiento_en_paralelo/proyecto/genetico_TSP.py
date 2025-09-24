import ast
from curses.panel import panel
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

"""
Python 3.12.9
Algoritmo Genético para el Problema del Viajante (TSP)
Autor: Fernando Leon Franco
Fecha: 20-Sep-2025
"""

# ---- Limitar hilos internos de BLAS/vecLib para evitar over-subscription ----
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")



# ==== Globals para workers de procesos ====
DISTANCIA_MATRIZ = None


def iniciar_worker(dmat):
    """Se ejecuta al arrancar cada proceso: deja la matriz de distancias en memoria local del worker."""
    global DISTANCIA_MATRIZ
    DISTANCIA_MATRIZ = np.asarray(dmat)


def fitness_individual(individual):
    """(fit_inv, dist) usando la matriz global DISTANCIA_MATRIZ (NumPy vectorizado, resultados idénticos a tu bucle)."""

    idx = np.asarray(individual, dtype=np.int64)
    nxt = np.roll(idx, -1)
    dist = float(DISTANCIA_MATRIZ[idx, nxt].sum())
    return (1.0 / dist if dist > 0 else float("inf"), dist)



def cruza(padre1, padre2, numero_ciudades, rnd):
    """ Cruza de orden (OX) para TSP. """
    hijo = [-1] * numero_ciudades

    a = rnd.randint(0, numero_ciudades - 1)
    b = rnd.randint(0, numero_ciudades - 1)
    if a > b:
        a, b = b, a

    visited = [False] * numero_ciudades
    for i in range(a, b + 1):
        g = padre1[i]
        hijo[i] = g
        visited[g] = True

    pos = (b + 1) % numero_ciudades
    for g in padre2:
        if not visited[g]:
            hijo[pos] = g
            visited[g] = True
            pos = (pos + 1) % numero_ciudades

    return hijo


def mutar(hijo, tasa_muta, rnd):
    """Mutación por intercambio (swap) con una probabilidad dada."""
    if rnd.random() < tasa_muta:
        i, j = rnd.sample(range(len(hijo)), 2)
        hijo[i], hijo[j] = hijo[j], hijo[i]
    return hijo


def cruza_mutar(argumentos_empaquetados):
    """
    Trabajador simple: (padre1, padre2, numero de ciudades, tasa de mutación, seed) -> child
    Top-level para funcionar con 'spawn' y ProcessPoolExecutor.
    """
    import random

    padre1, padre2, num_ciudades, tasa_de_mutacion, seed = argumentos_empaquetados
    rnd = random.Random(seed)
    child = cruza(padre1, padre2, num_ciudades, rnd)
    return mutar(child, tasa_de_mutacion, rnd)





class TSPGeneticAlgorithm:
    def __init__(
        self,
        distance_matrix,
        population_size=100,
        mutation_rate=0.01,
        elite_size=20,
        generations=500,
        tournament_size=5,
        *,
        max_workers=None,
        chunk_size=256,
    ):
        self.distance_matrix = np.array(distance_matrix)
        self.num_cities = len(distance_matrix)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.best_solution = None
        self.best_distance = float("inf")
        self.fitness_history = []

        # --- paralelismo configurable ---
        self.max_workers = max_workers or max(
            1, (os.cpu_count() or 2) - 1
        )  # Esto significa "uno menos que el total de CPUs"
        self.chunk_size: int = chunk_size

        self._pool = None  # Thread o Process executor (según backend)

    def open_pool(self):
        """Crea el pool según backend (una sola vez)."""
        
        self._pool = ProcessPoolExecutor(
            max_workers=self.max_workers,
            initializer=iniciar_worker,
            initargs=(self.distance_matrix,),
        )
        

    def close_pool(self):
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None


    def crear_individuo(self):
        """Crear un individuo (ruta) aleatorio"""
        individual = list(range(self.num_cities))
        random.shuffle(individual)
        return individual

    def crear_poblacion(self):
        """Crear población inicial"""
        return [self.crear_individuo() for _ in range(self.population_size)]



    def rank_population_backend(self, population, k=0):
        """
        Evalúa la población con el backend elegido y devuelve:
        (idx_elite, mejor_distancia, aptitud_arr, distancia_arr)
        """
        n = len(population)


        results = list(
            self._pool.map(
                fitness_individual, population, chunksize=self.chunk_size
            )
        )  # type: ignore


        aptitud_arr = np.fromiter((p[0] for p in results), dtype=float, count=n)
        distancia_arr = np.fromiter((p[1] for p in results), dtype=float, count=n)

        # Top-K por máximos de aptitud (1/dist)
        k = int(min(k, n))

        idx_k = np.argpartition(-aptitud_arr, k - 1)[:k]
        idx_elite = idx_k[np.argsort(-aptitud_arr[idx_k])]

        mejor_distancia = float(distancia_arr[int(np.argmax(aptitud_arr))])
        return idx_elite, mejor_distancia, aptitud_arr, distancia_arr



    def selecion_fast(self, population, idx_elite, aptitud_arr):
        """Elitismo + torneo leyendo aptitudes por índice (sin lista ordenada completa)."""
        seleccion = [population[int(i)] for i in idx_elite]  # élite ya ordenada
        n = len(population)
        while len(seleccion) < self.population_size:
            cand = random.sample(range(n), self.tournament_size)
            ganador = max(cand, key=lambda i: aptitud_arr[i])
            seleccion.append(population[ganador])
        return seleccion

    # ====== WORKERS DE CRUZA/MUTACIÓN (TOP-LEVEL) ======



    def evolucionar_poblacion(self, population):
        """Evolucionar la población"""
        # Top-K (élite) + arrays para torneo; deja use_threads en None para la heurística
        idx_elite, mejor_dist_gen, aptitud_arr, distancia_arr = (
            self.rank_population_backend(population, k=self.elite_size)
        )

        # Mejor de la generación sin recomputar
        if mejor_dist_gen < self.best_distance:
            self.best_distance = mejor_dist_gen
            self.best_solution = population[int(np.argmin(distancia_arr))].copy()

        self.fitness_history.append(self.best_distance)

        # Selección rápida por índice
        selected = self.selecion_fast(population, idx_elite, aptitud_arr)

        # Nueva generación (igual que tenías)
        # Nueva generación: mantenemos élite directa
        hijo = [selected[i] for i in range(self.elite_size)]

        # Generar descendientes
        num_descendientes = self.population_size - self.elite_size
        
        # Prepara pares de padres y semillas independientes por tarea
        parent_pairs = [tuple(random.sample(selected, 2)) for _ in range(num_descendientes)]
        seeds = [random.getrandbits(64) for _ in range(num_descendientes)]

        # Iterador de argumentos para el worker
        argumentos_empaquetados = ((padre1, padre2, self.num_cities, self.mutation_rate, seed) for (padre1, padre2), seed in zip(parent_pairs, seeds))

        # Map paralelo
        descendencia = list(self._pool.map(cruza_mutar, argumentos_empaquetados, chunksize=(self.chunk_size))) # type: ignore
        hijo.extend(descendencia)
        return hijo




    def paro_mejora(self, generacion, tiempo_generacion):
        """Criterio de paro basado en mejora; solo evaluamos e imprimimos cada 50 generaciones."""
        if generacion % 50 != 0:
            return False  # salida rápida: no hacemos cálculos ni formateo

        distancia_actual = self.best_distance

        # Base: si aún no acumulamos 51 puntos, compara contra la distancia actual (mismo comportamiento)
        if len(self.fitness_history) < 51:
            base = distancia_actual
        else:
            base = self.fitness_history[-51]

        # Porcentaje de cambio: positivo si mejoró (distancia bajó)
        cambio = (base - distancia_actual) / max(base, 1e-12)

        if distancia_actual < base:
            emoji = "🔼"
        elif distancia_actual > base:
            emoji = "🔽"
        else:
            emoji = "➖"

        tiempo_transcurrido = sum(tiempo_generacion)/60
        promedio_tiempo = sum(tiempo_generacion) / len(tiempo_generacion) if tiempo_generacion else 0.0
        tiempo_restante = (promedio_tiempo * (self.generations - generacion))/60
        tiempo_aproximado_total = (promedio_tiempo * self.generations)/60
        tiempo_simulado_10k = (promedio_tiempo * 10_000)/60
        tiempo_simulado_20k = (promedio_tiempo * 20_000)/60
        

        # Construye el texto del panel (5 líneas)
        panel = (
            f"{'=' * 70}\n"
            f"Gen {generacion}/{self.generations} ({generacion/self.generations:.1%}) | "
            f"📏 Distancia: {distancia_actual:.2f} | {emoji} {cambio:+.2%}\n"
            f"Tiempo restante: {tiempo_restante:.2f} min | "
            f"Tiempo promedio: {promedio_tiempo:.4f}s\n"
            f"Simulado: {tiempo_transcurrido:.2f}/{(tiempo_aproximado_total):.2f} min | "
            f"10k={(tiempo_simulado_10k):.2f} min | 20k={(tiempo_simulado_20k):.2f} min\n"
            f"{'=' * 70}"
        )

        # N es 5
        N = 5

        # 1) Vuelve al ancla
        print("\033[u", end="")  # restore cursor (ESC[u)

        # 2) Limpia exactamente N líneas, bajando una a una
        print(("\033[2K\033[1B" * N), end="")  # clear line + move down
        # 3) Sube N para regresar al inicio del panel
        print(f"\033[{N}A", end="")

        # 4) Dibuja el panel
        print(panel, end="", flush=True)
        


    def run(self):
        """Ejecutar el algoritmo genético"""
        inicio = time.perf_counter()
        population = self.crear_poblacion()
        fin = time.perf_counter()
        print(f"Tiempo de creación de población: {fin - inicio:.4f} segundos")

        print(f"Iniciando algoritmo genético para {self.num_cities} ciudades")
        print(f"Población: {self.population_size}, Generaciones: {self.generations}")
        print(
            f"Backend paralelo: Nucleos | max_workers={self.max_workers} | chunk_size={self.chunk_size}"
        )
        print(
            f"NUM_THREADS internos: OMP={os.environ.get('OMP_NUM_THREADS')} OPENBLAS={os.environ.get('OPENBLAS_NUM_THREADS')} VECLIB={os.environ.get('VECLIB_MAXIMUM_THREADS')}"
        )

        # Abrir pool
        self.open_pool()

        # === Ancla del panel: reserva 5 líneas y guarda la posición ===
        N = 5  # alto del panel: sep + 3 líneas + sep
        print("\n" * N, end="")          # crea el hueco (no lo pongas dentro del loop)
        print(f"\033[{N}F\033[s", end="")  # sube N líneas y guarda cursor (ESC[s)
        tiempo_generacion = []

        for generation in range(self.generations):
            inicio = time.perf_counter()
            population = self.evolucionar_poblacion(population)
            fin = time.perf_counter()
            tiempo_generacion.append(fin - inicio)

            self.paro_mejora(generation, tiempo_generacion)


        self.close_pool()

        print(
            f"\nTiempo promedio de generación: {sum(tiempo_generacion) / len(tiempo_generacion):.4f} segundos | "
            f"Tiempo total: {sum(tiempo_generacion):.4f} segundos"
        )
        
        # Visualización del tiempo por generación (normalizado)
        t = np.array(tiempo_generacion, dtype=float)
        t_norm = (t - t.min()) / (t.max() - t.min() + 1e-12)

        plt.plot(t_norm)
        plt.title("Tiempo por generación")
        plt.xlabel("Generación")
        plt.ylabel("Tiempo (segundos)")
        plt.show()

        print(f"Mejor solución encontrada: Distancia = {self.best_distance:.2f}")
        return self.best_solution, self.best_distance, self.fitness_history


# ================== Funciones de Visualización ==================


def plot_solution(coordinates, solution, distance, instance_id):
    """Visualizar la solución"""
    coordinates = np.array(coordinates)
    solution_coords = coordinates[solution]
    solution_coords = np.vstack(
        [solution_coords, solution_coords[0]]
    )  # Cerrar el ciclo

    plt.figure(figsize=(12, 6))

    # Plot ciudades
    plt.subplot(1, 2, 1)
    plt.scatter(coordinates[:, 0], coordinates[:, 1], c="blue", s=50)
    for i, (x, y) in enumerate(coordinates):
        plt.annotate(str(i), (x, y), xytext=(5, 5), textcoords="offset points")

    # Plot ruta
    plt.subplot(1, 2, 2)
    plt.plot(solution_coords[:, 0], solution_coords[:, 1], "o-", markersize=8)
    plt.scatter(coordinates[:, 0], coordinates[:, 1], c="red", s=50)

    for i, (x, y) in enumerate(coordinates):
        plt.annotate(str(i), (x, y), xytext=(5, 5), textcoords="offset points")

    plt.suptitle(f"Instancia {instance_id} - Distancia: {distance:.2f}")
    plt.tight_layout()
    plt.show()


def plot_convergence(fitness_history):
    """Visualizar convergencia del algoritmo"""
    plt.figure(figsize=(10, 6))
    plt.plot(fitness_history)
    plt.title("Convergencia del Algoritmo Genético")
    plt.xlabel("Generación")
    plt.ylabel("Mejor Distancia")
    plt.grid(True)
    plt.show()


# ================== Función Principal ==================
def main():
    # Cargar datos
    file_path = "/Users/ferleon/Github/semestre_v/procesamiento_en_paralelo/proyecto/tsp_dataset - copia.csv"
    df = pd.read_csv(file_path)

    # Seleccionar instancia
    instance_choice = 2

    selected_instance = df.iloc[instance_choice]

    # Parsear datos
    distance_matrix = ast.literal_eval(selected_instance["distance_matrix"])
    city_coordinates = ast.literal_eval(selected_instance["city_coordinates"])
    num_cities = selected_instance["num_cities"]
    instance_id = selected_instance["instance_id"]

    print(f"\nEjecutando instancia {instance_id} con {num_cities} ciudades")
    print(f"Distancia total de referencia: {selected_instance['total_distance']}\n")

    tamaño_poblacion = 100_000
    tasa_mutacion = 0.05
    tamaño_elite = 20
    generaciones = 300
    torneo = 10

    # 6. Finalmente, creamos el algoritmo genético con estos parámetros adaptativos.
    ga = TSPGeneticAlgorithm(
        distance_matrix=distance_matrix,
        population_size=tamaño_poblacion,
        mutation_rate=tasa_mutacion,
        elite_size=tamaño_elite,
        generations=generaciones,
        tournament_size=torneo,
        max_workers=min(4, (os.cpu_count() or 2)),  # ajusta 4–6 en el M2 Air
        chunk_size=256,
    )

    best_solution, best_distance, fitness_history = ga.run()

    print(f"\nMejor ruta encontrada: {best_solution}")
    print(f"Distancia de referencia: {selected_instance['total_distance']}")
    print(f"Distancia encontrada: {best_distance}")
    print(f"Diferencia: {abs(best_distance - selected_instance['total_distance']):.2f}")

    # Visualizar resultados
    plot_solution(city_coordinates, best_solution, best_distance, instance_id)
    plot_convergence(fitness_history)


if __name__ == "__main__":

    main()
