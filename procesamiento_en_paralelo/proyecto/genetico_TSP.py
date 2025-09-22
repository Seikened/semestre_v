import ast
import os

# ---- Limitar hilos internos de BLAS/vecLib para evitar over-subscription ----
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import random
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ==== Globals para workers de procesos ====
DISTANCIA_MATRIZ = None


def iniciar_worker(dmat):
    """Se ejecuta al arrancar cada proceso: deja la matriz de distancias en memoria local del worker."""
    import numpy as _np

    global DISTANCIA_MATRIZ
    DISTANCIA_MATRIZ = _np.asarray(dmat)


def fitness_individual(individual):
    """(fit_inv, dist) usando la matriz global _DMAT (NumPy vectorizado, resultados idénticos a tu bucle)."""
    import numpy as _np

    idx = _np.asarray(individual, dtype=_np.int64)
    nxt = _np.roll(idx, -1)
    dist = float(DISTANCIA_MATRIZ[idx, nxt].sum())
    return (1.0 / dist if dist > 0 else float("inf"), dist)


def fitness_individual_local(dmat, individual):
    """Mismo cálculo vectorizado pero en el proceso principal (para benchmark secuencial limpio)."""
    idx = np.asarray(individual, dtype=np.int64)
    nxt = np.roll(idx, -1)
    dist = float(dmat[idx, nxt].sum())
    return (1.0 / dist if dist > 0 else float("inf"), dist)


"""
Python 3.12.9
Algoritmo Genético para el Problema del Viajante (TSP)
Autor: Fernando Leon Franco
Fecha: 20-Sep-2025
"""


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
        parallel_backend="process",  # "process" | "thread" | "none"
        max_workers=None,
        chunk_size=256,
        do_benchmark=True,
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
        self.backend_mode = parallel_backend
        self.max_workers = max_workers or max(1, (os.cpu_count() or 2) - 1) # Esto significa "uno menos que el total de CPUs"
        self.chunk_size: int = chunk_size
        self.do_benchmark = do_benchmark

        self._pool = None  # Thread o Process executor (según backend)

    def open_pool(self):
        """Crea el pool según backend (una sola vez)."""
        if self._pool is not None:
            return
        if self.backend_mode == "thread":
            self._pool = ThreadPoolExecutor(max_workers=self.max_workers)
            
        elif self.backend_mode == "process":
            self._pool = ProcessPoolExecutor(
                max_workers=self.max_workers,
                initializer=iniciar_worker,
                initargs=(self.distance_matrix,),
            )
        else:
            self._pool = None  # secuencial

    def _close_pool(self):
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None


    def _time_one_generation(self, population, *, backend, include_pool_startup=False):
        """
        Temporiza una sola generación completa con el backend indicado.
        Retorna (new_population, dt_seg).
        """
        import time as _t
        # Guardar estado actual
        old_backend = self.backend_mode
        had_pool_before = self._pool is not None

        # Cerrar pool actual para no contaminar la medida si cambiamos de backend
        self._close_pool()
        self.backend_mode = backend

        # Abrir pool si aplica (opcionalmente medimos también el arranque)
        t0 = _t.perf_counter()
        if self.backend_mode in ("thread", "process"):
            if include_pool_startup:
                self.open_pool()
            else:
                # warmup ligero para estabilizar kernels BLAS/vecLib sin contar startup
                self.open_pool()
                _ = self.rank_population_backend(population, k=0)
        t1 = _t.perf_counter()

        # Ejecutar UNA generación end-to-end
        pop_in = [ind[:] for ind in population]  # clon barato por justicia
        g0 = _t.perf_counter()
        pop_out = self.evolucionar_poblacion(pop_in)
        g1 = _t.perf_counter()

        # Cerrar pool temporal y restaurar configuración previa
        self._close_pool()
        self.backend_mode = old_backend
        # No restauramos el objeto del pool antiguo (ya fue cerrado). Si existía antes, lo re-creamos.
        if had_pool_before and self.backend_mode in ("thread", "process"):
            self.open_pool()

        # dt = sólo la parte de evolución; si quieres incluir startup, súmalo
        dt = (g1 - g0) if not include_pool_startup else ((t1 - t0) + (g1 - g0))
        return pop_out, dt

    def real_speedup_generation(self, population, *, repeats=2, include_pool_startup=False):
        """
        Mide SpeedUp de UNA generación real: dt_seq / dt_par
        - Usa el backend actual como 'par'
        - Usa backend='none' como 'seq'
        - Repite y promedia
        """
        import statistics as _st

        # Medir secuencial
        t_seq = []
        for _ in range(max(1, repeats)):
            _, dt = self._time_one_generation(population, backend="none", include_pool_startup=False)
            t_seq.append(dt)

        # Medir paralelo (backend actual)
        t_par = []
        for _ in range(max(1, repeats)):
            _, dt = self._time_one_generation(
                population,
                backend=self.backend_mode,
                include_pool_startup=include_pool_startup
            )
            t_par.append(dt)

        m_seq = _st.mean(t_seq)
        m_par = _st.mean(t_par)
        speedup = (m_seq / m_par) if m_par > 0 else float("inf")
        return m_seq, m_par, speedup

    def hacer_benchmark(self, population):
        try:
            print(f"{'='*50}")
            print("\n[Benchmark] Evaluando SpeedUp real de UNA generación (paralelo vs secuencial):")
            print(f"  Backend actual: {self.backend_mode} | max_workers={self.max_workers} | chunk_size={self.chunk_size}")
            print("  - t_seq_gen: tiempo secuencial puro (sin paralelismo)")
            print("  - t_par_gen: tiempo paralelo (con backend seleccionado)")
            print("  - SpeedUp: t_seq_gen / t_par_gen\n")

            # 1) sin contar startup del pool (steady-state)
            seq, par, su = self.real_speedup_generation(population, repeats=2, include_pool_startup=False)
            print(f"  [Steady-State] t_seq_gen = {seq:.4f} s | t_par_gen = {par:.4f} s | SpeedUp = {su:.2f}x")

            # 2) incluyendo el costo de arranque del pool (más conservador)
            seq2, par2, su2 = self.real_speedup_generation(population, repeats=1, include_pool_startup=True)
            print(f"  [Incl. Startup] t_seq_gen = {seq2:.4f} s | t_par_gen = {par2:.4f} s | SpeedUp = {su2:.2f}x\n")
            print(f"{'='*50}")
        except Exception as e:
            print(f"[Benchmark] Benchmark omitido por error: {e}\n")
            print(f"{'='*50}")
            # Asegurar que el pool principal quede abierto tras los benchmarks
            if self.backend_mode in ("thread", "process"):
                self.open_pool()


    def crear_individuo(self):
        """Crear un individuo (ruta) aleatorio"""
        individual = list(range(self.num_cities))
        random.shuffle(individual)
        return individual

    def crear_poblacion(self):
        """Crear población inicial"""
        return [self.crear_individuo() for _ in range(self.population_size)]


    def calcular_distancia(self, route):
        dmat = self.distance_matrix
        total = 0.0
        # tramo 0..n-2
        for i in range(len(route) - 1):
            total += dmat[route[i]][route[i + 1]]
        # cierre del ciclo
        total += dmat[route[-1]][route[0]]
        return total

    def fitness(self, individual):
        """Calcular fitness (inverso de la distancia)"""
        distance = self.calcular_distancia(individual)
        return (1 / distance if distance > 0 else float("inf"), distance)

    def rank_population_backend(self, population, k=0):
        """
        Evalúa la población con el backend elegido y devuelve:
        (idx_elite, mejor_distancia, aptitud_arr, distancia_arr)
        """
        n = len(population)

        match self.backend_mode:
            case "thread":
                # Hilos: GIL aplica; cada worker usa tu bucle puro (no vectorizado)
                results = list(self._pool.map(self.fitness, population)) # type: ignore
            case "process":
                # Procesos: GIL no aplica; cada worker usa NumPy vectorizado
                results = list(self._pool.map(fitness_individual, population, chunksize=self.chunk_size)) # type: ignore
            case _:
                # Secuencial rápido (vectorizado por individuo; baseline para benchmark)
                dmat = self.distance_matrix
                results = [fitness_individual_local(dmat, ind) for ind in population]

        aptitud_arr = np.fromiter((p[0] for p in results), dtype=float, count=n)
        distancia_arr = np.fromiter((p[1] for p in results), dtype=float, count=n)

        
        # Top-K por máximos de aptitud (1/dist)
        k = int(min(k, n))
        if k > 0:
            idx_k = np.argpartition(-aptitud_arr, k - 1)[:k]
            idx_elite = idx_k[np.argsort(-aptitud_arr[idx_k])]
        else:
            idx_elite = np.empty((0,), dtype=int)

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



    def cruza(self, parent1, parent2):
        """Crossover OX (Order Crossover)"""
        child = [-1] * self.num_cities

        # Seleccionar segmento aleatorio
        start_pos = random.randint(0, self.num_cities - 1)
        end_pos = random.randint(0, self.num_cities - 1)

        if start_pos > end_pos:
            start_pos, end_pos = end_pos, start_pos

        # Copiar segmento del parent1 al child
        for i in range(start_pos, end_pos + 1):
            child[i] = parent1[i]

        # Completar con genes del parent2
        current_pos = (end_pos + 1) % self.num_cities
        for gene in parent2:
            if gene not in child:
                child[current_pos] = gene
                current_pos = (current_pos + 1) % self.num_cities

        return child

    def mutar(self, individual):
        """Mutación por intercambio"""
        if random.random() < self.mutation_rate:
            idx1, idx2 = random.sample(range(self.num_cities), 2)
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
        return individual

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
        children = []
        for i in range(self.elite_size):
            children.append(selected[i])
        for i in range(self.elite_size, self.population_size):
            p1, p2 = random.sample(selected, 2)
            child = self.cruza(p1, p2)
            child = self.mutar(child)
            children.append(child)
        return children

    def paro_mejora(self, generacion, tiempo_ultima_gen):
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

        print(
            f"\r\033[2KGen {generacion} | Tiempo: {tiempo_ultima_gen:.4f}s | Distancia: {distancia_actual:.2f} | Cambio: {cambio:+.2%} {emoji}",
            end="",
            flush=True,
        )

        # Tu lógica de paro (comentada) permanece intacta
        return False

    def run(self):
        """Ejecutar el algoritmo genético"""
        inicio = time.perf_counter()
        population = self.crear_poblacion()
        fin = time.perf_counter()
        print(f"Tiempo de creación de población: {fin - inicio:.4f} segundos")

        print(f"Iniciando algoritmo genético para {self.num_cities} ciudades")
        print(f"Población: {self.population_size}, Generaciones: {self.generations}")
        print(
            f"Backend paralelo: {self.backend_mode} | max_workers={self.max_workers} | chunk_size={self.chunk_size}"
        )
        print(
            f"NUM_THREADS internos: OMP={os.environ.get('OMP_NUM_THREADS')} OPENBLAS={os.environ.get('OPENBLAS_NUM_THREADS')} VECLIB={os.environ.get('VECLIB_MAXIMUM_THREADS')}"
        )

        # Abrir pool si aplica
        if self.backend_mode in ("thread", "process"):
            self.open_pool()


       #  Benchmark opcional
        if self.do_benchmark:
            self.hacer_benchmark(population)

        tiempo_generacion = []
        try:
            for generation in range(self.generations):
                inicio = time.perf_counter()
                population = self.evolucionar_poblacion(population)
                fin = time.perf_counter()
                tiempo_generacion.append(fin - inicio)

                ultima_tiempo_gen = tiempo_generacion[-1]
                if self.paro_mejora(generation, ultima_tiempo_gen):
                    break
        finally:
            self._close_pool()

        print(
            f"Tiempo promedio de generación: {sum(tiempo_generacion) / len(tiempo_generacion):.4f} segundos \n"
            f"Tiempo total: {sum(tiempo_generacion):.4f} segundos"
        )

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

    # Mostrar instancias disponibles
    print("Instancias disponibles:")
    for i, row in df.iterrows():
        print(f"{i}: {row['num_cities']} ciudades (ID: {row['instance_id']})")

    # Seleccionar instancia
    # instance_choice = int(input("\nSeleccione el número de instancia a ejecutar: "))
    instance_choice = 2

    selected_instance = df.iloc[instance_choice]

    # Parsear datos
    distance_matrix = ast.literal_eval(selected_instance["distance_matrix"])
    city_coordinates = ast.literal_eval(selected_instance["city_coordinates"])
    num_cities = selected_instance["num_cities"]
    instance_id = selected_instance["instance_id"]

    print(f"\nEjecutando instancia {instance_id} con {num_cities} ciudades")
    print(f"Distancia total de referencia: {selected_instance['total_distance']}")

    tamaño_poblacion = 20_000
    tasa_mutacion = 0.05
    tamaño_elite = 20
    generaciones = 10_000
    torneo = 10

    # 6. Finalmente, creamos el algoritmo genético con estos parámetros adaptativos.
    ga = TSPGeneticAlgorithm(
        distance_matrix=distance_matrix,
        population_size=tamaño_poblacion,
        mutation_rate=tasa_mutacion,
        elite_size=tamaño_elite,
        generations=generaciones,
        tournament_size=torneo,
        # --- NUEVO: config de paralelismo ---
        parallel_backend="process",  # "process" recomendado para CPU-bound; prueba "thread" y "none"
        max_workers=min(6, (os.cpu_count() or 2)),  # ajusta 4–6 en el M2 Air
        chunk_size=256,
        do_benchmark=True,
    )

    best_solution, best_distance, fitness_history = ga.run()

    print(f"\nMejor ruta encontrada: {best_solution}")
    print(f"Distancia de referencia: {selected_instance['total_distance']}")
    print(f"Distancia encontrada: {best_distance}")
    print(f"Diferencia: {abs(best_distance - selected_instance['total_distance']):.2f}")

    # Visualizar resultados
    # plot_solution(city_coordinates, best_solution, best_distance, instance_id)
    # plot_convergence(fitness_history)


if __name__ == "__main__":
    import multiprocessing as mp

    try:
        mp.set_start_method("spawn", force=False)
    except RuntimeError:
        pass
    main()
