import ast
import math
import os

import platform
import random
import time
from dataclasses import dataclass
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
        self.cache = {}

    def creae_individuo(self):
        """Crear un individuo (ruta) aleatorio"""
        individual = list(range(self.num_cities))
        random.shuffle(individual)
        return individual

    def crear_poblacion(self):
        """Crear población inicial"""
        return [self.creae_individuo() for _ in range(self.population_size)]

    def calcular_distancia(self, route):
        key = tuple(route)
        if key in self.cache:
            return self.cache.get(key)

        r = np.asarray(route, dtype=int)
        distancia = self.distance_matrix[r, np.roll(r, -1)].sum()
        self.cache[key] = distancia
        return distancia

    def fitness(self, individual):
        """Calcular fitness (inverso de la distancia)"""
        distance = self.calcular_distancia(individual)
        return (1 / distance if distance > 0 else float("inf"), distance)

    def rank_population(self, population):
        """Rankear población por fitness"""
        fitness_results = [
            (i, (self.fitness(individuo))) for i, individuo in enumerate(population)
        ]

        mejor_individuo = max(fitness_results, key=lambda x: x[1][0])

        return sorted(
            fitness_results, key=lambda x: x[1], reverse=True
        ), mejor_individuo[1][1]

    def selecion(self, population, ranked_population):
        """Selección por torneo"""
        selection_results = []

        # Elitismo: mantener los mejores individuos
        for i in range(self.elite_size):
            selection_results.append(population[ranked_population[i][0]])

        # Selección por torneo para el resto
        for _ in range(self.population_size - self.elite_size):
            tournament = random.sample(ranked_population, self.tournament_size)
            winner = max(tournament, key=lambda x: x[1])
            selection_results.append(population[winner[0]])

        return selection_results

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
        ranked_population, mejor_fitness = self.rank_population(population)

        if mejor_fitness < self.best_distance:
            self.best_distance = mejor_fitness
            self.best_solution = population[ranked_population[0][0]].copy()

        self.fitness_history.append(self.best_distance)

        # Selección
        selected = self.selecion(population, ranked_population)

        # Crear nueva generación
        children = []

        # Elitismo
        for i in range(self.elite_size):
            children.append(selected[i])

        # Crossover y mutación
        for i in range(self.elite_size, self.population_size):
            parent1, parent2 = random.sample(selected, 2)
            child = self.cruza(parent1, parent2)
            child = self.mutar(child)
            children.append(child)

        return children

    def paro_mejora(self, generacion):
        """Criterio de paro basado en mejora y solo checamos cada 50 generaciones"""

        distancia_actual = self.best_distance

        # Si no hay historial suficiente, base = distancia actual
        if len(self.fitness_history) < 51:
            base = distancia_actual
        else:
            base = self.fitness_history[-51]

        # Porcentaje de cambio (positivo si mejoró porque la distancia bajó)
        cambio = (base - distancia_actual) / max(base, 1e-12)

        if generacion % 50 == 0:
            if distancia_actual < base:
                emoji = "🔼"  # mejoró
            elif distancia_actual > base:
                emoji = "🔽"  # empeoró
            else:
                emoji = "➖"  # sin cambio

            print(
                f"\r\033[2KGen {generacion} | Distancia: {distancia_actual:.2f} | Cambio: {cambio:+.2%} {emoji}",
                end="",
                flush=True,
            )

            # === Aquí dejo tu misma lógica de paro, sin tocarla ===
            # if generacion >= self.generations // 1:
            #     if self.fitness_history[-1] == self.fitness_history[-50]:
            #         print("\nNo hay mejora en 50 generaciones, terminando...")
            #         return True

        return False

    def run(self):
        """Ejecutar el algoritmo genético"""

        # TODO: HACER QUE ESTA MATRIZ DE INDIVIDUOS SEA GLOBAL PARA QUE PUEDA SER ACCEDIDA POR OTROS
        inicio = time.perf_counter()
        population = self.crear_poblacion()
        fin = time.perf_counter()
        print(f"Tiempo de creación de población: {fin - inicio:.4f} segundos")

        print(f"Iniciando algoritmo genético para {self.num_cities} ciudades")
        print(f"Población: {self.population_size}, Generaciones: {self.generations}")

        tiempo_generacion = []
        for generation in range(self.generations):
            inicio = time.perf_counter()
            population = self.evolucionar_poblacion(population)
            fin = time.perf_counter()
            tiempo_generacion.append(fin - inicio)

            if self.paro_mejora(generation):
                break

        print(
            f"Tiempo promedio de generación: {sum(tiempo_generacion) / len(tiempo_generacion):.4f} segundos \n"
            f"Tiempo total: {sum(tiempo_generacion):.4f} segundos"
        )

        # Normalizar el tiempo para gráficar a mejor escala
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


# ================== Definición de Nucleo, Hilo y Director ==================
EstadoHilo = Literal["en_ejecucion", "completado", "esperando"]


@dataclass
class Nucleo:
    id: int
    tarea_actual: str
    disponible: bool = False


@dataclass
class Hilo:
    id: int
    nucleo_asignado: Nucleo
    tarea: str
    estado: EstadoHilo


@dataclass
class Director:
    nucleos_disponibles: int
    nucleos_usados: int
    nucleos: list[Nucleo]
    hilos: list[Hilo]


# ==================  ==================
class GetInfoSystem:
    def __init__(self) -> None:
        self.sistema = platform.system().lower()

    def obtener_nucleos_disponibles(self) -> int:
        return os.cpu_count() or 1

    def obtener_nucleos_usados(self) -> int | None:
        total = self.obtener_nucleos_disponibles()
        if hasattr(os, "getloadavg"):
            carga_1, carga_5, carga_15 = os.getloadavg()
            usados = int(math.ceil(carga_1))
            return max(0, min(usados, total))
        return None


# tiempo = 600
# for i in range(tiempo):
#     system = GetInfoSystem()
#     nucleos_disponibles = system.obtener_nucleos_disponibles()
#     nucleos_usados = system.obtener_nucleos_usados()

#     end_char = "\n" if i == tiempo - 1 else "\r"
#     print(
#         f"Iteración {i + 1:03d}/{tiempo}: | "
#         f"Nucleos disponibles: {nucleos_disponibles} | "
#         f"Nucleos usados: {nucleos_usados}",
#         end=end_char,
#         flush=True,
#     )
#     time.sleep(0.1)


def amortiguador_tamaños(num_cities: int, nivel_de_esfuerzo: int = 500):
    """
    Calcula parámetros adaptativos para inicializar un Algoritmo Genético en problemas tipo TSP.

    Parámetros:
      num_cities (int): número de ciudades de la instancia.
      nivel_de_esfuerzo (int): factor de cuántas evaluaciones totales hará el GA.
        Actúa como un "slider" de precisión vs. velocidad:
        - Valores bajos (200-400): búsqueda rápida pero menos exhaustiva.
        - Valores altos (600-800): búsqueda más completa pero tarda más.

    Qué hace:
      1. Calcula un "presupuesto" de evaluaciones totales proporcional al tamaño del problema.
      2. Ajusta el tamaño de la población con crecimiento √n·log₂(n), para balancear diversidad y costo.
      3. Deriva el número de generaciones a partir del presupuesto y el tamaño de población.
      4. Ajusta tasa de mutación, elitismo y torneo en función de la población.
      5. Devuelve todos los parámetros listos para instanciar el GA.

    Retorna:
      tuple(tamaño_poblacion, tasa_mutacion, tamaño_elite, generaciones, torneo)
    """

    # 1. Presupuesto de evaluaciones
    evaluaciones_totales = nivel_de_esfuerzo * num_cities * math.log2(num_cities + 1)

    # 2. Tamaño de población
    tamaño_poblacion = int(10 * math.sqrt(num_cities) * math.log2(num_cities))
    tamaño_poblacion = max(80, min(tamaño_poblacion, 1200))

    # 3. Número de generaciones
    generaciones = int(evaluaciones_totales / tamaño_poblacion)
    generaciones = max(150, min(generaciones, 10_000))

    # 4. Parámetros dependientes
    tasa_mutacion = min(0.25, max(0.01, 2.0 / num_cities))
    tamaño_elite = max(2, int(0.02 * tamaño_poblacion))
    torneo = max(3, min(int(0.02 * tamaño_poblacion), 7))

    print(
        f"{'=' * 60} \n"
        f"Configuración adaptativa -> \n"
        f"Población: {tamaño_poblacion} \n"
        f"Generaciones: {generaciones} \n"
        f"Mutación: {tasa_mutacion:.3f} \n"
        f"Elite: {tamaño_elite} \n"
        f"Torneo: {torneo} \n"
        f"{'=' * 60}"
    )

    return tamaño_poblacion, tasa_mutacion, tamaño_elite, generaciones, torneo


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

    # tamaño_poblacion, tasa_mutacion, tamaño_elite, generaciones, torneo = (
    #     amortiguador_tamaños(num_cities, nivel_de_esfuerzo=15_000)
    # )

    tamaño_poblacion = 654
    tasa_mutacion = 0.05
    tamaño_elite = 13
    generaciones = 5_000
    torneo = 7
    
    # 6. Finalmente, creamos el algoritmo genético con estos parámetros adaptativos.
    ga = TSPGeneticAlgorithm(
        distance_matrix=distance_matrix,
        population_size=tamaño_poblacion,
        mutation_rate=tasa_mutacion,
        elite_size=tamaño_elite,
        generations=generaciones,
        tournament_size=torneo,
    )

    best_solution, best_distance, fitness_history = ga.run()

    print(f"\nMejor ruta encontrada: {best_solution}")
    print(f"Distancia de referencia: {selected_instance['total_distance']}")
    print(f"Distancia encontrada: {best_distance}")
    print(f"Diferencia: {abs(best_distance - selected_instance['total_distance']):.2f}")

    # Visualizar resultados
    plot_solution(city_coordinates, best_solution, best_distance, instance_id)
    plot_convergence(fitness_history)


main()
