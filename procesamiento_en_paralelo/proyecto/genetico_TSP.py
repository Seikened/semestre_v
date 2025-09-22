import ast
import math
import os

import platform
import random
import time
from concurrent.futures import ThreadPoolExecutor
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
        self._pool = ThreadPoolExecutor()

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
            total += dmat[route[i]][route[i+1]]
        # cierre del ciclo
        total += dmat[route[-1]][route[0]]
        return total

    def fitness(self, individual):
        """Calcular fitness (inverso de la distancia)"""
        distance = self.calcular_distancia(individual)
        return (1 / distance if distance > 0 else float("inf"), distance)

    def rank_population(self, population, k=0):
        n = len(population)

        resultados = list(self._pool.map(self.fitness, population))
        #resultados = [self.fitness(ind) for ind in population]

        aptitud_arr   = np.fromiter((p[0] for p in resultados), dtype=float, count=n)
        distancia_arr = np.fromiter((p[1] for p in resultados), dtype=float, count=n)

        # Top-K por máximos de aptitud (1/dist)
        k = int(min(k, n))
        if k > 0:
            idx_k = np.argpartition(-aptitud_arr, k-1)[:k]           # bloque K, sin ordenar entre sí
            idx_elite = idx_k[np.argsort(-aptitud_arr[idx_k])]       # ahora sí, ordenamos solo K
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
        # Top-K (élite) + arrays para torneo; deja use_threads en None para la heurística
        idx_elite, mejor_dist_gen, aptitud_arr, distancia_arr = self.rank_population(population, k=self.elite_size)

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

    def paro_mejora(self, generacion):
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
            f"\r\033[2KGen {generacion} | Distancia: {distancia_actual:.2f} | Cambio: {cambio:+.2%} {emoji}",
            end="",
            flush=True,
        )

        # Tu lógica de paro (comentada) permanece intacta
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



    tamaño_poblacion = 2_000
    tasa_mutacion = 0.05
    tamaño_elite = 13
    generaciones = 20_000
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
    #plot_solution(city_coordinates, best_solution, best_distance, instance_id)
    #plot_convergence(fitness_history)


main()
