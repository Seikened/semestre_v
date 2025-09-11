import numpy as np
import pandas as pd
import ast
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class TSPGeneticAlgorithm:
    def __init__(self, distance_matrix, population_size=100, mutation_rate=0.01, 
                 elite_size=20, generations=500, tournament_size=5):
        self.distance_matrix = np.array(distance_matrix)
        self.num_cities = len(distance_matrix)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.best_solution = None
        self.best_distance = float('inf')
        self.fitness_history = []
        
    def create_individual(self):
        """Crear un individuo (ruta) aleatorio"""
        individual = list(range(self.num_cities))
        random.shuffle(individual)
        return individual
    
    def create_population(self):
        """Crear población inicial"""
        return [self.create_individual() for _ in range(self.population_size)]
    
    def calculate_distance(self, individual):
        """Calcular la distancia total de una ruta"""
        total_distance = 0
        for i in range(len(individual)):
            from_city = individual[i]
            to_city = individual[(i + 1) % len(individual)]
            total_distance += self.distance_matrix[from_city][to_city]
        return total_distance
    
    def calculate_fitness(self, individual):
        """Calcular fitness (inverso de la distancia)"""
        distance = self.calculate_distance(individual)
        return 1 / distance if distance > 0 else float('inf')
    
    def rank_population(self, population):
        """Rankear población por fitness"""
        fitness_results = [(i, self.calculate_fitness(ind)) for i, ind in enumerate(population)]
        return sorted(fitness_results, key=lambda x: x[1], reverse=True)
    
    def selection(self, population, ranked_population):
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
    
    def crossover(self, parent1, parent2):
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
    
    def mutate(self, individual):
        """Mutación por intercambio"""
        if random.random() < self.mutation_rate:
            idx1, idx2 = random.sample(range(self.num_cities), 2)
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
        return individual
    
    def evolve_population(self, population):
        """Evolucionar la población"""
        ranked_population = self.rank_population(population)
        
        # Guardar el mejor de esta generación
        best_idx = ranked_population[0][0]
        best_distance = self.calculate_distance(population[best_idx])
        
        if best_distance < self.best_distance:
            self.best_distance = best_distance
            self.best_solution = population[best_idx].copy()
        
        self.fitness_history.append(self.best_distance)
        
        # Selección
        selected = self.selection(population, ranked_population)
        
        # Crear nueva generación
        children = []
        
        # Elitismo
        for i in range(self.elite_size):
            children.append(selected[i])
        
        # Crossover y mutación
        for i in range(self.elite_size, self.population_size):
            parent1, parent2 = random.sample(selected, 2)
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            children.append(child)
        
        return children
    
    def run(self):
        """Ejecutar el algoritmo genético"""
        population = self.create_population()
        
        print(f"Iniciando algoritmo genético para {self.num_cities} ciudades")
        print(f"Población: {self.population_size}, Generaciones: {self.generations}")
        
        for generation in range(self.generations):
            population = self.evolve_population(population)
            
            if generation % 50 == 0:
                print(f"Generación {generation}: Mejor distancia = {self.best_distance:.2f}")
        
        print(f"Mejor solución encontrada: Distancia = {self.best_distance:.2f}")
        return self.best_solution, self.best_distance, self.fitness_history

def load_tsp_data(file_path):
    """Cargar datos del archivo TSP"""
    df = pd.read_csv(file_path)
    return df

def plot_solution(coordinates, solution, distance, instance_id):
    """Visualizar la solución"""
    coordinates = np.array(coordinates)
    solution_coords = coordinates[solution]
    solution_coords = np.vstack([solution_coords, solution_coords[0]])  # Cerrar el ciclo
    
    plt.figure(figsize=(12, 6))
    
    # Plot ciudades
    plt.subplot(1, 2, 1)
    plt.scatter(coordinates[:, 0], coordinates[:, 1], c='blue', s=50)
    for i, (x, y) in enumerate(coordinates):
        plt.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points')
    
    # Plot ruta
    plt.subplot(1, 2, 2)
    plt.plot(solution_coords[:, 0], solution_coords[:, 1], 'o-', markersize=8)
    plt.scatter(coordinates[:, 0], coordinates[:, 1], c='red', s=50)
    
    for i, (x, y) in enumerate(coordinates):
        plt.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points')
    
    plt.suptitle(f'Instancia {instance_id} - Distancia: {distance:.2f}')
    plt.tight_layout()
    plt.show()

def plot_convergence(fitness_history):
    """Visualizar convergencia del algoritmo"""
    plt.figure(figsize=(10, 6))
    plt.plot(fitness_history)
    plt.title('Convergencia del Algoritmo Genético')
    plt.xlabel('Generación')
    plt.ylabel('Mejor Distancia')
    plt.grid(True)
    plt.show()

def main():
    # Cargar datos
    file_path = "tsp_dataset - copia.csv"
    df = load_tsp_data(file_path)
    
    # Mostrar instancias disponibles
    print("Instancias disponibles:")
    for i, row in df.iterrows():
        print(f"{i}: {row['num_cities']} ciudades (ID: {row['instance_id']})")
    
    # Seleccionar instancia
    instance_choice = int(input("\nSeleccione el número de instancia a ejecutar: "))
    
    if instance_choice >= len(df):
        print("Instancia no válida")
        return
    
    selected_instance = df.iloc[instance_choice]
    
    # Parsear datos
    distance_matrix = ast.literal_eval(selected_instance['distance_matrix'])
    city_coordinates = ast.literal_eval(selected_instance['city_coordinates'])
    num_cities = selected_instance['num_cities']
    instance_id = selected_instance['instance_id']
    
    print(f"\nEjecutando instancia {instance_id} con {num_cities} ciudades")
    print(f"Distancia total de referencia: {selected_instance['total_distance']}")
    
    # Configurar y ejecutar algoritmo genético
    ga = TSPGeneticAlgorithm(
        distance_matrix=distance_matrix,
        population_size=100,
        mutation_rate=0.02,
        elite_size=20,
        generations=1000,
        tournament_size=5
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