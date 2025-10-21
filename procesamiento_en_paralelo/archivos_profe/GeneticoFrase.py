import tkinter as tk
from tkinter import ttk, messagebox
import random
import string
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

class GeneticAlgorithm:
    def __init__(self, target_phrase, population_size=100, mutation_rate=0.01):
        self.target_phrase = target_phrase.lower()
        self.target_length = len(target_phrase)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generation = 0
        self.best_fitness_history = []
        self.average_fitness_history = []
        
        # Caracteres permitidos (letras minúsculas, espacio y algunos signos)
        self.valid_chars = string.ascii_lowercase + " áéíóú"
        
        # Crear población inicial
        self.population = self.initialize_population()
        
    def initialize_population(self):
        """Crear población inicial con frases aleatorias"""
        population = []
        for _ in range(self.population_size):
            individual = ''.join(random.choice(self.valid_chars) for _ in range(self.target_length))
            population.append(individual)
        return population
    
    def calculate_fitness(self, individual):
        """Calcular fitness comparando con la frase objetivo"""
        score = 0
        for i in range(self.target_length):
            if individual[i] == self.target_phrase[i]:
                score += 1
        return score / self.target_length  # Fitness normalizado entre 0 y 1
    
    def select_parents(self, fitness_scores):
        """Seleccionar padres usando método de ruleta"""
        total_fitness = sum(fitness_scores)
        if total_fitness == 0:
            return random.choices(self.population, k=2)
        
        probabilities = [fitness / total_fitness for fitness in fitness_scores]
        parents = random.choices(self.population, weights=probabilities, k=2)
        return parents
    
    def crossover(self, parent1, parent2):
        """Realizar cruce de un punto"""
        crossover_point = random.randint(1, self.target_length - 1)
        child = parent1[:crossover_point] + parent2[crossover_point:]
        return child
    
    def mutate(self, individual):
        """Aplicar mutación a un individuo"""
        mutated = list(individual)
        for i in range(self.target_length):
            if random.random() < self.mutation_rate:
                mutated[i] = random.choice(self.valid_chars)
        return ''.join(mutated)
    
    def evolve(self):
        """Ejecutar una generación del algoritmo genético"""
        # Calcular fitness para cada individuo
        fitness_scores = [self.calculate_fitness(individual) for individual in self.population]
        
        # Encontrar el mejor individuo
        best_index = np.argmax(fitness_scores)
        best_individual = self.population[best_index]
        best_fitness = fitness_scores[best_index]
        
        # Guardar estadísticas para gráficos
        self.best_fitness_history.append(best_fitness)
        self.average_fitness_history.append(np.mean(fitness_scores))
        
        # Crear nueva población
        new_population = []
        
        # Elitismo: mantener al mejor individuo
        new_population.append(best_individual)
        
        # Crear el resto de la población
        while len(new_population) < self.population_size:
            # Seleccionar padres
            parent1, parent2 = self.select_parents(fitness_scores)
            
            # Cruzar
            child = self.crossover(parent1, parent2)
            
            # Mutar
            child = self.mutate(child)
            
            new_population.append(child)
        
        self.population = new_population
        self.generation += 1
        
        return best_individual, best_fitness

class GeneticApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Algoritmo Genético - Descifrar Frase")
        self.root.geometry("900x700")
        
        # Frase objetivo
        self.target_phrase = "bienvenidos ingenieros"
        
        # Variables de control
        self.ga = None
        self.running = False
        self.speed = 100  # ms entre generaciones
        
        # Configurar interfaz
        self.setup_ui()
        
    def setup_ui(self):
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configurar expansión de filas y columnas
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(6, weight=1)
        
        # Título
        title_label = ttk.Label(main_frame, text="Algoritmo Genético para Descifrar Frase", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=10)
        
        # Frase objetivo
        target_label = ttk.Label(main_frame, text=f"Frase objetivo: '{self.target_phrase}'", 
                                font=("Arial", 12))
        target_label.grid(row=1, column=0, columnspan=2, pady=5)
        
        # Controles
        ttk.Label(main_frame, text="Tamaño de población:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.population_size = tk.IntVar(value=100)
        ttk.Entry(main_frame, textvariable=self.population_size, width=10).grid(row=2, column=1, sticky=tk.W, pady=5)
        
        ttk.Label(main_frame, text="Tasa de mutación:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.mutation_rate = tk.DoubleVar(value=0.01)
        ttk.Entry(main_frame, textvariable=self.mutation_rate, width=10).grid(row=3, column=1, sticky=tk.W, pady=5)
        
        # Botones
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=4, column=0, columnspan=2, pady=10)
        
        ttk.Button(button_frame, text="Iniciar", command=self.start_algorithm).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Detener", command=self.stop_algorithm).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Reiniciar", command=self.reset_algorithm).pack(side=tk.LEFT, padx=5)
        
        # Velocidad
        ttk.Label(button_frame, text="Velocidad:").pack(side=tk.LEFT, padx=5)
        self.speed_scale = tk.Scale(button_frame, from_=10, to=500, orient=tk.HORIZONTAL, 
                                   length=150, showvalue=True)
        self.speed_scale.set(100)
        self.speed_scale.pack(side=tk.LEFT, padx=5)
        
        # Información de la ejecución
        info_frame = ttk.LabelFrame(main_frame, text="Progreso", padding="5")
        info_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        info_frame.columnconfigure(1, weight=1)
        
        ttk.Label(info_frame, text="Generación:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.generation_var = tk.StringVar(value="0")
        ttk.Label(info_frame, textvariable=self.generation_var).grid(row=0, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(info_frame, text="Mejor frase:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.best_phrase_var = tk.StringVar(value="")
        ttk.Label(info_frame, textvariable=self.best_phrase_var, wraplength=400).grid(row=1, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(info_frame, text="Fitness:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.fitness_var = tk.StringVar(value="0.0")
        ttk.Label(info_frame, textvariable=self.fitness_var).grid(row=2, column=1, sticky=tk.W, pady=2)
        
        # Gráfico
        self.figure, self.ax = plt.subplots(figsize=(8, 4))
        self.canvas = FigureCanvasTkAgg(self.figure, master=main_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        
        # Configurar el gráfico
        self.ax.set_xlabel('Generación')
        self.ax.set_ylabel('Fitness')
        self.ax.set_title('Evolución del Fitness')
        self.ax.grid(True)
        self.ax.set_ylim(0, 1.1)
        
        # Información para universitarios
        info_text = """
        Este algoritmo genético demuestra cómo evolucionan las soluciones hacia un objetivo.
        Conceptos clave:
        - Población: Conjunto de posibles soluciones (frases)
        - Fitness: Medida de qué tan buena es una solución
        - Selección: Los mejores individuos tienen más probabilidad de reproducirse
        - Cruzamiento: Combinación de características de dos padres
        - Mutación: Pequeños cambios aleatorios para mantener diversidad
        """
        info_label = ttk.Label(main_frame, text=info_text, justify=tk.LEFT)
        info_label.grid(row=7, column=0, columnspan=2, sticky=tk.W, pady=10)
    
    def start_algorithm(self):
        if self.running:
            return
            
        self.running = True
        population_size = self.population_size.get()
        mutation_rate = self.mutation_rate.get()
        
        # Inicializar algoritmo genético
        self.ga = GeneticAlgorithm(
            target_phrase=self.target_phrase,
            population_size=population_size,
            mutation_rate=mutation_rate
        )
        
        # Iniciar evolución
        self.evolve()
    
    def stop_algorithm(self):
        self.running = False
    
    def reset_algorithm(self):
        self.stop_algorithm()
        self.ga = None
        self.generation_var.set("0")
        self.best_phrase_var.set("")
        self.fitness_var.set("0.0")
        
        # Limpiar gráfico
        self.ax.clear()
        self.ax.set_xlabel('Generación')
        self.ax.set_ylabel('Fitness')
        self.ax.set_title('Evolución del Fitness')
        self.ax.grid(True)
        self.ax.set_ylim(0, 1.1)
        self.canvas.draw()
    
    def evolve(self):
        if not self.running or self.ga is None:
            return
        
        # Ejecutar una generación
        best_individual, best_fitness = self.ga.evolve()
        
        # Actualizar interfaz
        self.generation_var.set(str(self.ga.generation))
        self.best_phrase_var.set(best_individual)
        self.fitness_var.set(f"{best_fitness:.4f}")
        
        # Actualizar gráfico
        self.ax.clear()
        generations = range(1, self.ga.generation + 1)
        self.ax.plot(generations, self.ga.best_fitness_history, 'b-', label='Mejor fitness')
        self.ax.plot(generations, self.ga.average_fitness_history, 'r-', label='Fitness promedio')
        self.ax.set_xlabel('Generación')
        self.ax.set_ylabel('Fitness')
        self.ax.set_title('Evolución del Fitness')
        self.ax.legend()
        self.ax.grid(True)
        self.ax.set_ylim(0, 1.1)
        self.canvas.draw()
        
        # Verificar si se encontró la solución
        if best_individual == self.target_phrase:
            self.running = False
            messagebox.showinfo("¡Éxito!", f"¡Frase descifrada en {self.ga.generation} generaciones!")
            return
        
        # Continuar con la siguiente generación
        self.root.after(self.speed_scale.get(), self.evolve)

if __name__ == "__main__":
    root = tk.Tk()
    app = GeneticApp(root)
    root.mainloop()
