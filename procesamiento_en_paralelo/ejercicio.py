import numpy as np
from time import perf_counter

# Tabulador de salario en USD por años de experiencia (mensuales)

tabulñador_salario = {1: 1500, 2: 2200, 3: 3000, 4: 4250, 5: 5700}


# Tarea 1 | Generar los datos
# Lista de 100,000,000 trabajadores con años de experiencia aleatorios entre 1 y 5 (con 1,000,000,000 no se pudo se mato el proceso [1]    28133 killed     /Users/ferleon/Github/semestre_v/.venv/bin/python)
start_time = perf_counter()
trabajadores = np.random.randint(1, 6, size=100000000).tolist()
tiempo_generar_lista = perf_counter() - start_time

salarios_antes_incremento = [tabulñador_salario[xp] for xp in trabajadores]


# Factor de incremento de salario por año de experiencia
def factor(xp):
    return 1 + (xp * 0.05)


# Calcular salarios con incremento
def calcular_salario(xp):
    salario_base = tabulñador_salario[xp]
    return salario_base * factor(xp)


# Tarea 2 | Aplicar el incremento de salario
start_time = perf_counter()
salarios_con_incremento = list(map(calcular_salario, trabajadores))
tiempo_incremento_salario = perf_counter() - start_time

# Comparar antes y después
print("Salario de cada trabajador estimado antes del incremento:")
print(salarios_antes_incremento[:10]) 
print("Salario con incremento:")
print(salarios_con_incremento[:10])

# Medir tiempos estrictos
print(f"Tiempo estricto para generar la lista de trabajadores: {tiempo_generar_lista:.2f} segundos")
print(f"Tiempo estricto para aplicar el incremento: {tiempo_incremento_salario:.2f} segundos")
