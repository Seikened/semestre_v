# -*- coding: utf-8 -*-
# Proyecto 3 – Fase 1: aprender la condición inicial u(x,0) = (1/5) sin(3πx)
# Fer Leon + Sam | DEAP + Sympy

import math
import operator
import random
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import sympy
from deap import algorithms, base, creator, gp, tools

# =========================
# Reproducibilidad y dominio
# =========================
random.seed(42)
np.random.seed(42)

n_puntos = 400
x_grid = np.linspace(0.0, 1.0, n_puntos)
u_inicial = (1/5) * np.sin(3*np.pi * x_grid)

plt.figure(figsize=(10, 5))
plt.plot(x_grid, u_inicial, linewidth=2, label="Perfil inicial u(x,0)")
plt.scatter([0,1], [0,0], s=60, zorder=5)
plt.annotate("Extremo izquierdo\nCondición de contorno:\nu(0,t)=0",
             xy=(0,0), xytext=(0.08, 0.02),
             arrowprops=dict(arrowstyle="->", lw=1),
             fontsize=11, ha="left", va="bottom")
plt.annotate("Extremo derecho\nCondición de contorno:\nu(1,t)=0",
             xy=(1,0), xytext=(0.92, 0.02),
             arrowprops=dict(arrowstyle="->", lw=1),
             fontsize=11, ha="right", va="bottom")
plt.annotate("Condición inicial:\n$u(x,0)=\\frac{1}{5}\\,\\sin(3\\pi x)$",
             xy=(0.65, (1/5)*np.sin(3*np.pi*0.65)), xytext=(0.5, .02),
             arrowprops=dict(arrowstyle="->", lw=1),
             fontsize=12, ha="center", va="bottom" )
plt.xlabel("x (posición a lo largo de la varilla)", fontweight="bold")
plt.ylabel("u(x,0)  (temperatura inicial)", fontweight="bold")
plt.title("Perfil espacial de temperatura inicial en una varilla 1D", fontweight="bold")
plt.xlim(-0.02, 1.02)
plt.xticks([0.0, 0.25, 0.5, 0.75, 1.0], ["0 (izq.)", "0.25", "0.5", "0.75", "1 (der.)"])
plt.axvline(0, linestyle="--", linewidth=1)
plt.axvline(1, linestyle="--", linewidth=1)
plt.grid(True, linewidth=0.7, alpha=0.5)
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()

# =========================
# Conjunto de primitivas
# =========================
pset = gp.PrimitiveSet('MAIN', 1)  # una variable: x
pset.renameArguments(ARG0='x')

# básicas
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(math.sin, 1)
pset.addPrimitive(math.cos, 1)

# importante: NO usar exp aún (causa overflows al inicio)
# cuando quieras activarla, usa una versión protegida (te digo abajo)
# pset.addPrimitive(safe_exp, 1)

# constantes
pset.addTerminal(math.pi, name="pi")  # constante fija bien definida
pset.addEphemeralConstant("U11", partial(random.uniform, -1.0, 1.0))
# si quieres otra fuente:
# pset.addEphemeralConstant("G01", partial(random.gauss, 0.0, 1.0))

# =========================
# Tipos evolutivos
# =========================
try:
    creator.FitnessMin
except AttributeError:
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
try:
    creator.Individual
except AttributeError:
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
# un poco más de profundidad para permitir sin( (const)*pi*x )
toolbox.register('expr', gp.genHalfAndHalf, pset=pset, min_=2, max_=4)
toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)

toolbox.register("compile", gp.compile, pset=pset)

# =========================
# Datos objetivo (fase 1)
# =========================
data_points = x_grid.copy()
valores_objetivo = u_inicial.copy()

# =========================
# Evaluación robusta
# =========================
def evaluar(individual):
    func = toolbox.compile(expr=individual)
    try:
        y_pred = np.array([func(xi) for xi in data_points], dtype=float)
        if not np.all(np.isfinite(y_pred)):
            return (1e9,)
        # clipping suave por si algo se dispara
        y_pred = np.clip(y_pred, -1e6, 1e6)
        y_true = valores_objetivo
        mse = ((y_pred - y_true)**2).mean()
        # parsimony: penaliza árboles enormes para evitar constantes por pura suerte
        penalty = 1e-4 * len(individual)
        return (mse + penalty,)
    except Exception:
        return (1e9,)

toolbox.register("evaluate", evaluar)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# control de bloat adicional (altura máx)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

# =========================
# Evolución
# =========================
poblacion = toolbox.population(n=300)
hof = tools.HallOfFame(1)

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

algorithms.eaSimple(poblacion, toolbox, cxpb=0.5, mutpb=0.2, ngen=100,
                    stats=stats, halloffame=hof, verbose=True)

mejor = hof[0]
print("\nMejor individuo (DEAP):", mejor)

# =========================
# Visualización vs objetivo
# =========================
func_mejor = toolbox.compile(expr=mejor)
y_gp = np.array([func_mejor(xi) for xi in x_grid], dtype=float)

plt.figure(figsize=(10,5))
plt.plot(x_grid, u_inicial, label="Objetivo u(x,0)", linewidth=2)
plt.plot(x_grid, y_gp, label="Individuo GP", linestyle="--")
plt.xlabel("x (posición a lo largo de la varilla)")
plt.ylabel("temperatura inicial")
plt.title("Ajuste simbólico de la condición inicial")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# =========================
# Expresión simbólica legible
# =========================
sym_x = sympy.symbols('x')
replacements = {
    "add": lambda a,b: a + b,
    "sub": lambda a,b: a - b,
    "mul": lambda a,b: a * b,
    "neg": lambda a: -a,
    "sin": sympy.sin,
    "cos": sympy.cos,
    "pi": sympy.pi,
    "x": sym_x,
}
expr_sym = sympy.sympify(str(mejor), locals=replacements)
print("\nExpresión simbólica simplificada:", sympy.simplify(expr_sym))
