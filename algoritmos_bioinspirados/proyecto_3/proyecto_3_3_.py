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
plt.annotate(r"Condición inicial:\n$u(x,0)=\frac{1}{5}\,\sin(3\pi x)$",
             xy=(0.65, (1/5)*np.sin(3*np.pi*0.65)), xytext=(0.5, .02),
             arrowprops=dict(arrowstyle="->", lw=1),
             fontsize=12, ha="center", va="bottom")

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
#plt.show()

# =========================
# Conjunto de primitivas
# =========================
pset = gp.PrimitiveSet('MAIN', 2)  # dos variables: x y t
pset.renameArguments(ARG0='x', ARG1='t')

# primitivas básicas
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(math.sin, 1)
pset.addPrimitive(math.cos, 1)

# bloques auxiliares para componer k*pi*x
def times_pi(a):   return a * math.pi
def times_3(a):    return a * 3.0
pset.addPrimitive(times_pi, 1)
pset.addPrimitive(times_3, 1)



# --- Exponencial protegida para evitar overflow/NaN ---
# --- Exponencial protegida, clip suave con tanh ---
def safe_exp(x, cap=20.0):
    # numérica (para GP y visualización)
    return math.exp(cap * math.tanh(x / cap))

def safe_exp_sym(a, cap=20):
    # simbólica (para derivar y lambdify sin DiracDelta)
    return sympy.exp(cap * sympy.tanh(a / cap))




pset.addPrimitive(safe_exp, 1)

# senos "listos" sin(k*pi*x) para k=1,2,3
def sin_kpi_x_1(x): return math.sin(1.0 * math.pi * x)
def sin_kpi_x_2(x): return math.sin(2.0 * math.pi * x)
def sin_kpi_x_3(x): return math.sin(3.0 * math.pi * x)
pset.addPrimitive(sin_kpi_x_1, 1)
pset.addPrimitive(sin_kpi_x_2, 1)
pset.addPrimitive(sin_kpi_x_3, 1)

# constantes
pset.addTerminal(math.pi, name="pi")     # constante fija
pset.addTerminal(0.2, name="one_fifth")  # 1/5 exacto
pset.addTerminal(1.0, name="one")
pset.addTerminal(2.0, name="two")
pset.addTerminal(3.0, name="three")

# constantes efímeras (usar partial para evitar warnings de pickle)
pset.addEphemeralConstant("U55", partial(random.uniform, -2.0, 2.0))
pset.addEphemeralConstant("U11", partial(random.uniform, -0.5, 0.5))
pset.addEphemeralConstant("G01", partial(random.gauss, 0.0, 0.5))

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

# un poco más de profundidad + subárboles de mutación más ricos
toolbox.register('expr', gp.genHalfAndHalf, pset=pset, min_=3, max_=7)
toolbox.register('individual', tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register('population', tools.initRepeat, list, toolbox.individual)

toolbox.register("compile", gp.compile, pset=pset)

# =========================
# Datos objetivo (fase 1)
# =========================
cantidad_puntos = 100

# Longitud de la varilla
x = np.linspace(0.0, 1.0, cantidad_puntos)

# Tiempo inicial ( segundos )
t = np.linspace(0.0, 60, cantidad_puntos)


data_points = x  
valores_objetivo = t

# =========================
# Helper: forzar dependencia en x
# =========================


# =========================
# Evaluación robusta
# =========================


sx = sympy.symbols('x')   # símbolos reservados para Fase 2 (no usados aquí)
st = sympy.symbols('t')
alpha = 1.0  # difusividad térmica



replacements = {
  "add": lambda a,b: a + b,         "sub": lambda a,b: a - b,
  "mul": lambda a,b: a * b,         "neg": lambda a: -a,
  "sin": sympy.sin,                 "cos": sympy.cos,
  "safe_exp": safe_exp_sym,                 # o tu safe_exp simbólica
  "times_pi": lambda a: a*sympy.pi, "times_3": lambda a: 3*a,
  "pi": sympy.pi,
  "one_fifth": sympy.Rational(1,5),
  "one": sympy.Integer(1), "two": sympy.Integer(2), "three": sympy.Integer(3),
  "sin_kpi_x_1": lambda a: sympy.sin(sympy.pi*a),
  "sin_kpi_x_2": lambda a: sympy.sin(2*sympy.pi*a),
  "sin_kpi_x_3": lambda a: sympy.sin(3*sympy.pi*a),
  "x": sx, "t": st
}   

def evaluar(ind):
    # 1) Árbol -> Sympy
    try:
        u = sympy.sympify(str(ind), locals=replacements)
    except Exception:
        return (1e9,)

    # 2) Derivadas simbólicas
    du_dt  = sympy.diff(u, st)
    d2u_dx = sympy.diff(u, sx, 2)
    resid  = du_dt - alpha*d2u_dx

    # 3) Lambdify (vectorizable sobre NumPy)
    f_res = sympy.lambdify((sx, st), resid, 'numpy')
    f_u   = sympy.lambdify((sx, st), u,     'numpy')

    # 4) Mallas numéricas moderadas
    Nx, Nt = 64, 64
    X  = np.linspace(0.0, 1.0, Nx)
    T  = np.linspace(0.0, 0.15, Nt)      # tiempo corto para estabilidad
    XX, TT = np.meshgrid(X, T, indexing='xy')

    # 5) Residuo PDE
    R = f_res(XX, TT)
    if not np.all(np.isfinite(R)):
        return (1e8,)
    mse_res = float(np.mean(R*R))

    # 6) Condiciones de frontera u(0,t)=u(1,t)=0
    U_left  = f_u(0.0, T)
    U_right = f_u(1.0, T)
    if not (np.all(np.isfinite(U_left)) and np.all(np.isfinite(U_right))):
        return (1e8,)
    bc_pen = float(np.mean(U_left*U_left) + np.mean(U_right*U_right))

    # 7) Condición inicial u(x,0)= (1/5) sin(3πx)
    U_init = f_u(X, 0.0)
    target = (1.0/5.0)*np.sin(3*np.pi*X)
    if not np.all(np.isfinite(U_init)):
        return (1e8,)
    ic_pen = float(np.mean((U_init - target)**2))

    # 8) Regularización anti-bloat
    size_pen = 1e-4 * len(ind)

    # 9) Fitness total (minimizar)
    λ_bc, λ_ic = 1.0, 10.0
    deps = u.free_symbols
    dep_pen = 0.0
    if sx not in deps:
        dep_pen += 1.0
    if st not in deps:
        dep_pen += 0.2
    loss = mse_res + λ_bc*bc_pen + λ_ic*ic_pen + size_pen + dep_pen
    return (loss,)

toolbox.register("evaluate", evaluar)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=1, max_=4)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# control de bloat adicional (altura máx)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

# =========================
# Evolución
# =========================
poblacion = toolbox.population(n=600)
hof = tools.HallOfFame(1)




cxpb = 0.4
mutpb = 0.4
ngen  = 300

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

algorithms.eaSimple(poblacion, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen,
                    stats=stats, halloffame=hof, verbose=True)

mejor = hof[0]
print("\nMejor individuo (DEAP):", mejor)

# =========================
# Visualización vs objetivo
# =========================
func_mejor = toolbox.compile(expr=mejor)
y_gp = np.array([func_mejor(xi, 0.0) for xi in x_grid], dtype=float)
mse_final = float(((y_gp - u_inicial)**2).mean())
print(f"MSE mejor individuo: {mse_final:.8e}")

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

expr_sym = sympy.sympify(str(mejor), locals=replacements)
print("\nExpresión simbólica simplificada:", sympy.simplify(expr_sym))

