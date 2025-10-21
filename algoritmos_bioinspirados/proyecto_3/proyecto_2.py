
import math
import operator
import random
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import sympy
from deap import algorithms, base, creator, gp, tools

# =========================
# FASE 2: GP para la PDE completa (u_t - alpha u_xx = 0)
# =========================

alpha = 1.0      # difusividad térmica (ajústala si el profe pide otro valor)
T_max = 0.3      # horizonte temporal de entrenamiento (corto y estable)
N_int = 1500     # puntos interiores (x,t) para residuo PDE
N_bc  = 300      # puntos para contorno
N_ic  = len(x_grid)  # usamos toda la malla de Fase 1 para la CI

# ---------- Primitivas 2D (x, t) ----------
pset2 = gp.PrimitiveSet('MAIN2', 2)   # ahora dos variables: x, t
pset2.renameArguments(ARG0='x')       # coherencia con tu estilo
pset2.renameArguments(ARG1='t')

# básicas
pset2.addPrimitive(operator.add, 2)
pset2.addPrimitive(operator.sub, 2)
pset2.addPrimitive(operator.mul, 2)
pset2.addPrimitive(operator.neg, 1)
pset2.addPrimitive(math.sin, 1)
pset2.addPrimitive(math.cos, 1)

# funciones seguras
def safe_div(a, b):
    try:
        return a / b if abs(b) > 1e-12 else a
    except Exception:
        return 0.0

def safe_exp(a):
    # recorta entrada para evitar overflow
    a = max(min(a, 30.0), -30.0)
    try:
        return math.exp(a)
    except Exception:
        return math.exp(30.0 if a > 0 else -30.0)

pset2.addPrimitive(safe_div, 2)
pset2.addPrimitive(safe_exp, 1)

# bloques útiles
def times_pi(a): return a * math.pi
def times_3(a):  return a * 3.0
pset2.addPrimitive(times_pi, 1)
pset2.addPrimitive(times_3, 1)

# senos en x que respetan BC (sin(k*pi*x) = 0 en x=0,1)
def sin_kpi_x_1(x): return math.sin(1.0 * math.pi * x)
def sin_kpi_x_2(x): return math.sin(2.0 * math.pi * x)
def sin_kpi_x_3(x): return math.sin(3.0 * math.pi * x)
pset2.addPrimitive(sin_kpi_x_1, 1)
pset2.addPrimitive(sin_kpi_x_2, 1)
pset2.addPrimitive(sin_kpi_x_3, 1)

# decaimientos temporales tipo calor (e^{-k^2 pi^2 alpha t})
def decay1(t): return math.exp(-(1.0*math.pi)**2 * alpha * t)
def decay2(t): return math.exp(-(2.0*math.pi)**2 * alpha * t)
def decay3(t): return math.exp(-(3.0*math.pi)**2 * alpha * t)
pset2.addPrimitive(decay1, 1)
pset2.addPrimitive(decay2, 1)
pset2.addPrimitive(decay3, 1)

# constantes
pset2.addTerminal(math.pi, name="pi")
pset2.addTerminal(0.2, name="one_fifth")
pset2.addTerminal(1.0, name="one")
pset2.addTerminal(2.0, name="two")
pset2.addTerminal(3.0, name="three")

# efímeras (con partial -> sin warnings)
pset2.addEphemeralConstant("U55", partial(random.uniform, -5.0, 5.0))
pset2.addEphemeralConstant("U11", partial(random.uniform, -1.0, 1.0))
pset2.addEphemeralConstant("G01", partial(random.gauss, 0.0, 1.0))

# ---------- Tipos y toolbox 2D ----------
try:
    creator.FitnessMinPDE
except AttributeError:
    creator.create("FitnessMinPDE", base.Fitness, weights=(-1.0,))
try:
    creator.Individual2D
except AttributeError:
    creator.create("Individual2D", gp.PrimitiveTree, fitness=creator.FitnessMinPDE)

toolbox2 = base.Toolbox()
toolbox2.register('expr', gp.genHalfAndHalf, pset=pset2, min_=3, max_=7)
toolbox2.register('individual', tools.initIterate, creator.Individual2D, toolbox2.expr)
toolbox2.register('population', tools.initRepeat, list, toolbox2.individual)
toolbox2.register("compile", gp.compile, pset=pset2)

# ---------- Muestreo de collocation ----------
# interiores (x in (0,1), t in (0,T_max])
x_int = np.random.rand(N_int)
t_int = np.random.rand(N_int) * T_max

# contorno (x=0 o x=1, t in (0,T_max])
t_bc = np.random.rand(N_bc) * T_max
x_bc_left  = np.zeros(N_bc//2)
x_bc_right = np.ones(N_bc - N_bc//2)
x_bc = np.concatenate([x_bc_left, x_bc_right])

# inicial (t=0, x grid ya definido)
x_ic = x_grid.copy()
t_ic = np.zeros_like(x_ic)
u_ic = u_inicial.copy()

# ---------- utilidades derivadas finitas ----------
def du_dt(f, x, t, dt=1e-3):
    return (f(x, t+dt) - f(x, t-dt)) / (2*dt)

def d2u_dx2(f, x, t, dx=1e-3):
    return (f(x+dx, t) - 2.0*f(x, t) + f(x-dx, t)) / (dx*dx)

# ---------- checar dependencia en x y t ----------
def usa_xy(ind):
    names = [getattr(node, "name", "") for node in ind]
    return ("ARG0" in names) and ("ARG1" in names)

# ---------- evaluación PDE ----------
w_pde = 1.0   # peso residuo
w_bc  = 10.0  # peso contorno (fuerte para que respete BC)
w_ic  = 5.0   # peso condición inicial

def evaluar_pde(ind):
    # fuerza a depender de x y t (evita constantes o funciones de una sola variable)
    if not usa_xy(ind):
        return (1e6,)

    f = toolbox2.compile(expr=ind)

    # wrappers seguros (evita out-of-domain en diferencias)
    def f_safe(x, t):
        try:
            y = f(float(x), float(t))
            if not np.isfinite(y): return 0.0
            return max(min(y, 1e6), -1e6)
        except Exception:
            return 0.0

    # residuo PDE en puntos interiores
    try:
        ut  = np.array([du_dt(f_safe, xi, ti) for xi, ti in zip(x_int, t_int)], dtype=float)
        uxx = np.array([d2u_dx2(f_safe, xi, ti) for xi, ti in zip(x_int, t_int)], dtype=float)
        res = ut - alpha * uxx
        pde_loss = np.mean(res*res)
    except Exception:
        return (1e9,)

    # contorno: u(0,t)=u(1,t)=0
    try:
        u_bc = np.array([f_safe(xi, ti) for xi, ti in zip(x_bc, t_bc)], dtype=float)
        bc_loss = np.mean(u_bc*u_bc)
    except Exception:
        return (1e9,)

    # condición inicial: u(x,0)≈u_ic(x)
    try:
        u0_pred = np.array([f_safe(xi, 0.0) for xi in x_ic], dtype=float)
        ic_loss = np.mean((u0_pred - u_ic)**2)
    except Exception:
        return (1e9,)

    # parsimonia suave
    parsimony = 1e-6 * len(ind)

    total = w_pde*pde_loss + w_bc*bc_loss + w_ic*ic_loss + parsimony
    return (float(total),)

toolbox2.register("evaluate", evaluar_pde)
toolbox2.register("select", tools.selTournament, tournsize=3)
toolbox2.register("mate", gp.cxOnePoint)
toolbox2.register("expr_mut", gp.genFull, min_=1, max_=4)
toolbox2.register("mutate", gp.mutUniform, expr=toolbox2.expr_mut, pset=pset2)
toolbox2.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox2.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

# ---------- población y semillas dirigidas ----------
pob2 = toolbox2.population(n=600)
hof2 = tools.HallOfFame(1)

# siembra con la solución analítica exacta (ayuda al algoritmo y te sirve de validación)
from deap import gp as gp_mod
sol_analitica = "mul(one_fifth, mul(sin_kpi_x_3(x), decay3(t)))"
seed = creator.Individual2D(gp_mod.PrimitiveTree.from_string(sol_analitica, pset2))
pob2[0] = seed

# ---------- evolución ----------
cxpb2, mutpb2, ngen2 = 0.4, 0.4, 300
stats2 = tools.Statistics(lambda ind: ind.fitness.values)
stats2.register("avg", np.mean)
stats2.register("std", np.std)
stats2.register("min", np.min)
stats2.register("max", np.max)

algorithms.eaSimple(pob2, toolbox2, cxpb=cxpb2, mutpb=mutpb2, ngen=ngen2,
                    stats=stats2, halloffame=hof2, verbose=True)

best2 = hof2[0]
print("\nMejor individuo (PDE):", best2)

# ---------- evaluación visual en varios tiempos ----------
f_best = toolbox2.compile(expr=best2)
ts = [0.0, 0.02, 0.08, 0.2, 0.3]
plt.figure(figsize=(10,5))
for tt in ts:
    y_pred = np.array([f_best(xi, tt) for xi in x_grid], dtype=float)
    plt.plot(x_grid, y_pred, label=f"t={tt:.2f}")
# verdad a t=0
plt.plot(x_grid, u_inicial, 'k--', lw=2, label="u(x,0) objetivo")
plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.title("Evolución temporal aprendida por GP (PDE)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ---------- expresión simbólica legible (2D) ----------
sym_x, sym_t = sympy.symbols('x t')
rep2 = {
    "add": lambda a,b: a + b,
    "sub": lambda a,b: a - b,
    "mul": lambda a,b: a * b,
    "neg": lambda a: -a,
    "sin": sympy.sin,
    "cos": sympy.cos,
    "pi": sympy.pi,
    "one_fifth": sympy.Rational(1,5),
    "one": sympy.Integer(1),
    "two": sympy.Integer(2),
    "three": sympy.Integer(3),
    "times_pi": lambda a: a*sympy.pi,
    "times_3":  lambda a: a*3,
    "sin_kpi_x_1": lambda a: sympy.sin(sympy.pi*a),
    "sin_kpi_x_2": lambda a: sympy.sin(2*sympy.pi*a),
    "sin_kpi_x_3": lambda a: sympy.sin(3*sympy.pi*a),
    "decay1": lambda a: sympy.exp(-(sympy.pi**2)*alpha*a),
    "decay2": lambda a: sympy.exp(-(2*sympy.pi)**2*alpha*a),
    "decay3": lambda a: sympy.exp(-(3*sympy.pi)**2*alpha*a),
    "safe_div": lambda a,b: a/b,
    "safe_exp": lambda a: sympy.exp(a),
    "x": sym_x,
    "t": sym_t,
}
expr_sym2 = sympy.sympify(str(best2), locals=rep2)
print("\nExpr. simbólica PDE simplificada:", sympy.simplify(expr_sym2))
