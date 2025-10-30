
import random
import math as mt
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx




def funcion(x):
    return ((x-3)**2) + np.sin(5*x)



# Dominio de busqueda
imax = 10
imin = -imax



x = np.linspace(imin, imax, 100)
y = funcion(x)


# plt.plot(x, y)
# plt.title("Función a optimizar")
# plt.xlabel("x")
# plt.ylabel("f(x)")
# plt.grid()
# plt.show()


# =================== CONSTRUCCIÓN DEL GRÁFO DE DISTANCIAS ===================
def construir_grafo(xs, fx, k=4, eps=1e-6):
    grafo = {}
    dx = xs[1] - xs[0]
    N = len(xs)
    for i in range(N):
        grafo[i+1] = {}
        for step in range(1, k+1):
            j = i + step
            if j < N:
                # peso = longitud del salto * "energía" en el destino
                w = (j - i) * dx * (eps + fx[j])   # ojo: fx es np.array 0..N-1
                grafo[i+1][j+1] = float(w)
    return grafo

# =================== PARÁMETROS DEL ALGORITMO ===================
xmin, xmax = -5, 5
N = 200  # número de nodos
xs = np.linspace(xmin, xmax, N)
fx = funcion(xs)
fx = fx - fx.min() + 1e-6


GRAFO_DISTANCIAS = construir_grafo(xs, fx)



grafo = nx.Graph()
for i in GRAFO_DISTANCIAS:
    for j in GRAFO_DISTANCIAS[i]:
        grafo.add_edge(i, j, weight=GRAFO_DISTANCIAS[i][j])
nx.draw(grafo, with_labels=True)
plt.show()


NODO_INICIO = 1
NODO_FINAL  = len(xs)
TODOS_NODOS = list(range(1, len(xs)+1))
NUM_NODOS   = len(TODOS_NODOS)



# --- 2. PARÁMETROS ACO ---
NUM_HORMIGAS = 50
NUM_ITERACIONES = 100
RHO = 0.01
Q = 1.0
ALFA = 1.0
BETA = 1.0










# --- 3. FUNCIONES PRINCIPALES ---

def inicializar_feromonas(grafo):
    feromonas = {}
    for i in grafo:
        feromonas[i] = {}
        for j in grafo[i]:
            feromonas[i][j] = 0.1  # valor inicial
    return feromonas


def probabilidad_movimiento(actual, feromonas, grafo, alpha, beta):
    total = 0.0
    probabilidades = {}

    for siguiente in grafo[actual]:
        tau = feromonas[actual][siguiente]
        if siguiente <= len(xs):
            eta = 1.0 / (1.0 + fx[siguiente - 1])   # nodo real
              # sumidero S
        total += (tau ** alpha) * (eta ** beta)

    if total == 0.0:
        return {}

    for siguiente in grafo[actual]:
        tau = feromonas[actual][siguiente]
        if siguiente <= len(xs):
            eta = 1.0 / (1.0 + fx[siguiente - 1])
        
        probabilidades[siguiente] = ((tau ** alpha) * (eta ** beta)) / total

    return probabilidades




def seleccionar_nodo(probabilidades):
    r = random.random()
    acumulada = 0.0
    for nodo, prob in probabilidades.items():
        acumulada += prob
        if r <= acumulada:
            return nodo
    # Si por redondeo no se elige ninguno
    return list(probabilidades.keys())[-1]


def construir_ruta(feromonas, grafo, alpha, beta):
    ruta = [NODO_INICIO]
    actual = NODO_INICIO
    while actual != NODO_FINAL:
        probabilidades = probabilidad_movimiento(actual, feromonas, grafo, alpha, beta)
        if not probabilidades:
            break
        siguiente = seleccionar_nodo(probabilidades)
        ruta.append(siguiente)
        actual = siguiente
    return ruta


def calcular_longitud_ruta(ruta, grafo):
    distancia = 0.0
    for i in range(len(ruta) - 1):
        a, b = ruta[i], ruta[i+1]
        if b in grafo[a]:
            distancia += grafo[a][b]
        else:
            # Si no existe conexión directa, penalizamos
            distancia += 9999
    return distancia


def actualizar_feromonas(feromonas, rutas, distancias, rho, q):
    # Evaporación
    for i in feromonas:
        for j in feromonas[i]:
            feromonas[i][j] *= (1 - rho)

    # Deposito de feromonas
    for k in range(len(rutas)):
        ruta = rutas[k]
        dist = distancias[k]
        for i in range(len(ruta) - 1):
            a, b = ruta[i], ruta[i+1]
            if b in feromonas[a]:
                feromonas[a][b] += q / dist


def aco():
    feromonas = inicializar_feromonas(GRAFO_DISTANCIAS)
    mejor_ruta = None
    mejor_distancia = mt.inf

    for iteracion in range(NUM_ITERACIONES):
        rutas = []
        distancias = []

        for i in range(NUM_HORMIGAS):
            ruta = construir_ruta(feromonas, GRAFO_DISTANCIAS, ALFA, BETA)
            distancia = calcular_longitud_ruta(ruta, GRAFO_DISTANCIAS)
            rutas.append(ruta)
            distancias.append(distancia)

            if distancia < mejor_distancia and ruta[-1] == NODO_FINAL:
                mejor_ruta = ruta
                idx_min_en_ruta = min(mejor_ruta, key=lambda i: fx[i-1] if i<=len(xs) else float("inf"))
                x_min = xs[idx_min_en_ruta-1]
                f_min = funcion(x_min)
                print(f"x* ≈ {x_min:.4f}, f(x*) ≈ {f_min:.6f}")

                mejor_distancia = distancia

        actualizar_feromonas(feromonas, rutas, distancias, RHO, Q)

        print(f"Iteración {iteracion+1}: Mejor distancia = {mejor_distancia:.3f}")

    print("\n--- RESULTADO FINAL ---")
    print(f"Mejor ruta encontrada: {mejor_ruta}")
    print(f"Distancia total: {mejor_distancia:.3f}")


# --- 4. EJECUCIÓN ---
if __name__ == "__main__":
    aco()



