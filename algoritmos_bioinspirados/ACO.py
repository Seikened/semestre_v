import random
import math

# --- 1. PARÁMETROS DEL PROBLEMA ---

GRAFO_DISTANCIAS = {
    1: {2: 5.0, 3: 3.1, 6: 5.2},
    2: {1: 5.0, 3: 4.9, 7: 5.2},
    3: {1: 3.1, 2: 4.9, 6: 3.2, 7: 3.0, 5: 6.0},
    4: {7: 4.8, 5: 5.5},
    5: {3: 6.0, 6: 4.7, 4: 5.5},
    6: {1: 5.2, 3: 3.2, 5: 4.7},
    7: {2: 5.2, 3: 3.0, 4: 4.8}
}

NODO_INICIO = 1
NODO_FINAL = 4
TODOS_NODOS = [1, 2, 3, 4, 5, 6, 7]
NUM_NODOS = len(TODOS_NODOS)

# --- 2. PARÁMETROS ACO ---
NUM_HORMIGAS = 2
NUM_ITERACIONES = 50
RHO = 0.01       # Tasa de evaporación
Q = 1.0          # Constante de depósito de feromona
ALFA = 1.0       # Influencia del rastro de feromona
BETA = 1.0       # Influencia de la heurística (visibilidad)

# --- 3. FUNCIONES PRINCIPALES ---

def inicializar_feromonas(grafo):
    feromonas = {}
    for i in grafo:
        feromonas[i] = {}
        for j in grafo[i]:
            feromonas[i][j] = 0.1  # valor inicial
    return feromonas


def probabilidad_movimiento(actual, nodos_no_visitados, feromonas, grafo, alpha, beta):
    total = 0.0
    probabilidades = {}

    # Calcula el denominador (la suma de todos los pesos posibles)
    for siguiente in nodos_no_visitados:
        if siguiente in grafo[actual]:
            tau = feromonas[actual][siguiente]
            eta = 1.0 / grafo[actual][siguiente]
            total += (tau ** alpha) * (eta ** beta)

    # Calcula la probabilidad normalizada para cada destino
    for siguiente in nodos_no_visitados:
        if siguiente in grafo[actual]:
            tau = feromonas[actual][siguiente]
            eta = 1.0 / grafo[actual][siguiente]
            prob = ((tau ** alpha) * (eta ** beta)) / total
            probabilidades[siguiente] = prob

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
    nodos_no_visitados = TODOS_NODOS.copy()
    nodos_no_visitados.remove(actual)

    while actual != NODO_FINAL and len(nodos_no_visitados) > 0:
        probabilidades = probabilidad_movimiento(actual, nodos_no_visitados, feromonas, grafo, alpha, beta)
        if not probabilidades:
            break
        siguiente = seleccionar_nodo(probabilidades)
        ruta.append(siguiente)
        actual = siguiente
        if siguiente in nodos_no_visitados:
            nodos_no_visitados.remove(siguiente)

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
    mejor_distancia = math.inf

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
                mejor_distancia = distancia

        actualizar_feromonas(feromonas, rutas, distancias, RHO, Q)

        print(f"Iteración {iteracion+1}: Mejor distancia = {mejor_distancia:.3f}")

    print("\n--- RESULTADO FINAL ---")
    print(f"Mejor ruta encontrada: {mejor_ruta}")
    print(f"Distancia total: {mejor_distancia:.3f}")


# --- 4. EJECUCIÓN ---
if __name__ == "__main__":
    aco()
