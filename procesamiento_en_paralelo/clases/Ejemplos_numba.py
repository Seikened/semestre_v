import numba
from numba import jit, njit, vectorize, guvectorize
import numpy as np
import time
import math

# Ejemplo 1: Función básica con @jit
@jit(nopython=True)
def suma_vectorial(a, b):
    """Suma elemento a elemento de dos arrays"""
    return a + b

# Prueba
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])
resultado = suma_vectorial(a, b)
print(f"Suma vectorial: {resultado}")

-------------------------------------------

# Ejemplo 2: Optimización de bucle con @njit
@njit
def suma_elementos(arr):
    """Suma todos los elementos de un array"""
    total = 0.0
    for i in range(len(arr)):
        total += arr[i]
    return total

# Ejemplo 3: Producto punto manual
@njit
def producto_punto(a, b):
    """Calcula el producto punto de dos vectores"""
    if len(a) != len(b):
        raise ValueError("Los vectores deben tener la misma longitud")
    
    resultado = 0.0
    for i in range(len(a)):
        resultado += a[i] * b[i]
    return resultado

# Pruebas
arr_grande = np.random.random(1000000)
print(f"Suma elementos: {suma_elementos(arr_grande)}")

vector1 = np.array([1.0, 2.0, 3.0])
vector2 = np.array([4.0, 5.0, 6.0])
print(f"Producto punto: {producto_punto(vector1, vector2)}")

-------------------------------------------------------------------
# Ejemplo 4: Función universal para operaciones matemáticas
@vectorize(['float64(float64, float64)'], nopython=True)
def distancia_euclidiana(x, y):
    """Calcula la distancia euclidiana entre dos puntos"""
    return math.sqrt(x**2 + y**2)

# Ejemplo 5: Función personalizada con condiciones
@vectorize(['float64(float64)'], nopython=True)
def funcion_activacion_relu(x):
    """Función de activación ReLU"""
    return max(0.0, x)

# Pruebas
x_vals = np.array([3.0, 4.0, 5.0])
y_vals = np.array([4.0, 3.0, 12.0])
distancias = distancia_euclidiana(x_vals, y_vals)
print(f"Distancias euclidianas: {distancias}")

valores = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
activaciones = funcion_activacion_relu(valores)
print(f"Activaciones ReLU: {activaciones}")

-------------------------------------
# Ejemplo 6: Normalización de arrays
@guvectorize(['(float64[:], float64[:])'], '(n)->(n)', nopython=True)
def normalizar_vector(vector, resultado):
    """Normaliza un vector a longitud 1"""
    suma_cuadrados = 0.0
    for i in range(len(vector)):
        suma_cuadrados += vector[i] ** 2
    
    magnitud = math.sqrt(suma_cuadrados)
    
    for i in range(len(vector)):
        resultado[i] = vector[i] / magnitud

# Ejemplo 7: Aplicar operación a ventanas deslizantes
@guvectorize(['(float64[:], int64[:], float64[:])'], '(n),()->(n)', nopython=True)
def promedio_movil(arr, ventana, resultado):
    """Calcula promedio móvil"""
    k = ventana[0]
    for i in range(len(arr)):
        inicio = max(0, i - k + 1)
        fin = i + 1
        suma = 0.0
        for j in range(inicio, fin):
            suma += arr[j]
        resultado[i] = suma / (fin - inicio)

# Pruebas
vectores = np.random.random((5, 3))
vectores_normalizados = np.empty_like(vectores)
normalizar_vector(vectores, vectores_normalizados)
print("Vectores normalizados:")
print(vectores_normalizados)

datos = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
ventana_size = np.array([3])
promedios = promedio_movil(datos, ventana_size)
print(f"Promedios móviles: {promedios}")

-------------------------------------------------
# Ejemplo 8: Procesamiento de arrays 2D
@njit
def multiplicacion_matrices(A, B):
    """Multiplicación de matrices"""
    m, n = A.shape
    n2, p = B.shape
    
    if n != n2:
        raise ValueError("Dimensiones incompatibles")
    
    C = np.zeros((m, p))
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
    return C

# Ejemplo 9: Conversión de tipos con strings
@njit
def procesar_datos_mixtos(arr_enteros):
    """Procesa datos y genera descripciones"""
    resultados = []
    for num in arr_enteros:
        if num % 2 == 0:
            resultados.append(f"Par: {num}")
        else:
            resultados.append(f"Impar: {num}")
    return resultados

# Pruebas
A = np.random.random((3, 4))
B = np.random.random((4, 2))
C = multiplicacion_matrices(A, B)
print(f"Resultado multiplicación matrices: \n{C}")

numeros = np.array([1, 2, 3, 4, 5])
descripciones = procesar_datos_mixtos(numeros)
print("Descripciones:", descripciones)

--------------------------------------------------------
# Ejemplo 10: Comparación con Python puro
def fibonacci_python(n):
    """Fibonacci en Python puro"""
    if n <= 1:
        return n
    return fibonacci_python(n-1) + fibonacci_python(n-2)

@njit
def fibonacci_numba(n):
    """Fibonacci optimizado con Numba"""
    if n <= 1:
        return n
    return fibonacci_numba(n-1) + fibonacci_numba(n-2)

# Medición de tiempo
def comparar_rendimiento():
    n = 35
    
    # Python puro
    inicio = time.time()
    resultado_python = fibonacci_python(n)
    tiempo_python = time.time() - inicio
    
    # Numba (primera ejecución incluye compilación)
    inicio = time.time()
    resultado_numba = fibonacci_numba(n)
    tiempo_numba_primera = time.time() - inicio
    
    # Numba (ejecuciones posteriores)
    inicio = time.time()
    resultado_numba = fibonacci_numba(n)
    tiempo_numba_segunda = time.time() - inicio
    
    print(f"Fibonacci({n}) = {resultado_python}")
    print(f"Python puro: {tiempo_python:.4f} segundos")
    print(f"Numba (primera): {tiempo_numba_primera:.4f} segundos")
    print(f"Numba (segunda): {tiempo_numba_segunda:.4f} segundos")
    print(f"Speedup: {tiempo_python/tiempo_numba_segunda:.2f}x")

comparar_rendimiento()

----------------------------------------------------
# Ejemplo 11: Simulación Monte Carlo para π
@njit(parallel=True)
def monte_carlo_pi(n_muestras):
    """Estimación de π usando Monte Carlo"""
    dentro_circulo = 0
    
    for i in numba.prange(n_muestras):
        x = np.random.random()
        y = np.random.random()
        
        if x**2 + y**2 <= 1.0:
            dentro_circulo += 1
    
    return 4.0 * dentro_circulo / n_muestras

# Ejemplo 12: Procesamiento de imágenes simple
@njit
def aplicar_filtro_umbral(imagen, umbral):
    """Aplica filtro de umbral a una imagen"""
    resultado = np.empty_like(imagen)
    for i in range(imagen.shape[0]):
        for j in range(imagen.shape[1]):
            if imagen[i, j] > umbral:
                resultado[i, j] = 1.0
            else:
                resultado[i, j] = 0.0
    return resultado

# Pruebas
print(f"Estimación de π: {monte_carlo_pi(1000000)}")

# Simular una imagen en escala de grises
imagen_simulada = np.random.random((100, 100))
imagen_filtrada = aplicar_filtro_umbral(imagen_simulada, 0.5)
print(f"Imagen filtrada - píxeles activos: {np.sum(imagen_filtrada)}")

-----------------------------------------------------
# Ejemplo 13: Especificación explícita de tipos
@jit(nopython=True, cache=True)
def funcion_con_cache(a, b):
    """Función con cache para reutilización"""
    return a * b + np.sin(a) * np.cos(b)

# Ejemplo 14: Múltiples firmas de tipos
@jit(['float64(float64, float64)', 'int64(int64, int64)'], nopython=True)
def operacion_flexible(a, b):
    """Función que acepta múltiples tipos"""
    return a * 2 + b * 3

# Pruebas
resultado1 = funcion_con_cache(2.5, 3.7)
resultado2 = operacion_flexible(5.0, 2.0)
resultado3 = operacion_flexible(5, 2)
print(f"Función con cache: {resultado1}")
print(f"Operación flexible (float): {resultado2}")
print(f"Operación flexible (int): {resultado3}")

---------------------------------------------------------

Consejos
@njit en lugar de @jit(nopython=True) para mayor legibilidad

Especifica tipos cuando sea posible para mejor optimización

Usa cache=True para funciones que se llaman múltiples veces

Evita objetos de Python complejos dentro de funciones Numba

Preferir arrays NumPy sobre listas de Python
