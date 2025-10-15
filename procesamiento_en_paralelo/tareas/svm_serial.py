# svm_numba_paralelo.py
from time import perf_counter
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from numba import njit, prange, get_num_threads, set_num_threads

# =========================
# Configuración de precisión y hilos
# =========================
USAR_FLOAT32 = True
DTYPE = np.float32 if USAR_FLOAT32 else np.float64

# Ajusta los hilos de Numba aquí
NUCLEOS_DESEADOS = min(8, os.cpu_count() or 1)
print(f"Hilos Numba configurados: {NUCLEOS_DESEADOS} / {os.cpu_count()}")
set_num_threads(NUCLEOS_DESEADOS)

# =========================
# Kernels NUMBA (paralelos)
# =========================

@njit(parallel=True, fastmath=True, cache=True)
def calcular_gradiente_batch(X, y, w, b, lambda_param):
    """
    Subgradiente batch del SVM hinge.
    Paraleliza sobre características para evitar condiciones de carrera.
    grad_w = 2*lambda*w - (1/N) * sum_{i: margen_i<1} (y_i * x_i)
    grad_b = - (1/N) * sum_{i: margen_i<1} (y_i)
    """
    n_muestras, n_caracts = X.shape

    # Suma por característica (paralela) de los violadores del margen
    suma_yx = np.zeros(n_caracts, dtype=X.dtype)
    for j in prange(n_caracts):
        s = 0.0
        for i in range(n_muestras):
            margen = y[i] * (np.dot(X[i], w) - b)
            if margen < 1.0:
                s += y[i] * X[i, j]
        suma_yx[j] = s

    # Suma escalar para el sesgo (reducción escalar paralela)
    suma_y_viol = 0.0
    for i in prange(n_muestras):
        margen = y[i] * (np.dot(X[i], w) - b)
        if margen < 1.0:
            suma_y_viol += y[i]

    grad_w = 2.0 * lambda_param * w - (suma_yx / n_muestras)
    grad_b = - (suma_y_viol / n_muestras)
    return grad_w, grad_b


@njit(parallel=True, fastmath=True, cache=True)
def decision_lineal(X, w, b, out):
    """
    out = X @ w - b, paralelizando por muestras.
    """
    n_muestras = X.shape[0]
    for i in prange(n_muestras):
        out[i] = np.dot(X[i], w) - b


@njit(parallel=True, fastmath=True, cache=True)
def calcular_metricas_numba(y_real, y_pred):
    """
    Métricas básicas en Numba.
    """
    n = y_real.size
    aciertos = 0
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in prange(n):
        if y_real[i] == y_pred[i]:
            aciertos += 1
        if y_real[i] == 1 and y_pred[i] == 1:
            tp += 1
        elif y_real[i] == -1 and y_pred[i] == -1:
            tn += 1
        elif y_real[i] == -1 and y_pred[i] == 1:
            fp += 1
        elif y_real[i] == 1 and y_pred[i] == -1:
            fn += 1

    accuracy = aciertos / n
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # Matriz de confusión en orden [[tn, fp], [fn, tp]]
    matriz = np.empty((2, 2), dtype=np.int64)
    matriz[0, 0] = tn
    matriz[0, 1] = fp
    matriz[1, 0] = fn
    matriz[1, 1] = tp

    return accuracy, precision, recall, f1, matriz


# =========================
# Modelo SVM con Numba
# =========================

class SVM_DesdeCero_Numby:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None
        self.historial_tiempos = []

    def fit(self, X, y):
        """
        Entrenamiento batch con subgradiente del hinge, usando kernels Numba paralelos.
        """
        X = np.asarray(X, dtype=DTYPE)
        y = np.asarray(np.where(y <= 0, -1, 1), dtype=DTYPE)

        n_muestras, n_caracts = X.shape
        self.w = np.zeros(n_caracts, dtype=DTYPE)
        self.b = DTYPE(0.0)

        # WARM-UP para compilar kernels
        self._warm_up(X, y)

        # Entrenamiento
        for _ in range(self.n_iters):
            t0 = perf_counter()
            grad_w, grad_b = calcular_gradiente_batch(X, y, self.w, self.b, self.lambda_param)
            # Update
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
            t1 = perf_counter()
            self.historial_tiempos.append(t1 - t0)

    def predict(self, X):
        """
        Predicción con kernel Numba paralelizado.
        """
        X = np.asarray(X, dtype=DTYPE)
        salida = np.empty(X.shape[0], dtype=DTYPE)
        decision_lineal(X, self.w, self.b, salida)
        # np.sign en CPU vectorizado; convertimos a {-1, 1}, evitando 0
        pred = np.where(salida >= 0, 1, -1).astype(DTYPE)
        return pred

    def get_decision_boundary(self, X):
        """
        Igual que el original, sólo lectura de self.w/self.b.
        """
        if self.w is None or self.w.size != 2:
            raise ValueError("Solo funciona para características 2D")
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x1_vals = np.linspace(x_min, x_max, 200, dtype=DTYPE)
        x2_vals = (self.b - self.w[0] * x1_vals) / self.w[1]
        margin_upper = (self.b + 1 - self.w[0] * x1_vals) / self.w[1]
        margin_lower = (self.b - 1 - self.w[0] * x1_vals) / self.w[1]
        return x1_vals, x2_vals, margin_upper, margin_lower

    def _warm_up(self, X, y):
        """
        Compila los kernels una vez con tamaños pequeños.
        """
        # Pequeño subconjunto para compilar
        n0 = min(64, X.shape[0])
        X0 = X[:n0]
        y0 = y[:n0]
        # Warm gradiente
        _ = calcular_gradiente_batch(X0, y0, np.zeros(X0.shape[1], dtype=DTYPE), DTYPE(0.0), self.lambda_param)
        # Warm decision
        salida0 = np.empty(X0.shape[0], dtype=DTYPE)
        decision_lineal(X0, np.zeros(X0.shape[1], dtype=DTYPE), DTYPE(0.0), salida0)


# =========================
# Utilidades (datos/visualización/métricas)
# =========================

def generar_datos_ejemplo():
    X, y = make_blobs(
        n_samples=1000, centers=2, n_features=2,
        random_state=42, cluster_std=1.2
    )
    X = X.astype(DTYPE)
    y = np.where(y == 0, -1, 1).astype(DTYPE)
    return X, y

def visualizar_resultados(X, y, modelo, titulo="SVM desde Cero (Numba)"):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(X[y == -1, 0], X[y == -1, 1], color='red', marker='o', label='Clase -1', alpha=0.6)
    plt.scatter(X[y == 1, 0],  X[y == 1, 1],  color='blue', marker='s', label='Clase 1',  alpha=0.6)

    try:
        x1, x2, up, low = modelo.get_decision_boundary(X)
        plt.plot(x1, x2, 'k-', linewidth=2, label='Límite de decisión')
        plt.plot(x1, up, 'k--', linewidth=1, alpha=0.7, label='Márgenes')
        plt.plot(x1, low, 'k--', linewidth=1, alpha=0.7)

        # Vectores soporte aproximados (margen ~ 1)
        margenes = y * (X @ modelo.w - modelo.b)
        vs = np.where(np.abs(margenes - 1) < 1e-3)[0]
        if vs.size > 0:
            plt.scatter(X[vs, 0], X[vs, 1], s=100, facecolors='none', edgecolors='green',
                        linewidths=2, label='Vectores soporte aprox.')
    except ValueError as e:
        print(f"Advertencia: {e}")

    plt.xlabel('Característica 1')
    plt.ylabel('Característica 2')
    plt.title(titulo)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    if hasattr(modelo, "historial_tiempos") and len(modelo.historial_tiempos) > 0:
        tiempos = np.array(modelo.historial_tiempos, dtype=np.float64)
        plt.plot(np.arange(len(tiempos)), tiempos * 1000.0, 'g-', linewidth=2)
        plt.xlabel('Iteración')
        plt.ylabel('Tiempo por iteración (ms)')
        plt.title('Rendimiento por iteración (Numba)')
        plt.grid(True, alpha=0.3)
    else:
        iteraciones = np.arange(1, 101)
        curva = 1 - np.exp(-iteraciones / 30)
        plt.plot(iteraciones, curva * 100, 'g-', linewidth=2)
        plt.xlabel('Iteraciones')
        plt.ylabel('Precisión (%)')
        plt.title('Curva de Aprendizaje (Simulada)')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def calcular_metricas(y_real, y_pred):
    y_real = y_real.astype(DTYPE)
    y_pred = y_pred.astype(DTYPE)
    acc, prec, rec, f1, matriz = calcular_metricas_numba(y_real, y_pred)
    return {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1_score': float(f1),
        'matriz_confusion': [[int(matriz[0,0]), int(matriz[0,1])],
                             [int(matriz[1,0]), int(matriz[1,1])]]
    }

# =========================
# Demo de uso
# =========================
if __name__ == "__main__":
    print("=== SVM desde Cero con NUMBA Paralelo ===")
    print(f"Numba threads efectivos: {get_num_threads()} | dtype: {DTYPE}")

    print("\nGenerando datos de ejemplo...")
    X, y = generar_datos_ejemplo()
    print(f"Forma de X: {X.shape} | Etiquetas: {np.unique(y)}")

    print("\nEntrenando modelo...")
    svm = SVM_DesdeCero_Numby(learning_rate=0.01, lambda_param=0.01, n_iters=200)
    t0 = perf_counter()
    svm.fit(X, y)
    t1 = perf_counter()
    print(f"Tiempo total de entrenamiento: {t1 - t0:.6f} s")

    print("\nRealizando predicciones...")
    t0 = perf_counter()
    y_pred = svm.predict(X)
    t1 = perf_counter()
    print(f"Tiempo de predicción: {t1 - t0:.6f} s")

    metricas = calcular_metricas(y, y_pred)
    print("\n=== Resultados ===")
    print(f"Precisión: {metricas['accuracy']:.4f}")
    print(f"Precisión (PPV): {metricas['precision']:.4f}")
    print(f"Recall: {metricas['recall']:.4f}")
    print(f"Puntaje F1: {metricas['f1_score']:.4f}")
    print(f"Pesos (w): {svm.w}")
    print(f"Sesgo (b): {svm.b:.4f}")
    print(f"Matriz de confusión: {metricas['matriz_confusion']}")

    print("\nGenerando visualización...")
    visualizar_resultados(X, y, svm, "SVM Implementado desde Cero (Numba Paralelo)")

    print("\n=== Prueba con nuevos datos ===")
    nuevos_datos = np.array([[2, 2], [-2, -2], [0, 0]], dtype=DTYPE)
    pred_nuevas = svm.predict(nuevos_datos)
    for i, dato in enumerate(nuevos_datos):
        print(f"Dato {dato} → Predicción: {int(pred_nuevas[i])}")
