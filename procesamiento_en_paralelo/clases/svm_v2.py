import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from numba import njit, prange

@njit
def convert_labels(y):
    """Convierte etiquetas a -1 y 1 de forma eficiente"""
    y_ = np.empty_like(y)
    for i in prange(len(y)):
        if y[i] <= 0:
            y_[i] = -1
        else:
            y_[i] = 1
    return y_

@njit(parallel=True)
def parallel_gradient_descent(X, y_, w, b, learning_rate, lambda_param, n_iters):
    """Descenso de gradiente paralelizado"""
    n_samples, n_features = X.shape
    
    for _ in range(n_iters):
        # Paralelizar el procesamiento de muestras
        for idx in prange(n_samples):
            x_i = X[idx]
            y_i = y_[idx]
            
            # Calcular condición del margen
            linear_output = np.dot(x_i, w) - b
            condition = y_i * linear_output >= 1
            
            if condition:
                # Actualizar solo pesos (sin término de pérdida)
                for j in prange(n_features):
                    w[j] -= learning_rate * (2 * lambda_param * w[j])
            else:
                # Actualizar pesos y sesgo
                for j in prange(n_features):
                    w[j] -= learning_rate * (2 * lambda_param * w[j] - x_i[j] * y_i)
                b -= learning_rate * y_i
    
    return w, b

@njit(parallel=True)
def parallel_predict(X, w, b):
    """Predicción paralelizada"""
    n_samples = X.shape[0]
    predictions = np.empty(n_samples)
    
    for i in prange(n_samples):
        linear_output = np.dot(X[i], w) - b
        predictions[i] = 1.0 if linear_output >= 0 else -1.0
    
    return predictions

@njit(parallel=True)
def calculate_margins_parallel(X, y, w, b):
    """Cálculo paralelizado de márgenes"""
    n_samples = X.shape[0]
    margins = np.empty(n_samples)
    
    for i in prange(n_samples):
        margins[i] = y[i] * (np.dot(X[i], w) - b)
    
    return margins

class SVM_Paralelo:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        """
        Inicializa el clasificador SVM paralelo
        
        Parámetros:
        - learning_rate: Tasa de aprendizaje
        - lambda_param: Parámetro de regularización
        - n_iters: Número de iteraciones
        """
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None
    
    def fit(self, X, y):
        """
        Entrena el modelo SVM usando paralelización
        """
        n_samples, n_features = X.shape
        
        # Convertir etiquetas a -1 y 1
        y_ = convert_labels(y)
        
        # Inicializar parámetros
        self.w = np.zeros(n_features)
        self.b = 0.0
        
        # Entrenamiento paralelizado
        self.w, self.b = parallel_gradient_descent(
            X, y_, self.w, self.b, self.lr, self.lambda_param, self.n_iters
        )
    
    def predict(self, X):
        """
        Realiza predicciones usando paralelización
        """
        return parallel_predict(X, self.w, self.b)
    
    def get_decision_boundary(self, X):
        """
        Calcula la línea de decisión para visualización
        """
        if len(self.w) != 2:
            raise ValueError("Solo funciona para características 2D")
        
        # Calcular límites del gráfico
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        
        # Calcular línea de decisión: w1*x1 + w2*x2 - b = 0
        # Despejar x2: x2 = (b - w1*x1) / w2
        x1_vals = np.linspace(x_min, x_max, 100)
        x2_vals = (self.b - self.w[0] * x1_vals) / self.w[1]
        
        # Líneas de margen: w1*x1 + w2*x2 - b = ±1
        margin_upper = (self.b + 1 - self.w[0] * x1_vals) / self.w[1]
        margin_lower = (self.b - 1 - self.w[0] * x1_vals) / self.w[1]
        
        return x1_vals, x2_vals, margin_upper, margin_lower
    
    def get_support_vectors(self, X, y, tolerance=1e-3):
        """
        Encuentra vectores de soporte usando cálculo paralelizado
        """
        margins = calculate_margins_parallel(X, y, self.w, self.b)
        return np.where(np.abs(margins - 1) < tolerance)[0]

def generar_datos_ejemplo():
    """
    Genera datos de ejemplo para probar el SVM
    """
    # Generar datos linealmente separables
    X, y = make_blobs(n_samples=100, centers=2, 
                      n_features=2, random_state=42, cluster_std=1.2)
    
    # Convertir etiquetas a -1 y 1
    y = np.where(y == 0, -1, 1)
    
    return X, y

def visualizar_resultados(X, y, modelo, titulo="SVM Paralelo"):
    """
    Visualiza los datos y el modelo SVM
    """
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Datos y límite de decisión
    plt.subplot(1, 2, 1)
    
    # Graficar puntos de datos
    plt.scatter(X[y == -1, 0], X[y == -1, 1], 
                color='red', marker='o', label='Clase -1', alpha=0.7)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], 
                color='blue', marker='s', label='Clase 1', alpha=0.7)
    
    try:
        # Graficar límites de decisión y márgenes
        x1_vals, x2_vals, margin_upper, margin_lower = modelo.get_decision_boundary(X)
        
        plt.plot(x1_vals, x2_vals, 'k-', linewidth=2, label='Límite de decisión')
        plt.plot(x1_vals, margin_upper, 'k--', linewidth=1, alpha=0.7, label='Márgenes')
        plt.plot(x1_vals, margin_lower, 'k--', linewidth=1, alpha=0.7)
        
        # Resaltar vectores de soporte
        vectores_soporte = modelo.get_support_vectors(X, y)
        
        plt.scatter(X[vectores_soporte, 0], X[vectores_soporte, 1], 
                   s=100, facecolors='none', edgecolors='green', 
                   linewidths=2, label='Vectores soporte')
        
    except ValueError as e:
        print(f"Advertencia: {e}")
    
    plt.xlabel('Característica 1')
    plt.ylabel('Característica 2')
    plt.title(f'{titulo}\nLímite de Decisión')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Subplot 2: Comparación de velocidad (placeholder)
    plt.subplot(1, 2, 2)
    
    # Mostrar información sobre paralelización
    plt.text(0.1, 0.6, f'Características: {X.shape[1]}', fontsize=12)
    plt.text(0.1, 0.5, f'Muestras: {X.shape[0]}', fontsize=12)
    plt.text(0.1, 0.4, f'Iteraciones: {modelo.n_iters}', fontsize=12)
    plt.text(0.1, 0.3, 'Paralelizado con Numba', fontsize=12, color='green')
    plt.text(0.1, 0.2, 'Usando @njit(parallel=True)', fontsize=10, color='blue')
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    plt.title('Información de Paralelización')
    
    plt.tight_layout()
    plt.show()

def calcular_metricas(y_real, y_pred):
    """
    Calcula métricas de evaluación
    """
    accuracy = np.mean(y_real == y_pred)
    
    # Matriz de confusión básica
    tp = np.sum((y_real == 1) & (y_pred == 1))
    tn = np.sum((y_real == -1) & (y_pred == -1))
    fp = np.sum((y_real == -1) & (y_pred == 1))
    fn = np.sum((y_real == 1) & (y_pred == -1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'matriz_confusion': [[tn, fp], [fn, tp]]
    }

def comparar_rendimiento():
    """Compara el rendimiento entre la versión serial y paralela"""
    import time
    
    # Generar datos más grandes para prueba de rendimiento
    X_large, y_large = make_blobs(n_samples=10000, centers=2, 
                                 n_features=10, random_state=42, cluster_std=2.0)
    y_large = np.where(y_large == 0, -1, 1)
    
    print("=== Comparación de Rendimiento ===")
    
    # Versión paralela
    start_time = time.time()
    svm_paralelo = SVM_Paralelo(learning_rate=0.001, lambda_param=0.01, n_iters=100)
    svm_paralelo.fit(X_large, y_large)
    y_pred_paralelo = svm_paralelo.predict(X_large)
    tiempo_paralelo = time.time() - start_time
    
    print(f"Tiempo versión paralela: {tiempo_paralelo:.4f} segundos")
    print(f"Precisión paralela: {np.mean(y_large == y_pred_paralelo):.4f}")

# Ejemplo de uso
if __name__ == "__main__":
    print("=== SVM Paralelo con Numba ===")
    
    # 1. Generar datos de ejemplo
    print("Generando datos de ejemplo...")
    X, y = generar_datos_ejemplo()
    
    print(f"Forma de X: {X.shape}")
    print(f"Etiquetas únicas: {np.unique(y)}")
    
    # 2. Crear y entrenar el modelo paralelo
    print("\nEntrenando modelo SVM paralelo...")
    svm_paralelo = SVM_Paralelo(learning_rate=0.001, lambda_param=0.01, n_iters=1000)
    svm_paralelo.fit(X, y)
    
    # 3. Realizar predicciones
    print("Realizando predicciones...")
    y_pred = svm_paralelo.predict(X)
    
    # 4. Calcular métricas
    metricas = calcular_metricas(y, y_pred)
    
    print("\n=== Resultados ===")
    print(f"Precisión: {metricas['accuracy']:.4f}")
    print(f"Precisión: {metricas['precision']:.4f}")
    print(f"Recall: {metricas['recall']:.4f}")
    print(f"Puntaje F1: {metricas['f1_score']:.4f}")
    print(f"Pesos (w): {svm_paralelo.w}")
    print(f"Sesgo (b): {svm_paralelo.b:.4f}")
    
    # 5. Visualizar resultados
    print("\nGenerando visualización...")
    visualizar_resultados(X, y, svm_paralelo, "SVM Paralelo con Numba")
        
    # 7. Comparación de rendimiento
    comparar_rendimiento()