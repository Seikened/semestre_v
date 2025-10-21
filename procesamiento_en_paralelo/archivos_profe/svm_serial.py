import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class SVM_DesdeCero:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        """
        Inicializa el clasificador SVM
        
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
        Entrena el modelo SVM
        
        Parámetros:
        - X: Características (array 2D)
        - y: Etiquetas (array 1D, valores -1 o 1)
        """
        n_samples, n_features = X.shape
        
        # Convertir etiquetas a -1 y 1 si es necesario
        y_ = np.where(y <= 0, -1, 1)
        
        # Inicializar parámetros
        self.w = np.zeros(n_features)
        self.b = 0
        
        # Descenso de gradiente
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # Condición para vectores de soporte
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                
                if condition:
                    # Actualizar pesos sin penalización
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    # Actualizar pesos con penalización
                    self.w -= self.lr * (2 * self.lambda_param * self.w - 
                                        np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]
    
    def predict(self, X):
        """
        Realiza predicciones
        
        Parámetros:
        - X: Características a predecir
        
        Retorna:
        - Predicciones (-1 o 1)
        """
        linear_output = np.dot(X, self.w) - self.b
        return np.sign(linear_output)
    
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

def visualizar_resultados(X, y, modelo, titulo="SVM desde Cero"):
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
        predicciones = modelo.predict(X)
        margenes = y * (np.dot(X, modelo.w) - modelo.b)
        vectores_soporte = np.where(np.abs(margenes - 1) < 1e-3)[0]
        
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
    
    # Subplot 2: Evolución de los pesos (simulación)
    plt.subplot(1, 2, 2)
    
    # Simular evolución de precisión (en implementación real, guardar historial)
    iteraciones = np.arange(modelo.n_iters)
    # Para demostración, creamos una curva de aprendizaje simulada
    precision_simulada = 1 - np.exp(-iteraciones / (modelo.n_iters / 3))
    
    plt.plot(iteraciones, precision_simulada * 100, 'g-', linewidth=2)
    plt.xlabel('Iteraciones')
    plt.ylabel('Precisión (%)')
    plt.title('Curva de Aprendizaje (Simulada)')
    plt.grid(True, alpha=0.3)
    
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

# Ejemplo de uso
if __name__ == "__main__":
    print("=== SVM Implementado desde Cero ===")
    
    # 1. Generar datos de ejemplo
    print("Generando datos de ejemplo...")
    X, y = generar_datos_ejemplo()
    
    print(f"Forma de X: {X.shape}")
    print(f"Etiquetas únicas: {np.unique(y)}")
    
    # 2. Crear y entrenar el modelo
    print("\nEntrenando modelo SVM...")
    svm = SVM_DesdeCero(learning_rate=0.001, lambda_param=0.01, n_iters=1000)
    svm.fit(X, y)
    
    # 3. Realizar predicciones
    print("Realizando predicciones...")
    y_pred = svm.predict(X)
    
    # 4. Calcular métricas
    metricas = calcular_metricas(y, y_pred)
    
    print("\n=== Resultados ===")
    print(f"Precisión: {metricas['accuracy']:.4f}")
    print(f"Precisión: {metricas['precision']:.4f}")
    print(f"Recall: {metricas['recall']:.4f}")
    print(f"Puntaje F1: {metricas['f1_score']:.4f}")
    print(f"Pesos (w): {svm.w}")
    print(f"Sesgo (b): {svm.b:.4f}")
    
    # 5. Visualizar resultados
    print("\nGenerando visualización...")
    visualizar_resultados(X, y, svm, "SVM Implementado desde Cero")
    
    # 6. Probar con nuevos datos
    print("\n=== Prueba con nuevos datos ===")
    nuevos_datos = np.array([[2, 2], [-2, -2], [0, 0]])
    predicciones_nuevas = svm.predict(nuevos_datos)
    
    for i, dato in enumerate(nuevos_datos):
        print(f"Dato {dato} → Predicción: {predicciones_nuevas[i]}")