import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import time
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        """
        Inicializa el clasificador SVM
        
        Args:
            learning_rate: Tasa de aprendizaje
            lambda_param: Parámetro de regularización
            n_iters: Número de iteraciones
        """
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None
    
    def fit(self, X, y):
        """
        Entrena el modelo SVM
        
        PARTE PARALELIZABLE: El cálculo de gradientes por muestras
        podría distribuirse en múltiples núcleos
        """
        n_samples, n_features = X.shape
        
        # Convertir etiquetas a -1, 1
        y_ = np.where(y <= 0, -1, 1)
        
        # Inicializar parámetros
        self.w = np.zeros(n_features)
        self.b = 0
        
        # Optimización por gradiente descendente
        for epoch in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                
                if condition:
                    # PARTE PARALELIZABLE: Cálculo independiente por muestra
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    # PARTE PARALELIZABLE: Actualización independiente por muestra
                    self.w -= self.lr * (2 * self.lambda_param * self.w - 
                                       np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]
    
    def predict(self, X):
        """
        Realiza predicciones
        
        PARTE PARALELIZABLE: Las predicciones son independientes
        y pueden calcularse en paralelo
        """
        linear_output = np.dot(X, self.w) - self.b
        return np.sign(linear_output)

def predict_chunk(args):
    """
    Función auxiliar para predicción paralela
    Usa variables básicas en lugar de objetos complejos
    """
    chunk, w, b = args
    return np.sign(np.dot(chunk, w) - b)

class SVMParallelPredictor:
    """
    Clase auxiliar para manejar predicción paralela
    sin problemas de serialización
    """
    def __init__(self, w, b):
        self.w = w
        self.b = b
    
    def predict_parallel(self, X, n_workers=None):
        """
        Versión paralelizada de predicción sin problemas de serialización
        """
        if n_workers is None:
            n_workers = mp.cpu_count()
        
        # Convertir a arrays numpy para asegurar compatibilidad
        w_array = np.array(self.w)
        b_value = float(self.b)
        
        # Dividir los datos en chunks
        n_samples = X.shape[0]
        chunk_size = max(1, n_samples // n_workers)
        
        chunks = []
        for i in range(0, n_samples, chunk_size):
            chunk_end = min(i + chunk_size, n_samples)
            chunks.append((X[i:chunk_end], w_array, b_value))
        
        # Usar ThreadPoolExecutor en lugar de multiprocessing para evitar problemas de pickle
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(predict_chunk, chunks))
        
        return np.concatenate(results)

def cargar_datos_uci():
    """
    Carga datos desde el repositorio UCI
    
    PARTE PARALELIZABLE: La descarga y preprocesamiento de datos
    podría paralelizarse si hay múltiples fuentes
    """
    print("Cargando datos desde UCI repository...")
    
    try:
        # Cargar el dataset Iris como ejemplo
        iris = fetch_ucirepo(name='Iris')
        
        # Obtener características y target
        X = iris.data.features
        y = iris.data.targets
        
        # Convertir a numpy arrays
        X = X.values
        y = y.values.ravel()
        
        # Codificar etiquetas a 0,1
        le = LabelEncoder()
        y = le.fit_transform(y)
        
        print(f"Datos cargados: {X.shape[0]} muestras, {X.shape[1]} características")
        return X, y
        
    except Exception as e:
        print(f"Error cargando datos: {e}")
        print("Usando datos de ejemplo...")
        # Datos de ejemplo si falla la descarga
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=1000, n_features=4, 
                                 n_redundant=0, n_informative=2,
                                 n_clusters_per_class=1, random_state=42)
        return X, y

def preprocesar_datos(X, y):
    """
    Preprocesa los datos para SVM
    """
    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Estandarizar características
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, scaler

def entrenar_svm_secuencial(X_train, y_train):
    """
    Entrenamiento secuencial de SVM
    """
    print("Entrenando SVM (versión secuencial)...")
    start_time = time.time()
    
    svm = SVM(learning_rate=0.001, lambda_param=0.01, n_iters=1000)
    svm.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f"Tiempo de entrenamiento secuencial: {training_time:.4f} segundos")
    
    return svm, training_time

def evaluar_modelo(svm, X_test, y_test, usar_paralelo=False):
    """
    Evalúa el modelo entrenado
    """
    print("Evaluando modelo...")
    
    if usar_paralelo:
        # Usar el predictor paralelo sin problemas de serialización
        predictor = SVMParallelPredictor(svm.w, svm.b)
        y_pred = predictor.predict_parallel(X_test)
    else:
        y_pred = svm.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Precisión: {accuracy:.4f}")
    print("\nReporte de clasificación:")
    print(classification_report(y_test, y_pred, zero_division=1))
    
    return accuracy

def demostrar_paralelismo_prediccion(svm, X_test):
    """
    Demuestra la diferencia entre predicción secuencial y paralela
    sin problemas de serialización
    """
    print("\n" + "="*50)
    print("DEMOSTRACIÓN DE PARALELISMO EN PREDICCIÓN")
    print("="*50)
    
    # Predicción secuencial
    start_time = time.time()
    y_pred_seq = svm.predict(X_test)
    tiempo_seq = time.time() - start_time
    
    # Predicción paralela usando el predictor seguro
    predictor = SVMParallelPredictor(svm.w, svm.b)
    start_time = time.time()
    y_pred_par = predictor.predict_parallel(X_test)
    tiempo_par = time.time() - start_time
    
    print(f"Tiempo predicción secuencial: {tiempo_seq:.6f} segundos")
    print(f"Tiempo predicción paralela: {tiempo_par:.6f} segundos")
    print(f"Speedup: {tiempo_seq/tiempo_par:.2f}x")
    print(f"¿Predicciones iguales? {np.array_equal(y_pred_seq, y_pred_par)}")

def analizar_partes_paralelizables():
    """
    Análisis de partes paralelizables del algoritmo SVM
    """
    partes_paralelizables = [
        {
            "componente": "Cálculo de Gradientes",
            "descripcion": "Cada muestra calcula su gradiente independientemente",
            "beneficio": "Alto - Operaciones independientes por muestra",
            "implementacion": "Map-Reduce sobre batches de datos",
            "variables_usadas": "X_chunk, y_chunk, w, b (arrays numpy)"
        },
        {
            "componente": "Predicción",
            "descripcion": "Cada predicción es independiente",
            "beneficio": "Alto - Cálculos vectoriales paralelizables",
            "implementacion": "División de dataset en chunks",
            "variables_usadas": "X_chunk, w, b (arrays numpy básicos)"
        },
        {
            "componente": "Preprocesamiento",
            "descripcion": "Estandarización y transformación de características",
            "beneficio": "Moderado - Operaciones por columna",
            "implementacion": "Procesamiento paralelo de columnas",
            "variables_usadas": "X_columns (arrays numpy independientes)"
        },
        {
            "componente": "Validación Cruzada",
            "descripcion": "Múltiples folds se ejecutan independientemente",
            "beneficio": "Alto - Folds independientes",
            "implementacion": "Ejecución concurrente de folds",
            "variables_usadas": "X_train, y_train, X_val, y_val (datos separados)"
        }
    ]
    
    print("\n" + "="*50)
    print("ANÁLISIS DE PARTES PARALELIZABLES")
    print("="*50)
    
    for parte in partes_paralelizables:
        print(f"\n{parte['componente']}:")
        print(f"  Descripción: {parte['descripcion']}")
        print(f"  Beneficio: {parte['beneficio']}")
        print(f"  Implementación: {parte['implementacion']}")
        print(f"  Variables usadas: {parte['variables_usadas']}")

def main():
    """
    Función principal que ejecuta el pipeline completo
    """
    # 1. Cargar datos
    X, y = cargar_datos_uci()
    
    # 2. Preprocesar datos
    X_train, X_test, y_train, y_test, scaler = preprocesar_datos(X, y)
    
    # 3. Entrenar modelo SVM
    svm, tiempo_entrenamiento = entrenar_svm_secuencial(X_train, y_train)
    
    # 4. Evaluar modelo
    accuracy = evaluar_modelo(svm, X_test, y_test)
    
    # 5. Demostrar paralelismo en predicción
    demostrar_paralelismo_prediccion(svm, X_test)
    
    # 6. Análisis de partes paralelizables
    analizar_partes_paralelizables()

if __name__ == "__main__":
    main()