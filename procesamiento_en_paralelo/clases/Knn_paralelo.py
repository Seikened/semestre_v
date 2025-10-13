import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
import math
import random
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict, Any
import multiprocessing as mp

class ParallelKNN:
    def __init__(self, k=3, n_jobs=-1, parallel_method='process'):
        """
        Inicializa el clasificador k-NN paralelo
        
        Parameters:
        k (int): Número de vecinos a considerar
        n_jobs (int): Número de workers paralelos (-1 para usar todos los cores)
        parallel_method (str): 'process' para multiprocessing, 'thread' para threading
        """
        self.k = k
        self.X_train = None
        self.y_train = None
        self.n_jobs = mp.cpu_count() if n_jobs == -1 else n_jobs
        self.parallel_method = parallel_method
    
    def fit(self, X, y):
        """
        Almacena los datos de entrenamiento
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)
    
    def euclidean_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Calcula la distancia euclidiana entre dos puntos (optimizada)
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def _process_single_point(self, test_point: np.ndarray) -> Tuple[int, List[Tuple[float, int]]]:
        """
        Procesa un solo punto de prueba (función auxiliar para paralelización)
        """
        distances = []
        for i, train_point in enumerate(self.X_train):
            dist = self.euclidean_distance(test_point, train_point)
            distances.append((dist, self.y_train[i]))
        
        # Ordenar y seleccionar k vecinos
        distances.sort(key=lambda x: x[0])
        k_nearest = distances[:self.k]
        
        # Votación por mayoría
        votes = {}
        for dist, label in k_nearest:
            votes[label] = votes.get(label, 0) + 1
        
        return max(votes.items(), key=lambda x: x[1])[0]
    
    def _process_batch(self, test_points: np.ndarray) -> List[int]:
        """
        Procesa un lote de puntos de prueba
        """
        return [self._process_single_point(point) for point in test_points]
    
    def predict_parallel(self, X: np.ndarray, batch_size: int = 100) -> np.ndarray:
        """
        Predice las etiquetas usando paralelización
        
        Parameters:
        X (array): Datos de prueba
        batch_size (int): Tamaño del lote para procesamiento paralelo
        
        Returns:
        array: Predicciones
        """
        X_test = np.array(X)
        n_samples = len(X_test)
        
        # Dividir en lotes
        batches = []
        for i in range(0, n_samples, batch_size):
            batches.append(X_test[i:i + batch_size])
        
        predictions = []
        
        # Seleccionar el método de paralelización
        if self.parallel_method == 'process':
            Executor = ProcessPoolExecutor
        else:
            Executor = ThreadPoolExecutor
        
        with Executor(max_workers=self.n_jobs) as executor:
            # Enviar lotes para procesamiento paralelo
            future_to_batch = {
                executor.submit(self._process_batch, batch): i 
                for i, batch in enumerate(batches)
            }
            
            # Recolectar resultados en orden
            results = [None] * len(batches)
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    results[batch_idx] = future.result()
                except Exception as e:
                    print(f"Error procesando lote {batch_idx}: {e}")
                    results[batch_idx] = []
            
            # Combinar resultados
            for result in results:
                predictions.extend(result)
        
        return np.array(predictions)
    
    def predict_sequential(self, X: np.ndarray) -> np.ndarray:
        """
        Versión secuencial para comparación
        """
        X_test = np.array(X)
        predictions = []
        
        for test_point in X_test:
            predictions.append(self._process_single_point(test_point))
        
        return np.array(predictions)
    
    def accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calcula la precisión del modelo
        """
        correct = np.sum(y_true == y_pred)
        return correct / len(y_true)

def cargar_dataset():
    """
    Muestra los datasets disponibles y permite elegir uno
    """
    print("=" * 60)
    print("DATASETS DISPONIBLES DE UCI REPOSITORY")
    print("=" * 60)
    
    datasets = {
        1: {"id": 53, "name": "Iris", "desc": "Clasificación de flores iris", "samples": 150},
        2: {"id": 109, "name": "Wine", "desc": "Clasificación de vinos", "samples": 178},
        3: {"id": 15, "name": "Breast Cancer", "desc": "Diagnóstico de cáncer de mama", "samples": 569},
        4: {"id": 1, "name": "Abalone", "desc": "Predicción de edad de abulones", "samples": 4177},
        5: {"id": 12, "name": "Heart Disease", "desc": "Diagnóstico de enfermedades cardíacas", "samples": 303}
    }
    
    for key, value in datasets.items():
        print(f"{key}. {value['name']} - {value['desc']} ({value['samples']} muestras)")
    
    print("=" * 60)
    
    while True:
        try:
            choice = int(input("Selecciona el número del dataset (1-5): "))
            if 1 <= choice <= 5:
                return datasets[choice]["id"], datasets[choice]["name"]
            else:
                print("Por favor, selecciona un número entre 1 y 5.")
        except ValueError:
            print("Entrada inválida. Por favor ingresa un número.")

def preprocesar_datos(dataset):
    """
    Preprocesa los datos del dataset
    """
    X = dataset.data.features
    y = dataset.data.targets
    
    # Convertir a numpy arrays y manejar valores nulos
    X = np.array(X)
    y = np.array(y).flatten()
    
    # Normalizar características
    X_normalized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    
    return X_normalized, y, dataset.data.feature_names, dataset.data.target_names

def dividir_datos(X, y, test_size=0.2, random_state=42):
    """
    Divide los datos en conjuntos de entrenamiento y prueba
    """
    np.random.seed(random_state)
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

def benchmark_performance(knn: ParallelKNN, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
    """
    Benchmark de rendimiento entre versión paralela y secuencial
    """
    results = {}
    
    # Benchmark versión paralela
    start_time = time.time()
    predictions_parallel = knn.predict_parallel(X_test)
    parallel_time = time.time() - start_time
    accuracy_parallel = knn.accuracy(y_test, predictions_parallel)
    
    # Benchmark versión secuencial
    start_time = time.time()
    predictions_sequential = knn.predict_sequential(X_test)
    sequential_time = time.time() - start_time
    accuracy_sequential = knn.accuracy(y_test, predictions_sequential)
    
    results.update({
        'parallel_time': parallel_time,
        'sequential_time': sequential_time,
        'speedup': sequential_time / parallel_time,
        'accuracy_parallel': accuracy_parallel,
        'accuracy_sequential': accuracy_sequential,
        'n_cores_used': knn.n_jobs,
        'method': knn.parallel_method
    })
    
    return results

def main():
    """
    Función principal con benchmarking de rendimiento
    """
    print("IMPLEMENTACIÓN PARALELA DE ALGORITMO k-NN")
    print("=" * 50)
    
    # Seleccionar dataset
    dataset_id, dataset_name = cargar_dataset()
    
    try:
        # Cargar dataset
        print(f"\nCargando dataset: {dataset_name}...")
        dataset = fetch_ucirepo(id=dataset_id)
        
        # Preprocesar datos
        X, y, feature_names, target_names = preprocesar_datos(dataset)
        
        print(f"Dataset: {dataset_name}")
        print(f"Muestras: {X.shape[0]}, Características: {X.shape[1]}")
        print(f"Clases: {len(np.unique(y))}")
        print(f"Cores disponibles: {mp.cpu_count()}")
        
        # Dividir datos
        X_train, X_test, y_train, y_test = dividir_datos(X, y, test_size=0.3)
        print(f"\nEntrenamiento: {X_train.shape[0]}, Prueba: {X_test.shape[0]}")
        
        # Configurar parámetros
        k_value = int(input("\nIngresa el valor de k (número de vecinos): "))
        n_workers = int(input(f"Ingresa número de workers (1-{mp.cpu_count()}, -1 para todos): ") or -1)
        
        parallel_method = input("Método de paralelización (process/thread): ").lower() or 'process'
        batch_size = int(input("Tamaño del lote (recomendado 50-200): ") or 100)
        
        # Crear y entrenar modelo
        knn = ParallelKNN(k=k_value, n_jobs=n_workers, parallel_method=parallel_method)
        knn.fit(X_train, y_train)
        
        print(f"\nConfiguración: k={k_value}, workers={knn.n_jobs}, método={parallel_method}")
        print("=" * 50)
        
        # Benchmark de rendimiento
        print("Ejecutando benchmarking...")
        results = benchmark_performance(knn, X_test, y_test)
        
        # Mostrar resultados
        print("\nRESULTADOS DEL BENCHMARK:")
        print("=" * 40)
        print(f"Tiempo paralelo:    {results['parallel_time']:.4f} segundos")
        print(f"Tiempo secuencial:  {results['sequential_time']:.4f} segundos")
        print(f"Speedup:            {results['speedup']:.2f}x")
        print(f"Precisión paralela:  {results['accuracy_parallel']:.4f}")
        print(f"Precisión secuencial: {results['accuracy_sequential']:.4f}")
        print(f"Cores utilizados:    {results['n_cores_used']}")
        
        # Comparar diferentes configuraciones
        print("\n" + "=" * 50)
        comparar = input("¿Deseas comparar diferentes configuraciones? (s/n): ").lower()
        
        if comparar == 's':
            print("\nCOMPARANDO DIFERENTES CONFIGURACIONES:")
            print("k | Workers | Método  | Tiempo (s) | Speedup | Precisión")
            print("-" * 55)
            
            configs = [
                {'k': 3, 'workers': 2, 'method': 'process'},
                {'k': 3, 'workers': 4, 'method': 'process'},
                {'k': 5, 'workers': 2, 'method': 'process'},
                {'k': 5, 'workers': 4, 'method': 'process'},
                {'k': 3, 'workers': 4, 'method': 'thread'},
            ]
            
            for config in configs:
                knn_temp = ParallelKNN(k=config['k'], n_jobs=config['workers'], 
                                      parallel_method=config['method'])
                knn_temp.fit(X_train, y_train)
                
                start_time = time.time()
                preds = knn_temp.predict_parallel(X_test, batch_size=batch_size)
                exec_time = time.time() - start_time
                acc = knn_temp.accuracy(y_test, preds)
                
                # Obtener tiempo secuencial de referencia
                knn_ref = ParallelKNN(k=config['k'], n_jobs=1, parallel_method='process')
                knn_ref.fit(X_train, y_train)
                start_time = time.time()
                knn_ref.predict_sequential(X_test)
                seq_time = time.time() - start_time
                
                speedup = seq_time / exec_time if exec_time > 0 else 0
                
                print(f"{config['k']} | {config['workers']:7} | {config['method']:7} | {exec_time:9.4f} | {speedup:6.2f}x | {acc:.4f}")
    
    except Exception as e:
        print(f"Error: {e}")
        print("Asegúrate de tener instaladas las librerías requeridas")

if __name__ == "__main__":
    # Configuración para multiprocessing en Windows
    mp.freeze_support()
    main()