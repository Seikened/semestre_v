import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
import math
import random

class KNN:
    def __init__(self, k=3):
        """
        Inicializa el clasificador k-NN
        
        Parameters:
        k (int): Número de vecinos a considerar
        """
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        """
        Almacena los datos de entrenamiento
        
        Parameters:
        X (array-like): Características de entrenamiento
        y (array-like): Etiquetas de entrenamiento
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)
    
    def euclidean_distance(self, x1, x2):
        """
        Calcula la distancia euclidiana entre dos puntos
        
        Parameters:
        x1, x2 (array): Puntos a comparar
        
        Returns:
        float: Distancia euclidiana
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def predict(self, X):
        """
        Predice las etiquetas para los datos de prueba
        
        Parameters:
        X (array-like): Datos de prueba
        
        Returns:
        array: Predicciones
        """
        X_test = np.array(X)
        predictions = []
        
        for test_point in X_test:
            # Calcular distancias a todos los puntos de entrenamiento
            distances = []
            for i, train_point in enumerate(self.X_train):
                dist = self.euclidean_distance(test_point, train_point)
                distances.append((dist, self.y_train[i]))
            
            # Ordenar por distancia y seleccionar los k vecinos más cercanos
            distances.sort(key=lambda x: x[0])
            k_nearest = distances[:self.k]
            
            # Votación por mayoría
            votes = {}
            for dist, label in k_nearest:
                if label in votes:
                    votes[label] += 1
                else:
                    votes[label] = 1
            
            # Seleccionar la etiqueta con más votos
            predicted_label = max(votes.items(), key=lambda x: x[1])[0]
            predictions.append(predicted_label)
        
        return np.array(predictions)
    
    def accuracy(self, y_true, y_pred):
        """
        Calcula la precisión del modelo
        
        Parameters:
        y_true (array): Etiquetas reales
        y_pred (array): Etiquetas predichas
        
        Returns:
        float: Precisión del modelo
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
        1: {"id": 53, "name": "Iris", "desc": "Clasificación de flores iris"},
        2: {"id": 109, "name": "Wine", "desc": "Clasificación de vinos"},
        3: {"id": 15, "name": "Breast Cancer", "desc": "Diagnóstico de cáncer de mama"},
        4: {"id": 1, "name": "Abalone", "desc": "Predicción de edad de abulones"},
        5: {"id": 12, "name": "Heart Disease", "desc": "Diagnóstico de enfermedades cardíacas"}
    }
    
    for key, value in datasets.items():
        print(f"{key}. {value['name']} - {value['desc']}")
    
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
    
    Parameters:
    dataset: Dataset de UCI
    
    Returns:
    tuple: (X, y, feature_names, target_names)
    """
    # Obtener características y etiquetas
    X = dataset.data.features
    y = dataset.data.targets
    
    # Convertir a numpy arrays
    X = np.array(X)
    y = np.array(y).flatten()  # Aplanar el array si es necesario
    
    # Verificar si hay valores nulos
    if np.isnan(X).any():
        print("Advertencia: El dataset contiene valores nulos. Se reemplazarán con la media.")
        for i in range(X.shape[1]):
            col_mean = np.nanmean(X[:, i])
            X[:, i] = np.nan_to_num(X[:, i], nan=col_mean)
    
    # Normalizar características
    X_normalized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    
    return X_normalized, y, dataset.data.feature_names, dataset.data.target_names

def dividir_datos(X, y, test_size=0.2, random_state=None):
    """
    Divide los datos en conjuntos de entrenamiento y prueba
    
    Parameters:
    X (array): Características
    y (array): Etiquetas
    test_size (float): Proporción para prueba
    random_state (int): Semilla para reproducibilidad
    
    Returns:
    tuple: (X_train, X_test, y_train, y_test)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    
    # Mezclar índices
    indices = np.random.permutation(n_samples)
    
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test

def main():
    """
    Función principal que ejecuta el algoritmo k-NN
    """
    print("IMPLEMENTACIÓN DE ALGORITMO k-NN DESDE CERO")
    print("=" * 50)
    
    # Seleccionar dataset
    dataset_id, dataset_name = cargar_dataset()
    
    try:
        # Cargar dataset desde UCI
        print(f"\nCargando dataset: {dataset_name}...")
        dataset = fetch_ucirepo(id=dataset_id)
        
        # Preprocesar datos
        X, y, feature_names, target_names = preprocesar_datos(dataset)
        
        print(f"Dataset cargado: {dataset_name}")
        print(f"Características: {X.shape[1]}")
        print(f"Muestras: {X.shape[0]}")
        print(f"Clases: {len(np.unique(y))}")
        
        # Dividir datos
        X_train, X_test, y_train, y_test = dividir_datos(X, y, test_size=0.3, random_state=42)
        
        print(f"\nDatos de entrenamiento: {X_train.shape[0]} muestras")
        print(f"Datos de prueba: {X_test.shape[0]} muestras")
        
        # Elegir valor de k
        print("\n" + "=" * 30)
        k_value = int(input("Ingresa el valor de k (número de vecinos): "))
        
        # Crear y entrenar modelo
        knn = KNN(k=k_value)
        knn.fit(X_train, y_train)
        
        # Realizar predicciones
        print("\nRealizando predicciones...")
        predictions = knn.predict(X_test)
        
        # Calcular precisión
        accuracy = knn.accuracy(y_test, predictions)
        print(f"\nPrecisión del modelo: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Mostrar algunas predicciones
        print("\nPrimeras 10 predicciones vs valores reales:")
        print("Predicción | Real")
        print("-" * 20)
        for i in range(min(10, len(predictions))):
            print(f"{predictions[i]:9} | {y_test[i]}")
        
        # Opción para probar con diferentes valores de k
        print("\n" + "=" * 50)
        probar_otros_k = input("¿Deseas probar con diferentes valores de k? (s/n): ").lower()
        
        if probar_otros_k == 's':
            k_values = [1, 3, 5, 7, 9]
            print("\nProbando diferentes valores de k:")
            print("k  | Precisión")
            print("-" * 15)
            
            for k in k_values:
                knn_temp = KNN(k=k)
                knn_temp.fit(X_train, y_train)
                pred_temp = knn_temp.predict(X_test)
                acc_temp = knn_temp.accuracy(y_test, pred_temp)
                print(f"{k:2} | {acc_temp:.4f}")
    
    except Exception as e:
        print(f"Error al cargar el dataset: {e}")
        print("Asegúrate de tener instalada la librería: pip install ucimlrepo")

if __name__ == "__main__":
    main()