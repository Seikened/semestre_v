import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
from typing import List, Tuple, Dict, Any
import time
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Variables globales para evitar problemas de serialización
global_data = {}
global_config = {}

def init_global_variables(X, y, config):
    """Inicializa variables globales para todos los procesos"""
    global global_data, global_config
    global_data = {
        'X': X,
        'y': y,
        'kernel_matrix': None,
        'alpha': np.zeros(X.shape[0]),
        'gradients': np.zeros(X.shape[0])
    }
    global_config = config

def linear_kernel(x1: np.ndarray, x2: np.ndarray) -> float:
    """Kernel lineal"""
    return np.dot(x1, x2)

def rbf_kernel(x1: np.ndarray, x2: np.ndarray, gamma: float = 0.1) -> float:
    """Kernel RBF"""
    return np.exp(-gamma * np.sum((x1 - x2) ** 2))

def compute_kernel_chunk(args: Tuple[int, int, str]) -> Tuple[int, int, float]:
    """Calcula un elemento de la matriz kernel"""
    i, j, kernel_type = args
    global global_data
    
    X = global_data['X']
    if kernel_type == 'linear':
        value = linear_kernel(X[i], X[j])
    else:  # rbf
        value = rbf_kernel(X[i], X[j])
    
    return i, j, value

def compute_kernel_matrix_parallel(X: np.ndarray, kernel_type: str = 'linear', 
                                 n_workers: int = None) -> np.ndarray:
    """Calcula matriz kernel en paralelo usando ProcessPoolExecutor"""
    if n_workers is None:
        n_workers = mp.cpu_count()
    
    n_samples = X.shape[0]
    K = np.zeros((n_samples, n_samples))
    
    # Preparar tareas
    tasks = []
    for i in range(n_samples):
        for j in range(n_samples):
            tasks.append((i, j, kernel_type))
    
    print(f"Calculando matriz kernel con {n_workers} workers...")
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_task = {executor.submit(compute_kernel_chunk, task): task for task in tasks}
        
        for future in as_completed(future_to_task):
            i, j, value = future.result()
            K[i, j] = value
    
    return K

def compute_gradient_chunk(indices: List[int]) -> List[Tuple[int, float]]:
    """Calcula gradientes para un chunk de índices"""
    global global_data, global_config
    
    X = global_data['X']
    y = global_data['y']
    alpha = global_data['alpha']
    K = global_data['kernel_matrix']
    C = global_config['C']
    
    results = []
    for i in indices:
        grad = -1.0
        for j in range(len(alpha)):
            grad += alpha[j] * y[j] * K[i, j]
        results.append((i, grad * y[i]))
    
    return results

def compute_all_gradients_parallel(n_workers: int = None) -> np.ndarray:
    """Calcula todos los gradientes en paralelo"""
    global global_data
    
    if n_workers is None:
        n_workers = mp.cpu_count()
    
    n_samples = global_data['X'].shape[0]
    gradients = np.zeros(n_samples)
    
    # Dividir trabajo en chunks
    chunk_size = max(1, n_samples // n_workers)
    chunks = []
    for i in range(0, n_samples, chunk_size):
        chunks.append(list(range(i, min(i + chunk_size, n_samples))))
    
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        future_to_chunk = {executor.submit(compute_gradient_chunk, chunk): chunk for chunk in chunks}
        
        for future in as_completed(future_to_chunk):
            chunk_results = future.result()
            for i, grad in chunk_results:
                gradients[i] = grad
    
    global_data['gradients'] = gradients
    return gradients

def check_kkt_chunk(indices: List[int]) -> List[Tuple[int, bool]]:
    """Verifica condiciones KKT para un chunk de índices"""
    global global_data, global_config
    
    alpha = global_data['alpha']
    gradients = global_data['gradients']
    y = global_data['y']
    C = global_config['C']
    tolerance = global_config['tolerance']
    
    results = []
    for i in indices:
        alpha_i = alpha[i]
        grad_i = gradients[i]
        y_i = y[i]
        
        if alpha_i < tolerance:
            satisfies = y_i * grad_i >= -tolerance
        elif alpha_i > C - tolerance:
            satisfies = y_i * grad_i <= tolerance
        else:
            satisfies = abs(y_i * grad_i) <= tolerance
        
        results.append((i, satisfies))
    
    return results

def check_kkt_conditions_parallel(n_workers: int = None) -> bool:
    """Verifica condiciones KKT en paralelo"""
    global global_data
    
    if n_workers is None:
        n_workers = mp.cpu_count()
    
    n_samples = global_data['X'].shape[0]
    
    # Dividir trabajo en chunks
    chunk_size = max(1, n_samples // n_workers)
    chunks = []
    for i in range(0, n_samples, chunk_size):
        chunks.append(list(range(i, min(i + chunk_size, n_samples))))
    
    all_satisfied = True
    
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        future_to_chunk = {executor.submit(check_kkt_chunk, chunk): chunk for chunk in chunks}
        
        for future in as_completed(future_to_chunk):
            chunk_results = future.result()
            for i, satisfies in chunk_results:
                if not satisfies:
                    all_satisfied = False
    
    return all_satisfied

def update_alpha_pair(args: Tuple[int, int]) -> Tuple[int, int, float, float]:
    """Actualiza un par de multiplicadores alpha"""
    i, j = args
    global global_data, global_config
    
    X = global_data['X']
    y = global_data['y']
    alpha = global_data['alpha']
    K = global_data['kernel_matrix']
    C = global_config['C']
    tolerance = global_config['tolerance']
    
    eps = 1e-8
    alpha_i_old = alpha[i]
    alpha_j_old = alpha[j]
    
    if y[i] != y[j]:
        L = max(0, alpha_j_old - alpha_i_old)
        H = min(C, C + alpha_j_old - alpha_i_old)
    else:
        L = max(0, alpha_i_old + alpha_j_old - C)
        H = min(C, alpha_i_old + alpha_j_old)
    
    if L == H:
        return i, j, alpha_i_old, alpha_j_old
    
    eta = K[i, i] + K[j, j] - 2 * K[i, j]
    if eta <= 0:
        return i, j, alpha_i_old, alpha_j_old
    
    # Calcular gradientes localmente
    grad_i = -1.0
    grad_j = -1.0
    for k in range(len(alpha)):
        grad_i += alpha[k] * y[k] * K[i, k]
        grad_j += alpha[k] * y[k] * K[j, k]
    
    grad_i *= y[i]
    grad_j *= y[j]
    
    alpha_j_new = alpha_j_old + y[j] * (grad_i - grad_j) / (eta + eps)
    
    if alpha_j_new > H:
        alpha_j_new = H
    elif alpha_j_new < L:
        alpha_j_new = L
    
    alpha_i_new = alpha_i_old + y[i] * y[j] * (alpha_j_old - alpha_j_new)
    
    return i, j, alpha_i_new, alpha_j_new

class ConcurrentSVM:
    def __init__(self, C: float = 1.0, kernel: str = 'linear', max_iter: int = 1000, 
                 tol: float = 1e-3, n_workers: int = None):
        self.C = C
        self.kernel = kernel
        self.max_iter = max_iter
        self.tolerance = tol
        self.n_workers = n_workers if n_workers else mp.cpu_count()
        self.alpha = None
        self.b = 0.0
        self.support_vectors = None
        self.support_vector_labels = None
        self.support_vector_alphas = None
        self.kernel_matrix = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Entrenamiento del SVM con Concurrent.Futures"""
        n_samples, n_features = X.shape
        
        # Configuración global
        config = {
            'C': self.C,
            'tolerance': self.tolerance,
            'kernel': self.kernel
        }
        
        print("Fase 1: Inicialización de variables globales...")
        init_global_variables(X, y, config)
        
        print("Fase 2: Cálculo de matriz kernel en paralelo...")
        self.kernel_matrix = compute_kernel_matrix_parallel(X, self.kernel, self.n_workers)
        global_data['kernel_matrix'] = self.kernel_matrix
        
        print("Fase 3: Optimización paralela...")
        for iteration in range(self.max_iter):
            # Calcular gradientes en paralelo
            gradients = compute_all_gradients_parallel(self.n_workers)
            
            # Verificar convergencia en paralelo
            if check_kkt_conditions_parallel(self.n_workers):
                print(f"Convergencia alcanzada en iteración {iteration}")
                break
            
            # Seleccionar working set
            i, j = self._select_working_set(global_data['alpha'], gradients, global_data['y'])
            
            # Actualizar par alpha
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(update_alpha_pair, (i, j))
                i, j, alpha_i_new, alpha_j_new = future.result()
            
            global_data['alpha'][i] = alpha_i_new
            global_data['alpha'][j] = alpha_j_new
            
            if iteration % 100 == 0:
                violations = self._count_kkt_violations(global_data['alpha'], gradients, global_data['y'])
                print(f"Iteración {iteration}, Violaciones KKT: {violations}")
        
        # Fase 4: Extraer resultados
        print("Fase 4: Extracción de resultados...")
        self.alpha = global_data['alpha'].copy()
        self._extract_support_vectors(X, y)
        self._compute_bias(X, y)
    
    def _select_working_set(self, alpha: np.ndarray, gradients: np.ndarray, y: np.ndarray) -> Tuple[int, int]:
        """Selecciona working set para optimización"""
        n_samples = alpha.shape[0]
        
        # Buscar primera violación
        i = -1
        for idx in range(n_samples):
            alpha_i = alpha[idx]
            grad_i = gradients[idx]
            y_i = y[idx]
            
            if alpha_i < self.tolerance:
                if y_i * grad_i < -self.tolerance:
                    i = idx
                    break
            elif alpha_i > self.C - self.tolerance:
                if y_i * grad_i > self.tolerance:
                    i = idx
                    break
            else:
                if abs(y_i * grad_i) > self.tolerance:
                    i = idx
                    break
        
        if i == -1:
            i = np.random.randint(0, n_samples)
        
        # Buscar segunda variable
        j = -1
        max_grad_diff = -1
        for idx in range(n_samples):
            if idx != i:
                grad_diff = abs(gradients[i] - gradients[idx])
                if grad_diff > max_grad_diff:
                    max_grad_diff = grad_diff
                    j = idx
        
        if j == -1:
            j = (i + 1) % n_samples
        
        return i, j
    
    def _count_kkt_violations(self, alpha: np.ndarray, gradients: np.ndarray, y: np.ndarray) -> int:
        """Cuenta violaciones KKT"""
        violations = 0
        n_samples = alpha.shape[0]
        
        for i in range(n_samples):
            alpha_i = alpha[i]
            grad_i = gradients[i]
            y_i = y[i]
            
            if alpha_i < self.tolerance:
                if y_i * grad_i < -self.tolerance:
                    violations += 1
            elif alpha_i > self.C - self.tolerance:
                if y_i * grad_i > self.tolerance:
                    violations += 1
            else:
                if abs(y_i * grad_i) > self.tolerance:
                    violations += 1
        
        return violations
    
    def _extract_support_vectors(self, X: np.ndarray, y: np.ndarray) -> None:
        """Extrae vectores soporte"""
        sv_indices = self.alpha > self.tolerance
        self.support_vectors = X[sv_indices]
        self.support_vector_labels = y[sv_indices]
        self.support_vector_alphas = self.alpha[sv_indices]
    
    def _compute_bias(self, X: np.ndarray, y: np.ndarray) -> None:
        """Calcula el bias b"""
        if len(self.support_vector_alphas) == 0:
            self.b = 0.0
            return
        
        on_margin_indices = (self.alpha > self.tolerance) & (self.alpha < self.C - self.tolerance)
        
        if np.any(on_margin_indices):
            b_values = []
            for i in np.where(on_margin_indices)[0]:
                prediction = np.sum(self.alpha * global_data['y'] * self.kernel_matrix[i, :])
                b_values.append(global_data['y'][i] - prediction)
            self.b = np.mean(b_values)
        else:
            self.b = 0.0
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicción"""
        if self.support_vectors is None:
            raise ValueError("El modelo debe ser entrenado primero")
        
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples)
        
        for i in range(n_samples):
            kernel_vals = np.array([rbf_kernel(X[i], sv) for sv in self.support_vectors])
            prediction = np.sum(self.support_vector_alphas * self.support_vector_labels * kernel_vals) + self.b
            predictions[i] = np.sign(prediction)
        
        return predictions

# Ejemplo de uso
if __name__ == "__main__":
    # Generar datos de ejemplo
    print("Generando datos de ejemplo...")
    X, y = make_classification(n_samples=800, n_features=15, n_redundant=2, 
                              n_informative=8, random_state=42)
    y = np.where(y == 0, -1, 1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Entrenando SVM con Concurrent.Futures...")
    start_time = time.time()
    
    svm = ConcurrentSVM(C=1.0, kernel='linear', max_iter=300, n_workers=4)
    svm.fit(X_train, y_train)
    
    end_time = time.time()
    print(f"Tiempo de entrenamiento: {end_time - start_time:.2f} segundos")
    
    # Predicciones
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Precisión: {accuracy:.4f}")
    print(f"Número de vectores soporte: {len(svm.support_vectors)}")