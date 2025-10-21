import numpy as np
from numba import jit, prange
import time
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')

# Variables globales para evitar problemas de serialización
global_X_train = None
global_y_train = None
global_alpha = None
global_kernel_matrix = None
global_b = 0.0
global_C = 1.0
global_tolerance = 1e-3

@jit(nopython=True)
def linear_kernel(x1: np.ndarray, x2: np.ndarray) -> float:
    """Kernel lineal - optimizado con Numba"""
    return np.dot(x1, x2)

@jit(nopython=True)
def rbf_kernel(x1: np.ndarray, x2: np.ndarray, gamma: float = 0.1) -> float:
    """Kernel RBF - optimizado con Numba"""
    return np.exp(-gamma * np.sum((x1 - x2) ** 2))

@jit(nopython=True, parallel=True)
def compute_kernel_matrix_parallel(X: np.ndarray, kernel_type: str = 'linear') -> np.ndarray:
    """Calcula matriz kernel en paralelo"""
    n_samples = X.shape[0]
    K = np.zeros((n_samples, n_samples))
    
    for i in prange(n_samples):
        for j in prange(n_samples):
            if kernel_type == 'linear':
                K[i, j] = linear_kernel(X[i], X[j])
            else:  # rbf
                K[i, j] = rbf_kernel(X[i], X[j])
    return K

@jit(nopython=True)
def compute_gradient(i: int, alpha: np.ndarray, y: np.ndarray, K: np.ndarray) -> float:
    """Calcula gradiente para una muestra específica"""
    n_samples = alpha.shape[0]
    grad = -1.0
    for j in range(n_samples):
        grad += alpha[j] * y[j] * K[i, j]
    return grad * y[i]

@jit(nopython=True, parallel=True)
def compute_all_gradients(alpha: np.ndarray, y: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Calcula todos los gradientes en paralelo"""
    n_samples = alpha.shape[0]
    gradients = np.zeros(n_samples)
    
    for i in prange(n_samples):
        gradients[i] = compute_gradient(i, alpha, y, K)
    return gradients

@jit(nopython=True)
def satisfies_kkt(alpha_i: float, grad_i: float, y_i: float, C: float, tolerance: float) -> bool:
    """Verifica condiciones KKT"""
    if alpha_i < tolerance:
        return y_i * grad_i >= -tolerance
    elif alpha_i > C - tolerance:
        return y_i * grad_i <= tolerance
    else:
        return abs(y_i * grad_i) <= tolerance

@jit(nopython=True, parallel=True)
def check_kkt_conditions_parallel(alpha: np.ndarray, gradients: np.ndarray, 
                                y: np.ndarray, C: float, tolerance: float) -> bool:
    """Verifica condiciones KKT en paralelo"""
    n_samples = alpha.shape[0]
    violations = 0
    
    for i in prange(n_samples):
        if not satisfies_kkt(alpha[i], gradients[i], y[i], C, tolerance):
            violations += 1
    
    return violations == 0

@jit(nopython=True)
def update_alpha(i: int, j: int, alpha: np.ndarray, y: np.ndarray, 
                K: np.ndarray, C: float) -> Tuple[float, float]:
    """Actualiza par de multiplicadores alpha"""
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
        return alpha_i_old, alpha_j_old
    
    eta = K[i, i] + K[j, j] - 2 * K[i, j]
    if eta <= 0:
        return alpha_i_old, alpha_j_old
    
    grad_i = compute_gradient(i, alpha, y, K)
    grad_j = compute_gradient(j, alpha, y, K)
    
    alpha_j_new = alpha_j_old + y[j] * (grad_i - grad_j) / (eta + eps)
    
    if alpha_j_new > H:
        alpha_j_new = H
    elif alpha_j_new < L:
        alpha_j_new = L
    
    alpha_i_new = alpha_i_old + y[i] * y[j] * (alpha_j_old - alpha_j_new)
    
    return alpha_i_new, alpha_j_new

class NumbaParallelSVM:
    def __init__(self, C: float = 1.0, kernel: str = 'linear', max_iter: int = 1000, tol: float = 1e-3):
        self.C = C
        self.kernel = kernel
        self.max_iter = max_iter
        self.tolerance = tol
        self.alpha = None
        self.b = 0.0
        self.support_vectors = None
        self.support_vector_labels = None
        self.support_vector_alphas = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Entrenamiento del SVM con paralelización Numba"""
        global global_X_train, global_y_train, global_alpha, global_kernel_matrix, global_b, global_C
        
        # Configurar variables globales
        global_X_train = X.copy()
        global_y_train = y.copy()
        global_C = self.C
        
        n_samples, n_features = X.shape
        
        print("Fase 1: Preprocesamiento y cálculo de kernel...")
        # Calcular matriz kernel en paralelo
        global_kernel_matrix = compute_kernel_matrix_parallel(X, self.kernel)
        
        # Inicializar multiplicadores
        global_alpha = np.zeros(n_samples)
        global_b = 0.0
        
        print("Fase 2: Optimización paralela...")
        for iteration in range(self.max_iter):
            # Calcular gradientes en paralelo
            gradients = compute_all_gradients(global_alpha, global_y_train, global_kernel_matrix)
            
            # Verificar convergencia en paralelo
            if check_kkt_conditions_parallel(global_alpha, gradients, global_y_train, self.C, self.tolerance):
                print(f"Convergencia alcanzada en iteración {iteration}")
                break
            
            # Seleccionar working set (estrategia simple)
            i, j = self._select_working_set(global_alpha, gradients, global_y_train)
            
            # Actualizar alpha_i y alpha_j
            alpha_i_new, alpha_j_new = update_alpha(i, j, global_alpha, global_y_train, 
                                                  global_kernel_matrix, self.C)
            
            global_alpha[i] = alpha_i_new
            global_alpha[j] = alpha_j_new
            
            if iteration % 100 == 0:
                violations = np.sum(~self._check_kkt_conditions(global_alpha, gradients, global_y_train))
                print(f"Iteración {iteration}, Violaciones KKT: {violations}")
        
        # Fase 3: Identificar vectores soporte
        print("Fase 3: Identificación de vectores soporte...")
        
        # CORRECCIÓN: Asignar alpha ANTES de extraer vectores soporte
        self.alpha = global_alpha.copy()
        self._extract_support_vectors(X, y)
        
        # Fase 4: Calcular b
        print("Fase 4: Cálculo de parámetros finales...")
        self._compute_bias(X, y)
        
        self.b = global_b
    
    def _select_working_set(self, alpha: np.ndarray, gradients: np.ndarray, y: np.ndarray) -> Tuple[int, int]:
        """Selecciona el working set para optimización"""
        n_samples = alpha.shape[0]
        
        # Primera heurística: seleccionar i que viole KKT
        i = -1
        for idx in range(n_samples):
            if not satisfies_kkt(alpha[idx], gradients[idx], y[idx], self.C, self.tolerance):
                i = idx
                break
        
        if i == -1:
            i = np.random.randint(0, n_samples)
        
        # Segunda heurística: seleccionar j que maximice |grad_i - grad_j|
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
    
    def _check_kkt_conditions(self, alpha: np.ndarray, gradients: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Verifica condiciones KKT para todas las muestras"""
        n_samples = alpha.shape[0]
        kkt_satisfied = np.zeros(n_samples, dtype=np.bool_)
        
        for i in range(n_samples):
            kkt_satisfied[i] = satisfies_kkt(alpha[i], gradients[i], y[i], self.C, self.tolerance)
        
        return kkt_satisfied
    
    def _extract_support_vectors(self, X: np.ndarray, y: np.ndarray) -> None:
        """Extrae vectores soporte"""
        # CORRECCIÓN: Usar self.alpha en lugar de self.alpha (que ahora está asignado)
        if self.alpha is None:
            raise ValueError("alpha no ha sido inicializado")
            
        sv_indices = self.alpha > self.tolerance
        self.support_vectors = X[sv_indices]
        self.support_vector_labels = y[sv_indices]
        self.support_vector_alphas = self.alpha[sv_indices]
    
    def _compute_bias(self, X: np.ndarray, y: np.ndarray) -> None:
        """Calcula el bias b"""
        if len(self.support_vector_alphas) == 0:
            self.b = 0.0
            return
        
        # Usar vectores soporte en el margen para calcular b
        on_margin_indices = (self.alpha > self.tolerance) & (self.alpha < self.C - self.tolerance)
        
        if np.any(on_margin_indices):
            b_values = []
            for i in np.where(on_margin_indices)[0]:
                prediction = np.sum(self.alpha * global_y_train * global_kernel_matrix[i, :])
                b_values.append(global_y_train[i] - prediction)
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
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    print("Generando datos de ejemplo...")
    X, y = make_classification(n_samples=1000, n_features=20, n_redundant=2, 
                              n_informative=10, random_state=42)
    y = np.where(y == 0, -1, 1)  # Convertir a -1, 1
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Entrenando SVM con Numba...")
    start_time = time.time()
    
    svm = NumbaParallelSVM(C=1.0, kernel='linear', max_iter=500)
    svm.fit(X_train, y_train)
    
    end_time = time.time()
    print(f"Tiempo de entrenamiento: {end_time - start_time:.2f} segundos")
    
    # Predicciones
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Precisión: {accuracy:.4f}")
    print(f"Número de vectores soporte: {len(svm.support_vectors)}")