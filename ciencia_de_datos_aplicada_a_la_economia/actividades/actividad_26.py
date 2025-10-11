#Usa numpy para calcular la desviación estándar de estos rendimientos:
import numpy as np


rendimientos = [0.05, 0.01, -0.02, 0.03, 0.025]
volatilidad = np.std(rendimientos, ddof=1)

print(f"Volatilidad con numpy: {volatilidad:.4f}")