#Dada la siguiente lista de rendimientos:

rendimientos = [0.02, 0.03, -0.01, 0.015, 0.04]


rendimientos = [0.02, 0.03, -0.01, 0.015, 0.04]
n = len(rendimientos)
promedio = sum(rendimientos) / n

# Diferencias al cuadrado
cuadrados = [(r - promedio) ** 2 for r in rendimientos]
suma_cuadrados = sum(cuadrados)

# Desviación estándar muestral
volatilidad = (suma_cuadrados / (n - 1)) ** 0.5


print(f"Volatilidad (manual): {volatilidad:.4f}")