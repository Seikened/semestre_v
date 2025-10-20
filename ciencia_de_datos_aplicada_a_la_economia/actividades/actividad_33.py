"""
Objetivo:
Desarrollar un pequeño proyecto aplicando regresión lineal para predecir precios o rendimientos de un activo financiero elegido por el alumno.

Instrucciones:

Elige un activo: puede ser Bitcoin, Apple, Tesla o cualquier otro de Yahoo Finance.

Descarga sus datos históricos de los últimos 12 meses.

Calcula:

Rendimiento logarítmico diario

Precio del día anterior

Entrena un modelo de regresión lineal (LinearRegression).

Evalúa su rendimiento con R² y MAE.

Crea una gráfica comparativa entre precio real y precio predicho.

Redacta una interpretación de resultados en un párrafo.


"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import yfinance as yf



ticker = "AAPL"
data = yf.download(ticker, period="12mo")
data.dropna(inplace=True)

# Rendimiento logarítmico diario
rendimientos = data["Close"].pct_change().apply(lambda x: np.log(1 + x))

# Precio del día anterior
data["Rendimiento"] = rendimientos
data["Precio_prev"] = data["Close"].shift(1)
data.dropna(inplace=True)

X = data[["Precio_prev"]]
y = data["Rendimiento"]

model = LinearRegression()
model.fit(X, y)

predicciones = model.predict(X)

r2 = r2_score(y, predicciones)
mae = mean_absolute_error(y, predicciones)

print("Análisis de resultados:")
print(f"R²: {r2:.4f}")
print(f"MAE: {mae:.4f}")


plt.figure(figsize=(12, 6))
plt.plot(data.index, y, label="Rendimiento Real", color="blue")
plt.plot(data.index, predicciones, label="Rendimiento Predicho", color="red", linestyle="--")
plt.title(f"Rendimiento Real vs Predicho para {ticker}")
plt.xlabel("Fecha")
plt.ylabel("Rendimiento Logarítmico")
plt.legend()
plt.show()


print(f"{"="*20} Interpretación {"="*20}")
print(f"El modelo tiene R²={r2:.4f} ({r2*100:.2f}% explicado). MAE={mae:.4f}, error medio. La gráfica muestra cómo se ajustan predicciones y valores reales.")