"""
Objetivo:
Transformar los precios en rendimientos logarítmicos y predecir el rendimiento del día siguiente.

Actividad en Python:
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import yfinance as yf

btc = yf.download("BTC-USD", start="2024-01-01", end="2025-01-01")
btc["Close_prev"] = btc["Close"].shift(1)
btc = btc.dropna()




# ================== SOLUCION ACTIVIDAD 31 ==================

df = btc[["Close", "Close_prev", "Volume"]].dropna()

X = df[["Close_prev", "Volume"]].to_numpy(dtype=float)  # (n, 2) float
y = df["Close"].to_numpy(dtype=float).ravel()            # (n,)  1D

modelo = LinearRegression()
modelo.fit(X, y)

coef = np.ravel(modelo.coef_)            # garantiza (2,)
intercepto = float(modelo.intercept_)    # escalar

coef_close_prev, coef_volume = coef
print(f"Ecuación: Precio = {intercepto:.2f} + {coef_close_prev:.6f} × Close_prev + {coef_volume:.10f} × Volumen")

print("Análisis:")
print("¿Cómo afecta el volumen al precio?")
print("El volumen tiene un efecto positivo en el precio." if coef_volume > 0 else
      "El volumen tiene un efecto negativo en el precio." if coef_volume < 0 else
      "El volumen no muestra efecto lineal en este ajuste.")

print("¿Cuál de las variables tiene mayor peso en el modelo?")
print("La variable 'Close_prev' tiene mayor peso en el modelo."
      if abs(coef_close_prev) >= abs(coef_volume)
      else "La variable 'Volume' tiene mayor peso en el modelo.")