"""
Objetivo:
Aplicar el modelo de regresión lineal a datos históricos de Bitcoin.

Datos:
Descarga precios del Bitcoin desde Yahoo Finance (BTC-USD).

Análisis:

¿Qué valor tiene el coeficiente b1?

¿Qué indica si el valor de b1 es cercano a 1?
"""


import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

btc = yf.download("BTC-USD", start="2024-01-01", end="2025-01-01")
btc["Close_prev"] = btc["Close"].shift(1)
btc = btc.dropna()

X = btc[["Close_prev"]]
y = btc["Close"]

modelo = LinearRegression()
modelo.fit(X, y)

ultimo = btc.iloc[-1]
X_next = pd.DataFrame({"Close_prev": [float(ultimo["Close"])]})
prediccion = modelo.predict(X_next)
print(f"Predicción para el siguiente día: {prediccion[0]}")

print("Análisis de regresión lineal:")
print(f"¿Qué valor tiene el coeficiente b1? {modelo.coef_[0]}")
print("¿Qué indica si el valor de b1 es cercano a 1?")
if abs(modelo.coef_[0] - 1) < 0.1:
    print("Indica que el precio de Bitcoin tiende a mantenerse estable día a día.")
else:
    print("Indica que el precio de Bitcoin tiene una mayor variabilidad día a día.")