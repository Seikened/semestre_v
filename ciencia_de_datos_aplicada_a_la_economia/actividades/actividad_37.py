# =========================
# Ejercicio 4 - SVR (kernel RBF)
# Autor: Fernando Leon Franco
# =========================

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1 Descargar datos del BTC
btc = yf.download("BTC-USD", start="2024-01-01", end="2025-01-01")

# 2 Calcular variables
btc["Rendimiento"] = np.log(btc["Close"] / btc["Close"].shift(1))
btc["MA7"] = btc["Close"].rolling(7).mean()
btc["Volatilidad"] = btc["Rendimiento"].rolling(7).std()
btc = btc.dropna()

# 3 Variables predictoras y objetivo
X = btc[["MA7", "Volatilidad"]]
y = btc["Close"]

# 4 Escalado
scaler_X, scaler_y = StandardScaler(), StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1)).ravel()

# 5 Modelo con kernel RBF
modelo = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
modelo.fit(X_scaled, y_scaled)

# 6 Predicción y evaluación
y_pred_scaled = modelo.predict(X_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))

rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

print("RMSE:", round(rmse, 2))
print("R²:", round(r2, 4))

# 7 Gráfico
plt.figure(figsize=(12, 5))
plt.plot(btc.index, y, label="Precio real", color="black")
plt.plot(btc.index, y_pred, label="SVR (RBF)", color="green", linestyle="--")
plt.title("Predicción de precios del Bitcoin con SVR (kernel RBF)")
plt.legend()
plt.show()

# =========================
# Análisis simple
# =========================
print("\n================= ANÁLISIS =================")
print("1️⃣  El kernel RBF permite al modelo capturar relaciones no lineales.")
print("    A diferencia del kernel lineal, puede adaptarse a curvas o patrones complejos del precio.")

print("\n2️⃣  Cambiar los hiperparámetros tiene estos efectos:")
print("   - Aumentar C: el modelo se ajusta más al entrenamiento (riesgo de sobreajuste).")
print("   - Aumentar gamma: el modelo se vuelve más sensible, aprende detalles finos pero puede memorizar ruido.")
print("   - Aumentar epsilon: el modelo ignora pequeños errores y se suaviza.")

print("\n3️⃣  Si el R² es muy cercano a 1, verifica sobreajuste comparando entrenamiento vs prueba,")
print("    o usando validación cruzada para ver si el rendimiento se mantiene estable en nuevos datos.")

print("============================================")