# =========================
# Ejercicio 1 - Bitcoin Up/Down Clasifier
# Autor: Fernando Leon Franco
# Descripción: Regresión logística para predecir si BTC sube (1) o baja (0) al día siguiente,
# usando rendimiento previo y volumen como variables predictoras.
# =========================

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1 Descargar datos históricos
btc = yf.download("BTC-USD", start="2023-01-01", end="2025-01-01")

# 2 Calcular rendimientos y variable objetivo
btc["Rendimiento"] = np.log(btc["Close"] / btc["Close"].shift(1))   # rendimiento diario log
btc["Rend_prev"] = btc["Rendimiento"].shift(1)                      # rendimiento del día anterior
btc["Sube"] = np.where(btc["Rendimiento"] > 0, 1, 0)                # 1 si sube hoy, 0 si baja
btc = btc.dropna()                                                  # quitar filas con NaN

# 3 Definir variables (X: predictoras, y: objetivo)
X = btc[["Rend_prev", "Volume"]]
y = btc["Sube"]

# 4 División de datos (train pasado, test futuro)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    shuffle=False
)

# 5 Modelo de regresión logística
modelo = LogisticRegression(max_iter=500, class_weight='balanced')
modelo.fit(X_train, y_train)

# 6 Predicciones y evaluación
y_pred = modelo.predict(X_test)

print("Precisión:", round(accuracy_score(y_test, y_pred), 3))
print("\nMatriz de confusión:\n", confusion_matrix(y_test, y_pred))
print("\nReporte de clasificación:\n", classification_report(y_test, y_pred))


print("\n================= ANÁLISIS =================")

print("1️⃣  Precisión general:", round(accuracy_score(y_test, y_pred), 3))
print("👉  El modelo acierta en ~54% de los días.")

print("\n2️⃣  Matriz de confusión:")
print(confusion_matrix(y_test, y_pred))
print("👉  El modelo solo predijo la clase '1' (sube). No detectó ninguna bajada.")

print("\n3️⃣  Sesgo del modelo:")
print("⚠️  Está sesgado: siempre dice que sube. Recall=1.00 para subidas, 0.00 para bajadas.")

print("\n4️⃣  Posible mejora:")
print("💡  Agregar medias móviles (MA5, MA10) y volatilidad reciente como nuevas variables.")
print("   Esto ayudaría a capturar tendencias y giros en el mercado.")

print("\n============================================")