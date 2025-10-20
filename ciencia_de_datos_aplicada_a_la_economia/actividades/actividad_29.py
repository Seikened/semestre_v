"""
Actividad 29: Análisis de Series de Tiempo
Objetivo:
Comprender cómo la regresión lineal ajusta una línea a un conjunto de datos simples.

Datos:
Crea una tabla con la relación entre horas de estudio y calificación.

Horas | Calificación
1     | 60
2     | 65
3     | 70
4     | 80
5     | 85


 
"""

import numpy as np

from sklearn.linear_model import LinearRegression

X = np.array([1,2,3,4,5]).reshape(-1,1)

y = np.array([60,65,70,80,85])

modelo = LinearRegression()

modelo.fit(X, y)

print("Intercepto:", modelo.intercept_)

print("Coeficiente:", modelo.coef_)

print("Ecuación: Calificación = {:.2f} + {:.2f}×Horas".format(modelo.intercept_, modelo.coef_[0]))

prediccion = modelo.predict([[6]])

print("Predicción para 6 horas de estudio:", prediccion[0])