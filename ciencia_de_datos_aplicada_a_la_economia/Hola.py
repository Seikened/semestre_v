import numpy as np


def rendimiento(precio_inicial, precio_final):
    return (precio_final - precio_inicial) / precio_inicial


def rendimiento_logaritmico(precio_inicial, precio_final):
    return np.log(precio_final) - np.log(precio_inicial)


precio_inicial = 250
precio_final = 280
red = rendimiento(precio_inicial, precio_final)
print(f"   Rendimiento: {red}%")


# Precios
precios = [100, 105, 102, 107, 110]

rendimiento_list = []

for i in range(len(precios) - 1):
    prec_inc = precios[i]
    prec_fin = precios[i + 1]
    rendimiento_list.append(rendimiento(prec_inc, prec_fin))

print(f"Rendimientos ejemplo: {rendimiento_list}")


# Rendimientos logarítmicos
precios_2 = [120, 125, 130, 128, 132]

rendimiento_log_list = []

for i in range(len(precios_2) - 1):
    prec_inc = precios_2[i]
    prec_fin = precios_2[i + 1]
    rendimiento_log_list.append(rendimiento_logaritmico(prec_inc, prec_fin))

print(f"Rendimientos logarítmicos: {rendimiento_log_list}")
