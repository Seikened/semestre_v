def rendimiento(precio_inicial,precio_final):
    return (precio_final - precio_inicial) / precio_inicial



precio_inicial = 250
precio_final = 280
red = rendimiento(precio_inicial,precio_final)
print(f"   Rendimiento: {red}%")