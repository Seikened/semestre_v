"""
Recomendación financiera usando diccionarios
📌 Instrucciones:

Crea un diccionario que contenga tres activos financieros como claves (por ejemplo: "AAPL", "TSLA", "NFLX") y sus rendimientos promedio como valores.

Usa un ciclo for para recorrer el diccionario.

Según el rendimiento de cada activo, imprime una recomendación:

Si es mayor a 0.10 → "Alto rendimiento – Recomendado"

Si está entre 0 y 0.10 → "Rendimiento moderado"

Si es menor o igual a 0 → "Rendimiento negativo – Riesgoso"
"""

# Diccionario de activos financieros y sus rendimientos promedio
activos = {
    "AAPL": 0.12,
    "TSLA": 0.08,
    "NFLX": -0.05
}


def recomendar_inversion(activos):
    for activo, rendimiento in activos.items():
        if rendimiento > 0.10:
            recomendacion = "Alto rendimiento – Recomendado"
        elif 0 < rendimiento <= 0.10:
            recomendacion = "Rendimiento moderado"
        else:
            recomendacion = "Rendimiento negativo – Riesgoso"
        print(f"Activo: {activo}, Rendimiento: {rendimiento:.2%}, Recomendación: {recomendacion}")
        
    
recomendar_inversion(activos)