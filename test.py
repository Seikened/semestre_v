# -*- coding: utf-8 -*-
# Sentimiento en español (robertuito) — versión minimalista, estable y lista para M2 (MPS)

from transformers import pipeline
import torch

# ————————————————————————————
# Opcional: kernels más estables en M2
# ————————————————————————————
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# ————————————————————————————
# Textos (se mantienen idénticos)
# ————————————————————————————
noticia_positiva = "En un anuncio que ha generado entusiasmo a nivel internacional, la Agencia Espacial Global confirmó hoy que la misión Artemis-X ha logrado aterrizar con éxito en la superficie de Marte, convirtiéndose en la primera operación tripulada en la historia en hacerlo. Durante la transmisión en vivo, millones de personas alrededor del mundo celebraron cuando los astronautas descendieron de la nave y plantaron la bandera de la Tierra, símbolo de cooperación global. Según los expertos, este hito marcará el comienzo de una nueva era de exploración espacial, con planes para establecer un hábitat permanente en los próximos años. Las agencias participantes aseguraron que los datos recopilados permitirán avances en tecnología, medicina y energía, beneficiando directamente a la humanidad. La reacción en redes sociales fue inmediata, con mensajes de apoyo, felicitaciones y expresiones de esperanza por un futuro interplanetario. Los mercados bursátiles incluso reaccionaron positivamente, con un aumento notable en las acciones de compañías aeroespaciales. Los líderes mundiales han elogiado el esfuerzo conjunto y han prometido apoyo continuo para futuras misiones. Los científicos de la misión informaron que los primeros análisis de suelo confirman la presencia de agua en estado subterráneo, lo que podría facilitar la producción de combustible y el sustento de vida en el planeta rojo. En una conferencia posterior, los astronautas se mostraron emocionados y destacaron la importancia de seguir inspirando a nuevas generaciones a mirar hacia las estrellas."

noticia_negativa = "Una serie de desastres naturales ha sumido al país de Northridge en una crisis humanitaria sin precedentes. En menos de una semana, un terremoto de magnitud 7.8 destruyó gran parte de la infraestructura de la capital, dejando a cientos de miles de personas sin vivienda, seguido por fuertes lluvias que provocaron inundaciones generalizadas en las regiones costeras. Los hospitales se encuentran saturados, y la escasez de alimentos y medicinas amenaza con empeorar la situación. Las autoridades han declarado el estado de emergencia y han solicitado ayuda internacional, pero las operaciones de rescate se ven entorpecidas por el colapso de carreteras y puentes. Testigos reportan escenas de caos en los refugios improvisados, donde las familias luchan por conseguir agua potable. Organizaciones de derechos humanos han denunciado que la respuesta gubernamental ha sido lenta y descoordinada, lo que ha aumentado el descontento social y provocado manifestaciones en las principales ciudades. Expertos advierten que, si no se implementan medidas urgentes, podría desatarse una crisis sanitaria masiva debido a la proliferación de enfermedades infecciosas. Las imágenes transmitidas por la televisión muestran edificios derrumbados, vehículos arrastrados por la corriente y calles convertidas en ríos de lodo. La población vive en un estado de angustia constante, con réplicas sísmicas que mantienen el miedo latente en la población."

noticia_neutral = "El Instituto Nacional de Estadística publicó hoy su informe anual sobre indicadores económicos y sociales del país. Según el documento, el Producto Interno Bruto creció un 2.1% en comparación con el año anterior, mientras que la tasa de desempleo se mantuvo en 4.5%. La inflación promedio fue de 3.2%, ligeramente por debajo de la meta del banco central. El informe también detalló el comportamiento de sectores clave como manufactura, agricultura y servicios, mostrando variaciones moderadas pero estables. En el apartado de comercio internacional, las exportaciones crecieron un 1.8% y las importaciones un 2.3%, manteniendo un déficit comercial manejable. Se registró un aumento en la inversión extranjera directa del 5%, principalmente en proyectos de infraestructura y energía renovable. Los analistas señalan que estos datos reflejan una economía en fase de recuperación gradual, aunque advierten que factores externos, como la volatilidad de los mercados internacionales, podrían afectar el desempeño en el próximo semestre. El informe incluye estadísticas sobre educación, salud y movilidad urbana, sin cambios drásticos en relación con años anteriores. Las autoridades señalaron que los datos servirán como base para el diseño de políticas públicas y la planeación presupuestaria del siguiente ejercicio fiscal."

# ————————————————————————————
# Configuración del modelo
# ————————————————————————————
MODEL_NAME = "pysentimiento/robertuito-sentiment-analysis"

def init_pipeline():
    """
    Inicializa un pipeline de sentimiento en español:
      - Truncación a 128 tokens (alineado al entrenamiento típico).
      - return_all_scores=True para obtener NEG/NEU/POS con sus probabilidades.
      - device="mps" para Mac Apple Silicon; si no hay MPS, cae a CPU.
    """
    device = "mps" if torch.backends.mps.is_available() else -1
    pipe = pipeline(
        task="sentiment-analysis",
        model=MODEL_NAME,
        device=device,
        truncation=True,
        max_length=128,
        return_all_scores=True
    )
    return pipe

# ————————————————————————————
# Utilidades compactas
# ————————————————————————————
def contar_tokens(tokenizer, texto: str) -> int:
    # Cuenta tokens sin truncar, sin agregar especiales (medida honesta de longitud)
    return len(tokenizer.tokenize(texto))

def formatear_resultado(nombre: str, etiqueta: str, puntajes: dict, tokens: int):
    neg = puntajes.get("NEG", 0.0)
    neu = puntajes.get("NEU", 0.0)
    pos = puntajes.get("POS", 0.0)
    print(f"{nombre} [corto] tokens={tokens} -> NEG={neg:.2%} NEU={neu:.2%} POS={pos:.2%} => {etiqueta}")

def clasificar(pipe, texto: str):
    """
    Ejecuta una sola pasada con truncación. Devuelve:
      etiqueta (str), dict con {NEG, NEU, POS}, tokens reales (int).
    """
    salida = pipe(texto)  # con return_all_scores=True → lista con una entrada: [{label, score}, ...]
    # salida es p.ej. [[{'label': 'NEG', 'score': 0.03}, {'label': 'NEU', 'score': 0.27}, {'label': 'POS', 'score': 0.70}]]
    pares = salida[0]
    puntajes = {d["label"]: float(d["score"]) for d in pares}
    # etiqueta ganadora
    etiqueta = max(puntajes.items(), key=lambda kv: kv[1])[0]
    tokens = contar_tokens(pipe.tokenizer, texto)
    return etiqueta, puntajes, tokens

# ————————————————————————————
# Main
# ————————————————————————————
if __name__ == "__main__":
    pipe = init_pipeline()

    et_pos, sc_pos, tk_pos = clasificar(pipe, noticia_positiva)
    et_neg, sc_neg, tk_neg = clasificar(pipe, noticia_negativa)
    et_neu, sc_neu, tk_neu = clasificar(pipe, noticia_neutral)

    formatear_resultado("Positiva", et_pos, sc_pos, tk_pos)
    formatear_resultado("Negativa", et_neg, sc_neg, tk_neg)
    formatear_resultado("Neutral",  et_neu, sc_neu, tk_neu)