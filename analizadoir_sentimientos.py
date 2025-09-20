# -*- coding: utf-8 -*-
# Sentimiento en español (robertuito) — impresión tabular con Polars y métricas de contexto

from transformers import pipeline
import torch
import polars as pl

# ——————————————————————————————————
# Ajustes de ejecución
# ——————————————————————————————————
# Sube MAX_LEN a 256 o 512 si quieres procesar más contexto por texto.
MAX_LEN = 128

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# ——————————————————————————————————
# Textos (idénticos a los tuyos)
# ——————————————————————————————————
noticia_positiva = "En un anuncio que ha generado entusiasmo a nivel internacional, la Agencia Espacial Global confirmó hoy que la misión Artemis-X ha logrado aterrizar con éxito en la superficie de Marte, convirtiéndose en la primera operación tripulada en la historia en hacerlo. Durante la transmisión en vivo, millones de personas alrededor del mundo celebraron cuando los astronautas descendieron de la nave y plantaron la bandera de la Tierra, símbolo de cooperación global. Según los expertos, este hito marcará el comienzo de una nueva era de exploración espacial, con planes para establecer un hábitat permanente en los próximos años. Las agencias participantes aseguraron que los datos recopilados permitirán avances en tecnología, medicina y energía, beneficiando directamente a la humanidad. La reacción en redes sociales fue inmediata, con mensajes de apoyo, felicitaciones y expresiones de esperanza por un futuro interplanetario. Los mercados bursátiles incluso reaccionaron positivamente, con un aumento notable en las acciones de compañías aeroespaciales. Los líderes mundiales han elogiado el esfuerzo conjunto y han prometido apoyo continuo para futuras misiones. Los científicos de la misión informaron que los primeros análisis de suelo confirman la presencia de agua en estado subterráneo, lo que podría facilitar la producción de combustible y el sustento de vida en el planeta rojo. En una conferencia posterior, los astronautas se mostraron emocionados y destacaron la importancia de seguir inspirando a nuevas generaciones a mirar hacia las estrellas."

noticia_negativa = "Una serie de desastres naturales ha sumido al país de Northridge en una crisis humanitaria sin precedentes. En menos de una semana, un terremoto de magnitud 7.8 destruyó gran parte de la infraestructura de la capital, dejando a cientos de miles de personas sin vivienda, seguido por fuertes lluvias que provocaron inundaciones generalizadas en las regiones costeras. Los hospitales se encuentran saturados, y la escasez de alimentos y medicinas amenaza con empeorar la situación. Las autoridades han declarado el estado de emergencia y han solicitado ayuda internacional, pero las operaciones de rescate se ven entorpecidas por el colapso de carreteras y puentes. Testigos reportan escenas de caos en los refugios improvisados, donde las familias luchan por conseguir agua potable. Organizaciones de derechos humanos han denunciado que la respuesta gubernamental ha sido lenta y descoordinada, lo que ha aumentado el descontento social y provocado manifestaciones en las principales ciudades. Expertos advierten que, si no se implementan medidas urgentes, podría desatarse una crisis sanitaria masiva debido a la proliferación de enfermedades infecciosas. Las imágenes transmitidas por la televisión muestran edificios derrumbados, vehículos arrastrados por la corriente y calles convertidas en ríos de lodo. La población vive en un estado de angustia constante, con réplicas sísmicas que mantienen el miedo latente en la población."

noticia_neutral = "El Instituto Nacional de Estadística publicó hoy su informe anual sobre indicadores económicos y sociales del país. Según el documento, el Producto Interno Bruto creció un 2.1% en comparación con el año anterior, mientras que la tasa de desempleo se mantuvo en 4.5%. La inflación promedio fue de 3.2%, ligeramente por debajo de la meta del banco central. El informe también detalló el comportamiento de sectores clave como manufactura, agricultura y servicios, mostrando variaciones moderadas pero estables. En el apartado de comercio internacional, las exportaciones crecieron un 1.8% y las importaciones un 2.3%, manteniendo un déficit comercial manejable. Se registró un aumento en la inversión extranjera directa del 5%, principalmente en proyectos de infraestructura y energía renovable. Los analistas señalan que estos datos reflejan una economía en fase de recuperación gradual, aunque advierten que factores externos, como la volatilidad de los mercados internacionales, podrían afectar el desempeño en el próximo semestre. El informe incluye estadísticas sobre educación, salud y movilidad urbana, sin cambios drásticos en relación con años anteriores. Las autoridades señalaron que los datos servirán como base para el diseño de políticas públicas y la planeación presupuestaria del siguiente ejercicio fiscal."

# ——————————————————————————————————
# Pipeline: sin warnings, con top_k=None
# ——————————————————————————————————
MODEL_NAME = "pysentimiento/robertuito-sentiment-analysis"

def init_pipeline(max_len: int):
    device = "mps" if torch.backends.mps.is_available() else -1
    return pipeline(
        task="sentiment-analysis",
        model=MODEL_NAME,
        device=device,
        truncation=True,        # aplica truncación por defecto
        max_length=max_len,     # longitud efectiva
        top_k=None              # equivalente al viejo return_all_scores=True
    )

# ——————————————————————————————————
# Utilidades
# ——————————————————————————————————
def tokens_totales(tokenizer, texto: str) -> int:
    # Cuenta tokens "crudos" (sin truncar)
    return len(tokenizer.tokenize(texto))

def tokens_procesados(tokenizer, texto: str, max_len: int) -> int:
    # Cuenta tokens que realmente entran al modelo tras truncación
    # Nota: para modelos tipo RoBERTa, los especiales ocupan posiciones;
    # aquí tomamos min(total, max_len) como aproximación práctica de la carga.
    return min(tokens_totales(tokenizer, texto), max_len)

def clasificar_uno(pipe, texto: str, nombre: str):
    # Fuerza truncation y max_length en la llamada para evitar warnings
    salida = pipe(texto, truncation=True, max_length=MAX_LEN)[0]
    # salida es lista de dicts con {'label','score'} gracias a top_k=None
    puntajes = {d["label"]: float(d["score"]) for d in salida}
    etiqueta = max(puntajes, key=puntajes.get)
    tt = tokens_totales(pipe.tokenizer, texto)
    tp = tokens_procesados(pipe.tokenizer, texto, MAX_LEN)
    usado = tp / max(tt, 1)
    return {
        "Nombre": nombre,
        "Etiq": etiqueta,
        "NEG": puntajes.get("NEG", 0.0),
        "NEU": puntajes.get("NEU", 0.0),
        "POS": puntajes.get("POS", 0.0),
        "Tokens_totales": tt,
        "Tokens_procesados": tp,
        "Texto_usado_%": usado
    }

def formatear_df(df: pl.DataFrame) -> pl.DataFrame:
    # Redondeo amigable para porcentajes y casting de enteros
    return (
        df
        .with_columns([
            pl.col("NEG").map_elements(lambda x: round(x*100, 2)).alias("NEG_%"),
            pl.col("NEU").map_elements(lambda x: round(x*100, 2)).alias("NEU_%"),
            pl.col("POS").map_elements(lambda x: round(x*100, 2)).alias("POS_%"),
            pl.col("Texto_usado_%").map_elements(lambda x: round(x*100, 1)).alias("Texto_usado_%"),
        ])
        .drop(["NEG", "NEU", "POS"])
        .select(["Nombre", "Etiq", "NEG_%", "NEU_%", "POS_%", "Tokens_totales", "Tokens_procesados", "Texto_usado_%"])
    )

# ——————————————————————————————————
# Main
# ——————————————————————————————————
if __name__ == "__main__":
    pipe = init_pipeline(MAX_LEN)

    registros = []
    registros.append(clasificar_uno(pipe, noticia_positiva, "Positiva"))
    registros.append(clasificar_uno(pipe, noticia_negativa, "Negativa"))
    registros.append(clasificar_uno(pipe, noticia_neutral,  "Neutral"))

    df = pl.DataFrame(registros)
    df_pretty = formatear_df(df)

    # Opcional: estilos de impresión de Polars
    pl.Config.set_tbl_rows(10)
    pl.Config.set_tbl_width_chars(120)

    print("\n=== Sentimiento (robertuito) — resumen ===")
    print(df_pretty)
    print("\nNota: 'Texto_usado_%' indica el **porcentaje de tokens del texto** que realmente entraron al modelo con MAX_LEN =", MAX_LEN)