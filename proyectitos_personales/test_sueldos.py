import pandas as pd

# Diccionario meses en español con orden explícito
MESES_ES = {
    "ENE": "Enero", "FEB": "Febrero", "MAR": "Marzo", "ABR": "Abril",
    "MAY": "Mayo", "JUN": "Junio", "JUL": "Julio", "AGO": "Agosto",
    "SEP": "Septiembre", "OCT": "Octubre", "NOV": "Noviembre", "DIC": "Diciembre"
}

ORDEN = {
    "Enero": 1, "Febrero": 2, "Marzo": 3, "Abril": 4, "Mayo": 5,
    "Junio": 6, "Julio": 7, "Agosto": 8, "Septiembre": 9,
    "Octubre": 10, "Noviembre": 11, "Diciembre": 12
}

# Cargar CSV
df = pd.read_csv("/Users/ferleon/Github/semestre_v/proyectitos_personales/data/spei_enviado_banorte.csv")

# Extraer mes usando las abreviaturas del CSV
df["MES_ABREV"] = df["fecha"].str.split("/").str[1].str.upper()
df["mes"] = df["MES_ABREV"].map(MESES_ES)

print("\n================  REPORTE DE ENVÍOS A BANORTE  ================\n")

# Total enviado
costo_de_hora_laboral = 70
total = df["monto"].sum()
horas_equivalentes = total / costo_de_hora_laboral

print(f"Cantidad total enviada a BANORTE:  ${total:,.2f} MXN")
print(f"Equivale a:                        {horas_equivalentes:,.2f} horas de trabajo")
print("------------------------------------------------------------------\n")

# Agrupar por mes (ordenado correctamente)
gasto_mensual = df.groupby("mes")["monto"].sum().reset_index()
gasto_mensual["orden"] = gasto_mensual["mes"].map(ORDEN)
gasto_mensual = gasto_mensual.sort_values("orden")

print("Gasto por mes:")
for _, row in gasto_mensual.iterrows():
    print(f"  {row['mes']:12s} →  ${row['monto']:,.2f}")

print("\n------------------------------------------------------------------\n")

# Estadísticas descriptivas
promedio = df["monto"].mean()
maximo = df["monto"].max()
minimo = df["monto"].min()
num_movimientos = len(df)

print("Resumen estadístico claro e intuitivo:")
print(f"• Número total de envíos registrados:     {num_movimientos}")
print(f"• Monto promedio por envío:              ${promedio:,.2f}")
print(f"• Envío más alto registrado:             ${maximo:,.2f}")
print(f"• Envío más bajo registrado:             ${minimo:,.2f}")
print("------------------------------------------------------------------\n")

print("Interpretación sencilla:")
print(f"Durante el periodo analizado se hicieron {num_movimientos} envíos a BANORTE.")
print(f"Cada envío tuvo un costo promedio de ${promedio:,.2f}.")
print(f"El envío más pequeño fue de ${minimo:,.2f} y el más alto alcanzó ${maximo:,.2f}.")
print(f"En total, estos movimientos representan ${total:,.2f},")
print(f"que equivale aproximadamente a {horas_equivalentes:,.1f} horas de trabajo remunerado.")
print("\n==================================================================\n")