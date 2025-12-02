from paralelizado import main_paralelizado
from proyectofinal_evoluciondiferencial_recocidosimulado import main_no_paralelizado
import matplotlib.pyplot as plt


n_gen = 10_000
pso_no_paralelizado, lista_tiempos_no_paralelizado = main_no_paralelizado(n_gen)
pso_paralelizado, lista_tiempos_paralelizado = main_paralelizado(n_gen)

speedup = pso_no_paralelizado / pso_paralelizado
print(f"Speedup: {speedup:.2f}x")


# Gráfica de comparación de tiempos por generación
plt.figure(figsize=(10, 6))
plt.plot(lista_tiempos_no_paralelizado, label='No Paralelizado', alpha=0.7)
plt.plot(lista_tiempos_paralelizado, label='Paralelizado', alpha=0.7)
plt.xlabel('Generación')
plt.ylabel('Tiempo por generación (segundos)')
plt.title(f'Comparación de Tiempos por Generación, speedup: {speedup:.2f}x')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
