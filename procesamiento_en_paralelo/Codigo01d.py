import Codigo01a
import Codigo01b
import Codigo01c
import threading
import time
import random
import multiprocessing
import os
import concurrent.futures

def comparar_implementaciones():
    """
    Compara las tres implementaciones: threading, concurrent.futures, multiprocessing
    """
    print("=" * 80)
    print("🔬 COMPARATIVA DE IMPLEMENTACIONES PARALELAS")
    print("=" * 80)
    
    # Tamaño de prueba
    num_estudiantes = 8
    
    print(f"📊 Ejecutando prueba con {num_estudiantes} estudiantes")
    print("=" * 80)
    
    # Resultados de cada implementación
    resultados_threading = []
    resultados_futures = []
    resultados_multiprocessing = []
    
    # 1. Threading (implementación original)
    print("\n1️⃣  EJECUTANDO threading...")
    start_time = time.time()
    resultados_threading = Codigo01a.main_threading()
    tiempo_threading = time.time() - start_time
    
    # 2. concurrent.futures
    print("\n2️⃣  EJECUTANDO concurrent.futures...")
    start_time = time.time()
    resultados_futures = main_concurrent_futures()
    tiempo_futures = time.time() - start_time
    
    # 3. multiprocessing
    print("\n3️⃣  EJECUTANDO multiprocessing...")
    start_time = time.time()
    resultados_multiprocessing = main_multiprocessing()
    tiempo_multiprocessing = time.time() - start_time
    
    # Análisis comparativo
    print("\n" + "=" * 80)
    print("📈 ANÁLISIS COMPARATIVO")
    print("=" * 80)
    
    # Comparar tiempos
    print(f"⏱️  TIEMPOS DE EJECUCIÓN:")
    print(f"   threading:          {tiempo_threading:.3f} segundos")
    print(f"   concurrent.futures: {tiempo_futures:.3f} segundos")
    print(f"   multiprocessing:    {tiempo_multiprocessing:.3f} segundos")
    
    # Comparar resultados
    calif_threading = [r['calificacion'] for r in resultados_threading]
    calif_futures = [r['calificacion'] for r in resultados_futures]
    calif_multiprocessing = [r['calificacion'] for r in resultados_multiprocessing]
    
    print(f"\n📊 CALIFICACIONES PROMEDIO:")
    print(f"   threading:          {sum(calif_threading)/len(calif_threading):.1f}/100")
    print(f"   concurrent.futures: {sum(calif_futures)/len(calif_futures):.1f}/100")
    print(f"   multiprocessing:    {sum(calif_multiprocessing)/len(calif_multiprocessing):.1f}/100")
    
    # Conclusiones
    print(f"\n💡 CONCLUSIONES:")
    print(f"   • concurrent.futures: Más limpio y Pythonico")
    print(f"   • multiprocessing: Paralelismo real (sin GIL)")
    print(f"   • threading: Más control manual")
    
    return {
        'threading': {'tiempo': tiempo_threading, 'resultados': resultados_threading},
        'futures': {'tiempo': tiempo_futures, 'resultados': resultados_futures},
        'multiprocessing': {'tiempo': tiempo_multiprocessing, 'resultados': resultados_multiprocessing}
    }

def explicacion_tecnica():
    """
    Explica las diferencias técnicas entre los enfoques
    """
    print("\n" + "=" * 80)
    print("📚 EXPLICACIÓN TÉCNICA")
    print("=" * 80)
    
    explicaciones = [
        "🧵 THREADING:",
        "   • Hilos dentro del mismo proceso",
        "   • Comparten memoria (fácil comunicación)",
        "   • Limitado por GIL (Global Interpreter Lock)",
        "   • Bueno para I/O-bound tasks",
        "",
        "🎯 CONCURRENT.FUTURES:",
        "   • Abstracción de alto nivel sobre threading/multiprocessing",
        "   • ThreadPoolExecutor para I/O-bound",
        "   • ProcessPoolExecutor para CPU-bound",
        "   • API uniforme y Pythonica",
        "",
        "🖥️  MULTIPROCESSING:",
        "   • Procesos separados (memoria independiente)",
        "   • Sin limitación de GIL",
        "   • Comunicación más compleja (queues, pipes)",
        "   • Bueno para CPU-bound tasks",
        "",
        "🔍 ¿CUÁNDO USAR CADA UNO?",
        "   • I/O-bound (red, disco, BD): threading o concurrent.futures",
        "   • CPU-bound (cálculos intensivos): multiprocessing",
        "   • Código simple: concurrent.futures",
        "   • Control fino: threading o multiprocessing"
    ]
    
    for linea in explicaciones:
        print(linea)
        time.sleep(0.1)

if __name__ == "__main__":
    # Ejecutar comparativa
    resultados = comparar_implementaciones()
    
    # Mostrar explicación técnica
    input("\nPresiona Enter para ver la explicación técnica...")
    explicacion_tecnica()
    
    print("\n" + "=" * 80)
    print("🎉 COMPARATIVA COMPLETADA!")
    print("=" * 80)