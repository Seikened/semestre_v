import numpy as np
import random
import time

class RedNeuronalBackpropagation:
    def __init__(self, capas, tasa_aprendizaje: float = 0.1) -> None:
        self.capas = capas
        self.tasa_aprendizaje = tasa_aprendizaje
        self.num_capas = len(capas)
        
        # Inicializar pesos y biases
        self.pesos = []
        self.sesgos = []
        
        for i in range(self.num_capas - 1):
            peso = np.random.rand(self.capas[i+1], self.capas[i]) * 0.1
            self.pesos.append(peso)

            sesgo = np.zeros((capas[i+1], 1))
            self.sesgos.append(sesgo)
            
    
    def funcion_activacion(self,x, derivada = False):    
        """ 
        Función de activación a implementar es la sigmoide (su derivada también)
        Args:
            x (np.array): valor de entrada
            derivada (bool, optional): Si es True, calcula la derivada. Defaults to False.
        Retorna: 
            np.array: valor de salida
        """
        
        
        sig = 1 / (1 + np.exp(-x))
        
        if derivada:
            return sig * (1 - sig)
        return sig


    def propagacion_adelante(self, entrada):
        """
        Propagación hacia adelante a implementar
        Args:
            entrada (np.array): vector de entrada
        Retorna: 
            salidas (tuple): (salida_capas, activaciones_capas)
        """
        # Convertir la entrada en un array 2D columna si es necesario
        entrada = np.array(entrada).reshape(-1, 1)
        
        # Almacenar todas las salidas y activaciones de cada capa
        salidas = [entrada] # La capa de entrada es la misma
        activaciones = [entrada] # Para cada capa de entrada, activación = entrada
        
        
        for i in range(self.num_capas - 1):
            # Calculas la entrada neta para la siguiente capa
            # z = W * a + b
            """
            W: pesos de la capa actual
            a: activación de la capa anterior
            b: sesgo de la capa actual
            z: entrada neta a la capa actual
            """
            W = self.pesos[i]
            a = salidas[-1]
            b = self.sesgos[i]
            
            z = np.dot(W, a) + b
            salidas.append(z)
            
            # Aplicar funcion de activacion
            a = self.funcion_activacion(z)
            activaciones.append(a)

        return salidas, activaciones
            

        
    
    
    
    def propagacion_atras(self,entrada, objetivo, salidas, activaciones):
        """
        Realizar la propagación hacia atrás para calcular los gradientes de los pesos y sesgos
        Args:
            entrada (np.array): vector de entrada
            objetivo (np.array): vector objetivo
            salidas (list): lista de salidas de cada capa
            activaciones (list): lista de activaciones de cada capa
        Retorna:
            tuple: (gradientes_pesos, gradientes_sesgos)
        """
        # Convertir el objetivo en un array 2D
        objetivo = np.array(objetivo, ndmin=2).T
        
        # Inicializar listas para los gradinetes
        gradientes_pesos = [np.zeros_like(peso) for peso in self.pesos]
        gradientes_sesgos = [np.zeros_like(sesgo) for sesgo in self.sesgos]
        
        # Calcular el error en la capa de salida
        error = (activaciones[-1] - objetivo) * self.funcion_activacion(salidas[-1], derivada=True)
        
        # Almacenar gradientes para la última capa
        gradientes_pesos[-1] = np.dot(error, activaciones[-2].T)
        gradientes_sesgos[-1] = error
        
        
        # Propagar el error hacia atrás
        # SECCIÓN PARALELIZABLE: Calculo de errores por capa puede ser paralelizado
        for layer in range(2, self.num_capas):
            z = salidas[-layer]
            derivada_activacion = self.funcion_activacion(z, derivada=True)
            
            error = np.dot(self.pesos[-layer+1].T, error) * derivada_activacion
            
            # Calcular gradientes
            gradientes_pesos[-layer] = np.dot(error, activaciones[-layer-1].T)
            gradientes_sesgos[-layer] = error
            
        return gradientes_pesos, gradientes_sesgos
    
    
    def actializar_pesos(self, gradientes_pesos, gradientes_sesgos):
        """
        Actualiza los pesos y sesgos usando los gradientes calculados
        Args:
            gradientes_pesos (list): lista de gradientes de pesos
            gradientes_sesgos (list): lista de gradientes de sesgos
        """
        
        # SECCIÓN PARALELIZABLE: Actualización de pesos y sesgos puede ser paralelizado
        for i in range(len(self.pesos)):
            self.pesos[i] -= self.tasa_aprendizaje * gradientes_pesos[i]
            self.sesgos[i] -= self.tasa_aprendizaje * gradientes_sesgos[i]
    
    
    def entrenar(self, entradas_entrenamiento,objetivos_entrenamiento, epocas, mostrar_progreso = True):
        """
        Entrenar la red neuronal usando backpropagation
        Args:
            entradas_entrenamiento (list): lista de vectores de entrada
            objetivos_entrenamiento (list): lista de vectores objetivo
            epocas (int): número de épocas para entrenar
            mostrar_progreso (bool, optional): Si es True, muestra el progreso. Defaults to True.
        """
        num_muestras = len(entradas_entrenamiento)
        
        
        for epoca in range(epocas):
            error_total = 0
            inicio_epoca = time.time()
            
            # SECCIÓN PARALELIZABLE: Entrenamiento por épocas puede ser paralelizado
            for i in range(num_muestras):
                salidas, activaciones = self.propagacion_adelante(entradas_entrenamiento[i])
                
                # Calcular el error cuadrático medio
                objetivo = np.array(objetivos_entrenamiento[i], ndmin=2).T
                error_total += np.mean((activaciones[-1] - objetivo) ** 2)
                
                # Propagación hacia atrás
                gradcientes_pesos, gradientes_sesgos = self.propagacion_atras(
                    entradas_entrenamiento[i],
                    objetivos_entrenamiento[i],
                    salidas,
                    activaciones
                )
                
                # Actualizar pesos
                self.actializar_pesos(gradcientes_pesos, gradientes_sesgos)
            fin_epoca = time.time()
            
            # Mostrar progreso
            if mostrar_progreso:
                print(f"Época {epoca+1}/{epocas} - Error: {error_total/num_muestras:.6f} - Tiempo: {fin_epoca - inicio_epoca:.2f}s")
    
    
    def predecir(self, entrada):
        """
        Realiza una predicción para una entrada dada
        Args:
            entrada (np.array): vector de entrada
        Retorna:
            np.array: vector de salida predicho
        """
        _, activaciones = self.propagacion_adelante(entrada)
        return activaciones[-1]


    def calcular_precision(self, entradas_prueba, objetivos_prueba, umbral=0.5):
        predicciones = []
        for entrada in entradas_prueba:
            salida = self.predecir(entrada)
            predicciones.append((salida >= umbral).astype(int))
        predicciones = np.array(predicciones).squeeze()
        objetivos = np.array(objetivos_prueba).squeeze()
        precision = np.mean(predicciones == objetivos)
        return precision
    

if __name__ == "__main__":
        # Datos de entrenamiento: y = 2x
    X_train = np.array([[x] for x in range(11)])  # 0,1,2,...,10
    y_train = np.array([[2 * x] for x in range(11)])  # 0,2,4,...,20

    # Arquitectura: 1 entrada, 1 salida, con 1 capa oculta pequeña
    topologia = [1, 3, 1]  
    red = RedNeuronalBackpropagation(topologia, tasa_aprendizaje=0.01)

    # Entrenamiento
    red.entrenar(X_train, y_train, epocas=5000)

    # Prueba
    x_prueba = np.array([[4.0]])
    prediccion = red.predecir(x_prueba)
    print(f"\nPredicción para x={x_prueba[0][0]} → y={prediccion[0][0]:.2f}")

    # Mostrar todos los resultados
    print("\nResultados:")
    for x in range(11):
        pred = red.predecir(np.array([[x]]))[0][0]
        print(f"x={x:2d}, y_real={2*x:4.1f}, y_pred={pred:6.2f}")
    