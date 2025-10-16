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
            

        
    
    
    
    def retropropagacion(self,entrada, objetivo, salidas, activaciones):
        pass
    
    
    def actializar_pesos(self, gradientes_pesos, gradientes_sesgos):
        pass
    
    
    def entrenar(self, entradas_entrenamiento,objetivos_entrenamiento, epocas, mostrar_progreso = True):
        pass
    
    
    def predecir(self, entrada):
        pass


    def calcular_precision(self, entradas_prueba, objetivos_prueba, umbral=0.5):
        pass
    
    
    