import math
import random
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
class Algoritmos:
    class AlgGen:
        def ejecutaAlGen(tam_poblacion, num_generaciones, longitud_individuo, prob_mutacion): 
            # Función de aptitud
            def aptitud(individuo):
                suma = sum(individuo)
                return suma

            # Función de selección
            def seleccion(poblacion):
                puntuaciones = [aptitud(i) for i in poblacion]
                max_punt = max(puntuaciones)
                indice = puntuaciones.index(max_punt)
                return poblacion[indice]

            # Función de cruce
            def cruce(padre, madre):
                punto_cruce = random.randint(0, len(padre)-1)
                hijo1 = padre[:punto_cruce] + madre[punto_cruce:]
                hijo2 = madre[:punto_cruce] + padre[punto_cruce:]
                return hijo1, hijo2


            # Función de mutación
            def mutacion(individuo, prob_mutacion):
                for i in range(len(individuo)):
                    if random.random() < prob_mutacion:
                        individuo[i] = 1 - individuo[i]
                return individuo

            # Inicialización de la población
            poblacion = [[random.randint(0, 1) for i in range(longitud_individuo)] for j in range(tam_poblacion)]

            # Bucle principal
            for i in range(num_generaciones):
                # Selección de padres
                padre = seleccion(poblacion)
                madre = seleccion(poblacion)
                # Cruce
                hijo1, hijo2 = cruce(padre, madre)
                # Mutación
                hijo1 = mutacion(hijo1, prob_mutacion)
                hijo2 = mutacion(hijo2, prob_mutacion)
                # Reemplazo de la población
                poblacion.remove(seleccion(poblacion))
                poblacion.remove(seleccion(poblacion))
                poblacion.append(hijo1)
                poblacion.append(hijo2)

            # Obtención del mejor individuo
            mejor_individuo = seleccion(poblacion)
            print("El mejor individuo es:", mejor_individuo)
    

    class RecSim:
        def ejecuta_RecSim(solucion_inicial, solucion_actual, temperatura_inicial, factor_enfriamiento, num_iteraciones):
            # Función de evaluación
            def evaluacion(solucion):
                return sum(solucion)

            # Función de vecindario
            def vecindario(solucion):
                vecino = solucion[:]#revisar strip de cadenas [:] significa rango desde el principio al final
                i = random.randint(0, len(solucion)-1)
                vecino[i] = 1 - vecino[i]
                return vecino

            # Inicialización
            solucion_actual = solucion_inicial
            temperatura_actual = temperatura_inicial
            mejor_solucion = solucion_actual
            mejor_evaluacion = evaluacion(mejor_solucion)

            for i in range(num_iteraciones):
                # Generar vecino
                vecino = vecindario(solucion_actual)
                # Evaluar vecino
                evaluacion_vecino = evaluacion(vecino)
                # Calcular delta de evaluación
                delta_e = evaluacion_vecino - evaluacion(solucion_actual)
                # Si es mejor, aceptar vecino
                if delta_e < 0:
                    solucion_actual = vecino
                    if evaluacion_vecino < mejor_evaluacion:
                        mejor_solucion = vecino
                        mejor_evaluacion = evaluacion_vecino
                # Si es peor, aceptar con probabilidad e^(-delta_e/T)
                else:
                    prob_aceptar = math.exp(-delta_e / temperatura_actual)
                    if random.random() < prob_aceptar:
                        solucion_actual = vecino
                # Enfriar temperatura
                temperatura_actual *= factor_enfriamiento
            print("La mejor solución encontrada es:", mejor_solucion)
            print("Con una evaluación de:", mejor_evaluacion)



    class AntCol:
            def ejecutaAntCol(self, num_hormigas, alpha, beta, evaporacion_feromonas, max_iter, grafo):
                self.num_hormigas = num_hormigas
                self.alpha = alpha
                self.beta = beta
                self.evaporacion_feromonas = evaporacion_feromonas
                self.max_iter = max_iter
                self.grafo = grafo
                
                self.matriz_feromonas = np.ones_like(self.grafo) / len(self.grafo)
                self.matriz_distancias = 1 / self.grafo
                
            def ejecutar(self):
                mejor_camino = None
                mejor_distancia = float('inf')
                
                for i in range(self.max_iter):
                    caminos_hormigas = self.generar_caminos_hormigas()
                    distancias = [self.calcular_distancia(camino) for camino in caminos_hormigas]
                    
                    if min(distancias) < mejor_distancia:
                        mejor_distancia = min(distancias)
                        mejor_camino = caminos_hormigas[np.argmin(distancias)]
                    
                    cambios_feromonas = self.calcular_cambios_feromonas(caminos_hormigas, distancias)
                    self.actualizar_matriz_feromonas(cambios_feromonas)
                    
                return mejor_camino, mejor_distancia
            
            def generar_caminos_hormigas(self):
                caminos_hormigas = []
                
                for i in range(self.num_hormigas):
                    camino_hormiga = self.generar_camino_hormiga()
                    caminos_hormigas.append(camino_hormiga)
                    
                return caminos_hormigas
            
            def generar_camino_hormiga(self):
                nodo_inicial = np.random.randint(len(self.grafo))
                camino = [nodo_inicial]
                
                for i in range(len(self.grafo) - 1):
                    nodo_actual = camino[-1]
                    prob_transicion = self.calcular_prob_transicion(nodo_actual, camino)
                    nodo_siguiente = np.random.choice(range(len(self.grafo)), p=prob_transicion)
                    camino.append(nodo_siguiente)
                    
                return camino
            
            def calcular_prob_transicion(self, nodo_actual, camino):
                numerador = np.zeros(len(self.grafo))
                denominador = 0
                
                for nodo in range(len(self.grafo)):
                    if nodo not in camino:
                        feromona = self.matriz_feromonas[nodo_actual, nodo] ** self.alpha
                        distancia = self.matriz_distancias[nodo_actual, nodo] ** (-self.beta)
                        feromona_por_distancia = feromona * distancia

                        numerador[nodo] = feromona_por_distancia
                        denominador += feromona_por_distancia
                    
                return numerador / denominador
            
            def calcular_distancia(self, camino):
                distancia_total = 0
                
                for i in range(len(camino) - 1):
                    distancia_total += self.matriz_distancias[camino[i], camino[i+1]]
                    
                distancia_total += self.matriz_distancias[camino[-1], camino[0]]
                
                return distancia_total
            
            def calcular_cambios_feromonas(self, caminos_hormigas, distancias):
                cambios_feromonas = np.zeros_like(self.matriz_feromonas)
        
                for camino_hormiga, distancia in zip(caminos_hormigas, distancias):
                    for i in range(len(camino_hormiga) - 1):
                        nodo_actual = camino_hormiga[i]
                        nodo_siguiente = camino_hormiga[i+1]
                        cambios_feromonas[nodo_actual, nodo_siguiente] += 1 / distancia
                        
                        return cambios_feromonas
    
            def actualizar_matriz_feromonas(self, cambios_feromonas):
                self.matriz_feromonas = (1 - self.evaporacion_feromonas) * self.matriz_feromonas + cambios_feromonas
                

    class pso:
        
        def pso_2d(population, dimension, position_min, position_max, generation, fitness_criterion):
             # Fitness function
        # We assume the problem can be expressed by the following equation: 
        # f(x1,x2)=(x1+2*-x2+3)^2 + (2*x1+x2-8)^2
        # The objective is to find a minimum which is 0

            def fitness_function(x1,x2):
                f1=x1+2*-x2+3
                f2=2*x1+x2-8
                z = f1**2+f2**2
                return z
            
            def update_velocity(particle, velocity, pbest, gbest, w_min=0.5, max=1.0, c=0.1):
                # Initialise new velocity array
                num_particle = len(particle)
                new_velocity = np.array([0.0 for i in range(num_particle)])
                # Randomly generate r1, r2 and inertia weight from normal distribution
                r1 = random.uniform(0,max)
                r2 = random.uniform(0,max)
                w = random.uniform(w_min,max)
                c1 = c
                c2 = c
                # Calculate new velocity
                for i in range(num_particle):
                    new_velocity[i] = w*velocity[i] + c1*r1*(pbest[i]-particle[i])+c2*r2*(gbest[i]-particle[i])
                return new_velocity

            def update_position(particle, velocity):
                # Move particles by adding velocity
                new_particle = particle + velocity
                return new_particle
            
            # Initialisation
            # Population
            particles = [[random.uniform(position_min, position_max) for j in range(dimension)] for i in range(population)]
            # Particle's best position
            pbest_position = particles
            # Fitness
            pbest_fitness = [fitness_function(p[0],p[1]) for p in particles]
            # Index of the best particle
            gbest_index = np.argmin(pbest_fitness)
            # Global best particle position
            gbest_position = pbest_position[gbest_index]
            # Velocity (starting from 0 speed)
            velocity = [[0.0 for j in range(dimension)] for i in range(population)]
            
            # Loop for the number of generation
            for t in range(generation):
                # Stop if the average fitness value reached a predefined success criterion
                if np.average(pbest_fitness) <= fitness_criterion:
                    break
                else:
                    for n in range(population):
                        # Update the velocity of each particle
                        velocity[n] = update_velocity(particles[n], velocity[n], pbest_position[n], gbest_position)
                        # Move the particles to new position
                        particles[n] = update_position(particles[n], velocity[n])
                # Calculate the fitness value
                pbest_fitness = [fitness_function(p[0],p[1]) for p in particles]
                # Find the index of the best particle
                gbest_index = np.argmin(pbest_fitness)
                # Update the position of the best particle
                gbest_position = pbest_position[gbest_index]

            # Print the results
            print('Global Best Position: ', gbest_position)
            print('Best Fitness Value: ', min(pbest_fitness))
            print('Average Particle Best Fitness Value: ', np.average(pbest_fitness))
            print('Number of Generation: ', t)

            
            # Plotting prepartion
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            x = np.linspace(position_min, position_max, 80)
            y = np.linspace(position_min, position_max, 80)
            X, Y = np.meshgrid(x, y)
            Z= fitness_function(X,Y)
            ax.plot_wireframe(X, Y, Z, color='r', linewidth=0.2)

            # Animation image placeholder
            images = []

            # Add plot for each generation (within the generation for-loop)
            image = ax.scatter3D([
                                particles[n][0] for n in range(population)],
                                [particles[n][1] for n in range(population)],
                                [fitness_function(particles[n][0],particles[n][1]) for n in range(population)], c='b')
            images.append([image])

            # Generate the animation image and save
            animated_image = animation.ArtistAnimation(fig, images)
            animated_image.save('./pso_simple.gif', writer='pillow') 

    class RedNeuronal:
        def red_neuronal(datos_entrenamiento, datos_prueba, capas, num_iteraciones=1000, tasa_aprendizaje=0.1):
            
            def evaluar_red_neuronal(datos, pesos, bias):
                aciertos = 0
                for x, y in datos:
                    a = x
                    for b, w in zip(bias, pesos):
                        a = sigmoid(np.dot(w, a) + b)
                    prediccion = np.argmax(a)
                    if prediccion == np.argmax(y):
                        aciertos += 1
                return aciertos / len(datos)

            def sigmoid(z):
                return 1.0 / (1.0 + np.exp(-z))

            def sigmoid_derivada(z):
                return sigmoid(z) * (1 - sigmoid(z))
            # Inicializar los pesos y bias aleatoriamente
            num_entradas = capas[0]
            num_salidas = capas[-1]
            num_capas_ocultas = len(capas) - 2
            bias = [np.random.randn(y, 1) for y in capas[1:]]
            pesos = [np.random.randn(y, x) for x, y in zip(capas[:-1], capas[1:])]

            # Entrenar la red neuronal
            for i in range(num_iteraciones):
                # Calcular el gradiente para cada dato de entrenamiento
                grad_bias = [np.zeros(b.shape) for b in bias]
                grad_pesos = [np.zeros(w.shape) for w in pesos]
                for x, y in datos_entrenamiento:
                    # Forward propagation
                    a = x
                    activaciones = [x]
                    zs = []
                    for b, w in zip(bias, pesos):
                        z = np.dot(w, a) + b
                        zs.append(z)
                        a = sigmoid(z)
                        activaciones.append(a)

                    # Backward propagation
                    delta = (activaciones[-1] - y) * sigmoid_derivada(zs[-1])
                    grad_bias[-1] = delta
                    grad_pesos[-1] = np.dot(delta, activaciones[-2].transpose())
                    for j in range(2, num_capas_ocultas+2):
                        z = zs[-j]
                        sp = sigmoid_derivada(z)
                        delta = np.dot(pesos[-j+1].transpose(), delta) * sp
                        grad_bias[-j] = delta
                        grad_pesos[-j] = np.dot(delta, activaciones[-j-1].transpose())

                # Actualizar los pesos y bias con el gradiente
                for j in range(len(pesos)):
                    pesos[j] -= tasa_aprendizaje * grad_pesos[j] / len(datos_entrenamiento)
                    bias[j] -= tasa_aprendizaje * grad_bias[j] / len(datos_entrenamiento)

                # Evaluar la precisión de la red neuronal en los datos de prueba
                precision = evaluar_red_neuronal(datos_prueba, pesos, bias)
                print("Iteración {0}: precisión en datos de prueba {1}".format(i, precision))

            return pesos, bias

        
