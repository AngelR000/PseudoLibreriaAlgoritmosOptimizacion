import Algoritmos_Anidados as AE
import Aloptimizacion_numerica as AO
import numpy as np

Select= print("---Ingrese la opcion del algoritmo que desea testear---")
opciones = int(input(" 1= Recocido Simulado\n 2= Algoritmo Genetico\n 3= Colonia de Hormigas\n 4= Optimización por Enjambre de Partículas\n 5= Golden Section Search\n 6= Algoritmo Newton\n 7= Quasi Newton\n 8= Biseccion\n 9= Red Neuronal 1\n 10= Red Neuronal 2:\n"))

match(opciones):
    case 1:
        solucion_inicial = input("Ingrese la solución inicial (una lista de valores separados por comas): ")
        solucion_inicial = [int(x) for x in solucion_inicial.split(",")] 
        solucion_actual = solucion_inicial
        temperatura_inicial = float(input("Ingrese la temperatura inicial: "))
        factor_enfriamiento = float(input("Ingrese el factor de enfriamiento: "))
        num_iteraciones = int(input("Ingrese el número de iteraciones: "))
        AE.Algoritmos.RecSim.ejecuta_RecSim(solucion_inicial, solucion_actual, temperatura_inicial, factor_enfriamiento, num_iteraciones)
    case 2:
        tam_poblacion= int(input("tam_poblacion: \n")) 
        num_generaciones = int(input("num_generaciones: \n")) 
        longitud_individuo = int(input("longitud_individuo: \n")) 
        prob_mutacion = float(input("prob_mutacion: \n")) 
        AE.Algoritmos.AlgGen.ejecutaAlGen(tam_poblacion, num_generaciones, longitud_individuo, prob_mutacion)
    case 3:
        num_hormigas = int(input("num_hormigas: \n")) 
        max_iter = int(input("max_iter: \n")) 
        evaporacion_feromonas = float(input("evaporacion_feromonas: \n")) 
        alpha = float(input("alpha: \n")) 
        beta = float(input("beta: \n"))  
        # Crear el grafo según los datos ingresados
        grafo = {
                    'A': {'B': 2, 'C': 3, 'D': 4},
                    'B': {'A': 2, 'C': 5, 'D': 6},
                    'C': {'A': 3, 'B': 5, 'D': 7},
                    'D': {'A': 4, 'B': 6, 'C': 7}
                }
        ant_col = AE.Algoritmos.AntCol(num_hormigas, alpha, beta, evaporacion_feromonas, max_iter, grafo)
        mejor_camino, mejor_distancia= ant_col.ejecutaAntCol()
        print("Mejor camino:", mejor_camino)
        print("Mejor distancia:", mejor_distancia)
    case 4:
        population = int(input("population: \n")) 
        dimension = int(input("dimension: \n")) 
        position_min = int(input("position_min: \n"))
        position_max = int(input("position_max: \n")) 
        generation = int(input("generation: \n")) 
        fitness_criterion = float(input("fitness_criterion: \n"))  
        AE.Algoritmos.pso.pso_2d(population, dimension, position_min, position_max, generation, fitness_criterion)

    case 5:
        def funcion(x):
            return x**2 - 4*x
        Valor1_Intervalo = int(input("Valor1_Intervalo: \n")) 
        Valor2_Intervalo = int(input("Valor2_Intervalo: \n"))
        r = AO.AlgON.AlSeccionDorada.golden_section_search(funcion, Valor1_Intervalo, Valor2_Intervalo)
        print("Resultado: ", r)

    case 6:
        funcion_str = input("Ingrese la función: ")
        funcion = lambda x: eval(funcion_str)
        derivada_str = input("Ingrese la derivada de la función: ")
        derivada = lambda x: eval(derivada_str)
        x0 = float(input("Ingrese el valor inicial: "))
        r = AO.AlgON.AlNewton.newton(funcion, derivada, x0)
        print("Resultado: ", r)

    case 7:
        def f(x):
            return x[0]**2 + x[1]**2

        def grad_f(x):
            return np.array([2*x[0], 2*x[1]])
        
        x0 = np.array([float(input("Ingrese el valor inicial para x0: ")), float(input("Ingrese el valor inicial para x1: "))])
        tol = float(input("Ingrese la tolerancia: "))
        max_iter = int(input("Ingrese el número máximo de iteraciones: "))
        r = AO.AlgON.QuasiNewton.bfgs(f, grad_f, x0)
        print("Resultado:", r)

    case 8:
        def f(x):
            return x**3 - 2*x - 5
        a = float(input("Extremo izquierdo del intervalo: \n")) 
        b = float(input("extremo derecho del intervalo: \n"))
        r = AO.AlgON.AlBisec.biseccion(f, a, b)
        print("Resultado: ", r)

    case 9:
        print("Ingrese los datos de entrenamiento en el siguiente formato: 'x1 y1 z1, x2 y2 z2, ...'")
        data_entrenamiento = input()
        data_entrenamiento = [tuple(map(float, punto.split())) for punto in data_entrenamiento.split(",")]
        print("Ingrese los datos de prueba en el siguiente formato: 'x1 y1 z1, x2 y2 z2, ...'")
        data_prueba = input()
        data_prueba = [tuple(map(float, punto.split())) for punto in data_prueba.split(",")]
        Capas = float(input("Capas: \n"))
        AE.Algoritmos.RedNeuronal.red_neuronal(data_entrenamiento, data_prueba, Capas)

    case 10:
        
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) 
        y = np.array([[0], [1], [1], [0]])  

        input_size = X.shape[1]  
        hidden_size = 4  
        output_size = 1  

        learning_rate = 0.1  
        epochs = 10000  
        
        nn = AE.Algoritmos.NeuralNetwork2(input_size, hidden_size, output_size)
        nn.train(X, y, learning_rate, epochs)
        predictions = nn.forward(X)
        print("Predicciones:", predictions)