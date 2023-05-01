import Algoritmos_Anidados as AE
import Aloptimizacion_numerica as AO

Select= int(input("Ingrese la opcion del algoritmo que desea testear:\n"))
match(Select):
    case 1:
        solucion_inicial= int(input("Solucion inicial: \n")) 
        solucion_actual = int(input("Solucion actual: \n")) 
        temperatura_inicial = int(input("Temperatura inicial: \n")) 
        factor_enfriamiento = int(input("Factor de enfriamiento: \n")) 
        num_iteraciones = int(input("Numero de iteraciones: \n")) 
        AE.Algoritmos.RecSim.ejecuta_RecSim(solucion_inicial, solucion_actual, temperatura_inicial, factor_enfriamiento, num_iteraciones)
    case 2:
        tam_poblacion= int(input("tam_poblacion: \n")) 
        num_generaciones = int(input("num_generaciones: \n")) 
        longitud_individuo = int(input("longitud_individuo: \n")) 
        prob_mutacion = int(input("prob_mutacion: \n")) 
        AE.Algoritmos.AlgGen.ejecutaAlGen(tam_poblacion, num_generaciones, longitud_individuo, prob_mutacion)
    case 3:
        num_hormigas = int(input("num_hormigas: \n")) 
        max_iter = int(input("max_iter: \n")) 
        n_iteraciones = int(input("n_iteraciones: \n"))
        evaporacion_feromonas = int(input("evaporacion_feromonas: \n")) 
        alpha = int(input("alpha: \n")) 
        beta = int(input("beta: \n"))  
        grafo = int(input("grafo: \n"))  
        AE.Algoritmos.AntCol.ejecutaAntCol(num_hormigas, alpha, beta, evaporacion_feromonas, max_iter, grafo)
    case 4:
        population = int(input("population: \n")) 
        dimension = int(input("dimension: \n")) 
        position_min = int(input("position_min: \n"))
        position_max = int(input("position_max: \n")) 
        generation = int(input("generation: \n")) 
        fitness_criterion = int(input("fitness_criterion: \n"))  
        AE.Algoritmos.pso.pso_2d(population, dimension, position_min, position_max, generation, fitness_criterion)

    case 5:
        Funcion = int(input("Funcion: \n")) 
        Valor1_Intervalo = int(input("Valor1_Intervalo: \n")) 
        Valor2_Intervalo = int(input("Valor2_Intervalo: \n"))
        r = AO.AlgON.AlSeccionDorada.golden_section_search(Funcion, Valor1_Intervalo, Valor2_Intervalo)
        print("Resultado: ", r)

    case 6:
        Funcion = int(input("Funcion: \n")) 
        df = int(input("Derivada de la Funcion: \n")) 
        x0 = float(input("Valor inicial de la Aproximacion: \n"))
        r = AO.AlgON.AlNewton.newton(Funcion, df, x0)
        print("Resultado: ", r)

    case 7:
        Funcion = int(input("Funcion: \n")) 
        grad_f = int(input("Gradiente de la Función Objetivo F: \n")) 
        x0 = float(input("Valor inicial de la Aproximacion: \n"))
        r = AO.AlgON.QuasiNewton.bfgs(Funcion, grad_f, x0)
        print("Resultado: ", r)

    case 8:
        Funcion = int(input("Funcion: \n")) 
        a = int(input("Extremo izquierdo del intervalo: \n")) 
        b = float(input("extremo derecho del intervalo: \n"))
        r = AO.AlgON.AlBisec.biseccion(Funcion, a, b)
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