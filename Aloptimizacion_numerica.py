import math
import numpy as np
class AlgON:
    class AlSeccionDorada:
        def golden_section_search(f, a, b, tol=1e-6):
        #Encuentra el mínimo de una función f en el intervalo [a,b] utilizando el método de la sección dorada.
            gr = (math.sqrt(5) + 1) / 2  # razón áurea
            c = b - (b - a) / gr
            d = a + (b - a) / gr
            while abs(c - d) > tol:
                if f(c) < f(d):
                    b = d
                else:
                    a = c
                # calcula los nuevos valores de c y d
                c = b - (b - a) / gr
                d = a + (b - a) / gr

            return (b + a) / 2

    class AlNewton:
        def newton(f, df, x0, tol=1e-6, max_iter=100):
            # Inicializamos la aproximación y la iteración
            x = x0
            iteracion = 0

            # Iteramos hasta cumplir el criterio de parada o alcanzar el número máximo de iteraciones
            while abs(f(x)) > tol and iteracion < max_iter:

                # Calculamos el paso utilizando la fórmula de Newton
                paso = f(x) / df(x)

                # Actualizamos la aproximación
                x = x - paso

                # Incrementamos la iteración
                iteracion += 1

            # Devolvemos la aproximación del mínimo
            return x
        
    class QuasiNewton:
        def bfgs(f, grad_f, x0, tol=1e-6, max_iter=100):
            # Inicializamos la aproximación y la iteración
            x = x0
            H = np.eye(len(x0))
            iteracion = 0

            # Iteramos hasta cumplir el criterio de parada o alcanzar el número máximo de iteraciones
            while np.linalg.norm(grad_f(x)) > tol and iteracion < max_iter:

                # Calculamos la dirección de descenso como la solución del sistema H p = -grad_f(x)
                p = np.linalg.solve(H, -grad_f(x))

                # Actualizamos la aproximación
                x_new = x + p

                # Calculamos el vector s y el vector y para actualizar la matriz H
                s = x_new - x
                y = grad_f(x_new) - grad_f(x)

                # Actualizamos la matriz H utilizando la fórmula de BFGS
                rho = 1 / np.dot(y, s)
                H = (np.eye(len(x0)) - rho * np.outer(s, y)) @ H @ (np.eye(len(x0)) - rho * np.outer(y, s)) + rho * np.outer(s, s)

                # Actualizamos la aproximación y la iteración
                x = x_new
                iteracion += 1

            # Devolvemos la aproximación del mínimo
            return x
        
    class AlBisec:
        def biseccion(f, a, b, tol=1e-6, max_iter=100):
            # Comprobamos que f(a) y f(b) tengan signos diferentes
            assert f(a) * f(b) < 0, "f(a) y f(b) deben tener signos diferentes"

            # Inicializamos la aproximación y la iteración
            c = (a + b) / 2
            iteracion = 0

            # Iteramos hasta cumplir el criterio de parada o alcanzar el número máximo de iteraciones
            while abs(f(c)) > tol and iteracion < max_iter:

                # Calculamos el punto medio del intervalo [a,b]
                c = (a + b) / 2

                # Si f(c) tiene el mismo signo que f(a), actualizamos a = c
                if f(c) * f(a) > 0:
                    a = c

                # Si f(c) tiene el mismo signo que f(b), actualizamos b = c
                elif f(c) * f(b) > 0:
                    b = c

                # Si f(c) es 0, hemos encontrado la raíz exacta
                else:
                    return c

                # Actualizamos la iteración
                iteracion += 1

            # Devolvemos la aproximación de la raíz
            return c