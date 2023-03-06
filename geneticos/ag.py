# Librerias de terceros.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Librerias propias.
from .utils import conteo, decodificar_genoma, aptitud_poblacion, combinar_poblaciones
from .operadores import inicializar_poblacion, seleccion, cruce, mutacion


def AG(
    Pi: int,
    SG: 'list | tuple',
    MAXi: float,
    PM: 'function | float',
    PMg: 'function | float',
    FA: 'function',
    continuo: 'list | tuple | None' = None
):
    '''
        Algoritmo genético.

        Params:
            - Pi: longitud de poblacion inicial.
            - SG: secciones del genoma.
            - MAXi: iteraciones maximas.
            - PM: prob. mutación por individuo.
            - PMg: prob. mutación por gen.
            - FA: función de aptitud.
            - continuo: Rangos de los valores
                        continuos contenidos en el genoma.
        
        Implementación de algoritmos genéticos simples, los operadores
        usados son:
            - INICIALIZAR POBLACION.
            - SELECCION.
            - CRUZE.
            - MUTACIÓN.
    '''

    # Conteo de id's de los individuos.
    conteo_id = conteo()

    # Longitud del genoma.
    long_genoma = sum(SG)

    # Generamos la polbación inical.
    poblacion = inicializar_poblacion(Pi, long_genoma, conteo_id)

    # Aptitudes de la población inicial.
    aptitudes = aptitud_poblacion(poblacion, SG, FA, continuo)

    # Calculamos la aptitud promedio de la población inicial.
    aptitud_prom = sum(aptitudes) / len(aptitudes)

    # Recuperamos la optimicidad del mejor individuo.
    optimo = aptitudes[aptitudes.index[0]]
    
    # Retornamos El primer individuo optimo.
    yield (
        aptitud_prom,
        optimo,
        len(poblacion),
        decodificar_genoma(
            poblacion.loc[aptitudes.index[0]],
            SG,
            continuo
        )
    )

    i = 0
    while i < MAXi and len(poblacion) > 1:
        # El primer operador que se usa es la seleccion de las
        # parejas.
        parejas = seleccion(aptitudes)

        # El operador de cruce genera la nueva pobalcion.
        nueva_poblacion = cruce(
            parejas,
            poblacion,
            SG,
            conteo_id,
        )

        # Combinamos las dos poblaciones.
        poblacion = combinar_poblaciones(
            Pi,
            [poblacion, nueva_poblacion],
            SG,
            FA,
            continuo
        )

        # El operador de mutacion determina si un genoma muto.
        poblacion = mutacion(
            poblacion,
            PM if type(PM) is float else PM(i),
            PMg if type(PMg) is float else PMg(i),
        )
        
        # Se calculan las aptitudes de la poblacion.
        aptitudes = aptitud_poblacion(poblacion, SG, FA, continuo)

        # Calculamos la aptitud promedio.
        aptitud_prom = sum(aptitudes) / len(aptitudes)

        # Si la nueva población excede el limite, se acorta al maximo.
        if len(poblacion) > Pi:
            print('Limite de población excedido')

        # Si la aptitud del mejor individuo es mejor que la del ultimo
        # mejor individuo.
        if aptitudes[aptitudes.index[0]] > optimo:
            # Se cambia el optimo por el nuevo individuo optimo.
            optimo = aptitudes[aptitudes.index[0]]

        # Retornamos la aptitud promedio, el mejor individuo y su genoma.
        yield (
            aptitud_prom,
            aptitudes[aptitudes.index[0]],
            len(poblacion),
            decodificar_genoma(
                poblacion.loc[aptitudes.index[0]],
                SG,
                continuo
            )
        )

        # Aumentamos el conteo de la iteración actual.
        i += 1