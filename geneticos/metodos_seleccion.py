# Librerias de terceros.
import numpy as np
import pandas as pd


def generar_ruleta(contendientes: pd.Series) -> pd.Series:
    '''
        Método de la ruleta para la selección de las parejas.

        PARAMS:
        - contendientes: lista de contendientes.
    '''

    # Consultamos la aptitud minima de los contendientes.
    minimo = contendientes.min()

    # Cambiamos los individuos con aptitud de 0 a la aptitud minima.
    contendientes.replace({
        0: minimo if minimo > 0 else 1
    }, inplace=True)

    # Calculamos la aptitud de la población.
    aptitud_poblacion = sum(contendientes)

    # Calculamos el area de la ruleta correspondiente a cada individuo.
    areas_ruleta: pd.Series = contendientes / aptitud_poblacion

    # Generamos la ruleta con los individuos validos
    ruleta = areas_ruleta.cumsum()

    return ruleta