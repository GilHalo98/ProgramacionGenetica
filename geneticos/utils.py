# Librerias de terceros.
import numpy as np
import pandas as pd


def discreto_a_continuo(
    z: int,
    l: int,
    r_min: float,
    r_max: float,
) -> float:
    '''
        Retorna un mapeo de un valor discreto a un valor continuo.
    '''
    return (
        (r_max - r_min) / (2**l - 1)
    ) * (z + r_min)


def conteo(id_inicio: int = 0):
    '''
        Generador de id de individuos.
    '''
    i = id_inicio
    while True:
        i += 1
        yield i


def decodificar_genoma(
    genoma: pd.Series,
    SG: 'list | tuple',
    continuo: 'list | tuple | None' = None
) -> tuple:
    '''
        Decodifica el genoma, si el genoma es multiparametrico,
        decodifica por secciones el genoma.

        PARAMS:
            - genoma: genoma a decodificar.
            - SG: secciones del genoma.
            - continuo: Rangos de los valores
                        continuos contenidos en el genoma.
    '''
    # Decodifica un genoma dado.
    deco = []

    # Valor de la seccion del genoma.
    seccion_deco = 0

    # El valor de n en 2^n de la decodificacion
    # de la seccion.
    bin_deco = SG[0] - 1

    # Secciones del genoma.
    secciones = len(SG)

    i = 0
    j = 0
    for gen in genoma:
        # Por cada gen del genoma, si esta activo.
        if gen == 1:
            # Se agrega a la decodificacion de la seccion.
            seccion_deco += 2 ** bin_deco

        i += 1
        bin_deco -= 1

        # Si se alcanzo el limite de la seccion del genoma
        # entonces se reinicia la decodificacion de la seccion.
        if i > SG[j] - 1:
            # Si es un genoma continuo.
            if continuo != None:
                seccion_deco = discreto_a_continuo(
                    seccion_deco,
                    SG[j],
                    *continuo[j],
                )
                
            # Si es un genoma discreto unicamente agrega
            # el valor decodificado.
            deco.append(seccion_deco)

            i = 0
            j += 1
            seccion_deco = 0

            if j >= secciones:
                break

            bin_deco = SG[j] - 1

    return deco


def aptitud_poblacion(
    poblacion: pd.DataFrame,
    SG: 'list | tuple',
    fun_apt: 'function',
    continuo: 'list | tuple | None' = None
) -> pd.Series:
    '''
        Calcula las aptitudes de la poblaci??n.

        PARAMS:
            - pobalcion: poblacion a la que se calculara la aptitud.
            - SG: secciones del genoma.
            - fun_apt: funci??n de aptitud de la poblaci??n.
            - continuo: Rangos de los valores
                        continuos contenidos en el genoma.
    '''

    # Calculamos las aptitudes de la poblaci??n.
    aptitudes = {}
    for id in poblacion.index:
        # Se recupera el genoma del individuo.
        individuo = poblacion.loc[id]

        # Calculamos la aptitud del individuo.
        aptitudes[id] = fun_apt(
            individuo,
            decodificar_genoma(individuo, SG, continuo)
        )

    # Se genera un dataframe con las aptitudes de la poblaci??n.
    aptitudes = pd.Series(aptitudes)

    # Ordenamos las aptitudes de manera descendente.
    aptitudes.sort_values(ascending=False, inplace=True)

    return aptitudes


def combinar_poblaciones(
    maximo_poblacion: int,
    poblaciones: 'list[pd.DataFrame]',
    SG: 'list | tuple',
    fun_apt: 'function',
    continuo: 'list | tuple | None' = None
) -> pd.DataFrame:
    '''
        Combina las poblaciones, la antigua y la nueva y filtra la
        poblaci??n con menor aptitud.

        PARAMS:
            - maximo_poblacion: Maximo de poblacion posible.
            - poblaciones: lista de poblaciones a concatenar.
            - pobalcion: poblacion a la que se calculara la aptitud.
            - SG: secciones del genoma.
            - fun_apt: funci??n de aptitud de la poblaci??n.
            - continuo: Rangos de los valores
                        continuos contenidos en el genoma.
    '''

    # Concatenamos las poblaciones.
    poblacion_combinada = pd.concat(poblaciones)

    # Calculamos las aptitudes de la poblacion combinada.
    aptitudes = aptitud_poblacion(
        poblacion_combinada,
        SG,
        fun_apt, 
        continuo
    )

    # Consultamos los individuos con menor aptitud.
    poblacion_eliminada = aptitudes.index[maximo_poblacion:]

    # Eliminamos la poblacion con menor aptitud.
    poblacion_combinada.drop(poblacion_eliminada, inplace=True)

    return poblacion_combinada