'''
'''

# !/usr/bin/env python3
# -*- coding: UTF-8 -*-

__author__ = "Diego Gil"


# Librerias estandar.
import os
import platform
import time

# Librerias de terceros
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Librerias propias.
from geneticos import ag
from geneticos import utils


# Funcion de aptitud de los individuos.
def fun_aptitud(
    genoma: pd.Series,
    genoma_decodificado: 'list | tuple'
) -> float:
    # El algoritmo genetico pasa el genoma y el genoma decodificado
    # a la funcion de aptitud, de esta forma se puede calcular una
    # funcion de aptitud personalizada.
    x_1 = genoma_decodificado[0]
    x_2 = genoma_decodificado[1]

    restriccion_1 = (6 * x_1 + 4 * x_2) <= 24
    restriccion_2 = (x_1 + 2 * x_2) <= 6
    restriccion_3 = (-x_1 + x_2) <= 1
    restriccion_4 = x_2 <= 2
    restriccion_no_negatividad = (x_1 >= 0) and (x_2 >= 0)

    aptitud = 5 * x_1 + 4 * x_2

    if not (
        restriccion_1
        and restriccion_2
        and restriccion_3
        and restriccion_4
        and restriccion_no_negatividad
    ):
        aptitud = 0

    else:
        # Minimizar la aptitud.
        # aptitud = 1 / aptitud

        # Maximizar la aptitud.
        aptitud = aptitud

    return aptitud


# Funcion main.
def main(*args, **kargs) -> None:
    secciones_genoma: 'list[int]' = [32, 32]
    secciones_continuas: 'list[int]' = [(0, 5), (0, 5)]

    # Instanciamos el algoritmo genetico.
    algo_gen = ag.AG(
        100,
        secciones_genoma,
        100,
        0.5,  # lambda i : 1 / np.exp(i - 50 / 100) if i > 50 else 0.8,
        0.2,
        fun_aptitud,
        continuo=secciones_continuas
    )

    historico = {}

    mejor = {
        'z': 0,
        'x': None,
        'i': -1,
    }

    a = time.perf_counter()
    i = 0
    for aptitud_prom, optimo, config in algo_gen:
        historico[i] = {
            'aptitud_prom': aptitud_prom,
            'optimo': optimo,
            'X': config,
        }
        if optimo > mejor['z']:
            mejor['z'] = optimo
            mejor['x'] = config
            mejor['i'] = i
        i += 1

    b = time.perf_counter()

    historico = pd.DataFrame(historico)
    print(historico)
    print(mejor)
    print('--> Tiempo de ejecucion: {0:.2f}s'.format(b - a))

    plt.plot(
        [i for i in range(len(historico.loc['optimo']))],
        historico.loc['optimo'],
        'o-',
        label='Optimo'
    )
    plt.plot(
        [i for i in range(len(historico.loc['aptitud_prom']))],
        historico.loc['aptitud_prom'],
        'o-',
        label='Aptitud Global'
    )
    plt.grid()
    plt.legend()
    plt.show()


# Ejecutamos la funcion main si se llama desde el programa main.
if __name__ == '__main__':
    os.system(
        'cls' if platform.system() == 'Windows' else 'clear'
    )
    main()
