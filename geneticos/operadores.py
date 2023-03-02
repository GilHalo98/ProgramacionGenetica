# Librerias de terceros.
import numpy as np
import pandas as pd


def inicializar_poblacion(
    N: int,
    M: int,
    asignar_id
) -> pd.DataFrame:
    '''
        Operador genético, crea una población inicial con
        estados aleatorios.

        PARAMS:
            - N: cantidad de individuos en la poblacion.
            - M: longitud del genoma.
            - asignar_id: generador de id's
    '''

    # Crea una poblacion inicial de N individuos con M genes.
    poblacion = {}

    # Por cada gen del individuo.
    for i in range(M):
        poblacion[i] = []

        # Por cada individuo en la poblacion.
        for j in range(N):
            # Asigna el gen aleatoriamente.
            poblacion[i].append(np.random.randint(0, 2))

    return pd.DataFrame(
        poblacion,
        index=['ID_{}'.format(asignar_id.__next__()) for _ in range(N)]
    )


def seleccion(
    aptitudes: pd.Series
) -> list:
    '''
        Operador de selección de parejas, se usa la estrategia
        de la ruleta.

        PARAMS:
            - aptitudes: lista de aptitudes de la población.
    '''

    # Operador genetico de selccion de parejas.
    parejas = []
    
    # Ahora calculamos la probabilidad de la ruleta por individuo.
    aptitud_pobalcion = sum(aptitudes)

    # Si la aptitud de la pobalcion es distinta de 0
    # entonces se calculan las probabilidades de cruce.
    probabilidades = aptitudes / aptitud_pobalcion

    # Si la aptitud de la poblacion es 0, etonces las probabilidades
    # sin equitativas.
    if aptitud_pobalcion == 0:
        probabilidades = None

    else:
        # Eliminamos el problema de que exista un unico 
        if probabilidades[0] == 1:
            probabilidades *= 0
            probabilidades += 0.1
            probabilidades /= len(aptitudes) - 1
            probabilidades[0] = 0.9

    # Calculamos cuantas parejas se generaran.
    total_parejas = int(len(aptitudes.index)) / 2

    # Lista de id de los individuos.
    id_individuos = aptitudes.index
    
    # Por último generamos las parejas con la ruleta.
    while total_parejas > 0:
        parejas.append(np.random.choice(
            id_individuos,
            2,
            replace=False,
            p=probabilidades
        ))

        total_parejas -= 1

    return parejas


def cruce(
    parejas: list,
    poblacion: pd.DataFrame,
    SG: 'list | tuple',
    asignar_id,
) -> pd.DataFrame:
    '''
        Operador de cruce de las parejas.
        PARAMS:
            - parejas: lista de parejas
            - poblacion: población a cruzar.
            - SG: secciones del genoma.
            - asignar_id: generaador de id's.
    '''

    # Operador genetico de cruce.
    nueva_poblacion = {}
    
    # Por cada pareja en la lista de parejas.
    for id_a, id_b in parejas:
        # Se consultan los genes de los individuos.
        individuo_a = poblacion.loc[id_a].to_numpy()
        individuo_b = poblacion.loc[id_b].to_numpy()

        # index inferior del genoma.
        i_inf = 0

        # index superior del genoma.
        j = 0
        secciones = len(SG)
        i_sup = SG[j] - 1

        # Longitud del genoma.
        long_genoma = len(individuo_b)

        # Instancias de los nuevos individuos.
        nuevo_individuo_a = np.array([])
        nuevo_individuo_b = np.array([])

        # Iteramos entre cada seccion del genoma.
        while i_sup < long_genoma:
            # Instanciamos las sub-secciones de los genomas.
            sub_seccion_a = individuo_a[i_inf:i_sup + 1]
            sub_seccion_b = individuo_b[i_inf:i_sup + 1]

            # Se calcula el index del punto medio del genoma.
            punto_medio = int(len(sub_seccion_a) / 2)

            # Creamos las nuevas sub_secciones.
            nueva_sub_seccion_a = np.concatenate(
                (sub_seccion_a[:punto_medio], sub_seccion_b[punto_medio:]),
                axis=0
            )

            nueva_sub_seccion_b = np.concatenate(
                (sub_seccion_b[:punto_medio], sub_seccion_a[punto_medio:]),
                axis=0
            )

            # Contatenamos las nuevas sub secciones a su respectivo
            # nuevo individuo.
            nuevo_individuo_a = np.concatenate(
                (nueva_sub_seccion_a, nuevo_individuo_a),
                axis=0
            )
            
            nuevo_individuo_b = np.concatenate(
                (nueva_sub_seccion_b, nuevo_individuo_b),
                axis=0
            )

            j += 1
            i_inf = i_sup + 1

            if j >= secciones:
                break

            i_sup += SG[j]

        nueva_poblacion[
            'ID_{}'.format(asignar_id.__next__())
        ] = nuevo_individuo_a

        nueva_poblacion[
            'ID_{}'.format(asignar_id.__next__())
        ] = nuevo_individuo_b

    return pd.DataFrame(
        nueva_poblacion.values(),
        index=nueva_poblacion.keys()
    )


def mutacion(
    nueva_poblacion: pd.DataFrame,
    probabilidad_mutacion: float,
    probabilidad_mutacion_gen: float
):
    '''
        Operador de mutación de los genomas.

        PARAMS:
            - nueva_poblacion: publacion a mutar.
            - probabilidad_mutacion: probabilidad de mutacion
                del individuo
            - probabilidad_mutacion_gen: probabilidad de mutacion
                del gen
    '''

    # Por cada individuo en la nueva generacion.
    for id_individuo in nueva_poblacion.index:
        # Calculamos si el individuo
        # es candidato a recibir una mutación.
        if np.random.random() <= probabilidad_mutacion:
            individuo = nueva_poblacion.loc[id_individuo]

            # Una vez que el individuo es apto para recibir
            # una mutación checamos cuales son los genes que mutaran.
            for i in individuo.index:
                # Calculamos si el gen tendra una mutación.
                if np.random.random() <= probabilidad_mutacion_gen:
                    # En este caso, en la mutación, el
                    # gen se cambia a su valor negado.
                    individuo[i] = int(not individuo[i])

    return nueva_poblacion

