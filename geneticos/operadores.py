# Librerias de terceros.
import numpy as np
import pandas as pd

# Propios.
from .metodos_seleccion import generar_ruleta


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

    # Realizamos un split del 50/50, una parte de la población
    # la más apta sera parte de la ruleta, la otra sera parte de los
    # individuos que giraran la ruleta
    contendientes = aptitudes[:int(len(aptitudes)*.5)]
    individuos = aptitudes[int(len(aptitudes)*.5):]

    # Generamos una ruleta para la selección de las parejas.
    ruleta = generar_ruleta(contendientes)

    # Total de contendientes en la ruleta.
    total_contendientes = len(ruleta.index)

    # Por cada individuo en la población
    for individuo in individuos.index:
        # Generamos un numero en el intervalo de [0, 1)
        giro = np.random.random()

        # intervalo inferior es en 0
        int_inf = 0

        # Verificamos que pareja le corresponde.
        for i in range(total_contendientes):
            # Contendiente actual.
            contendiente = ruleta.index[i]

            # El intervalo superior pasa a ser el area de la ruleta
            # del contendiente actual.
            int_sup = ruleta[contendiente]

            # Si el giro se encuentra en el intervalo [inf, sup)
            if giro >= int_inf and giro < int_sup:
                break

            # Si no se encuentra en el intervalo, avanzamos
            # al siguiente intervalo.
            int_inf += int_sup

        # Agregamos la pareja.
        parejas.append((individuo, contendiente))

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
        nuevo_individuo_a = np.array([], dtype=np.integer)
        nuevo_individuo_b = np.array([], dtype=np.integer)

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