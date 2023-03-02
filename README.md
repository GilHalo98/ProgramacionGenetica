### ¿Qué son los Algoritmos Genéticos?
Los algoritmos genéticos son algoritmos de optimización inspirados en la selección natural y la genética, una de las ventajas con los algoritmos genéticos es que pueden ser implementados en una gran cantidad de problemas de optimización.

El algoritmo es simple de entender y de implementar, lo cual facilita el mantenimiento de este, los algoritmos genéticos tienen la siguiente estructura base:
1. Una población con soluciones iniciales.
2. Una medida para medir la optimización de la solución.
3. Una forma de mezclar los genes de la población.
4. Una forma de mutar dichos genes.

Esta serie de requisitos es necesaria para que el algoritmo genético pueda encontrar las soluciones optimas a los problemas dados, una de las ventajas de los algoritmos genéticos es que pueden ser usados en espacios de búsqueda complejos, así como tener múltiples soluciones iniciales, evitando así el problema del óptimo local.

Los Algoritmos genéticos hacen uso de un conjunto de operadores genéticos, los cuales describen funciones base de la metodología, los tres más importantes son los siguientes:
1. Selección: La selección Busca simular la selección natural, una competencia entre la población para saber cuales individuos son los más aptos para pasar los genes a la siguiente generación.
2. Cruce: Simula la reproducción sexual, intercambiando la información genética a la siguiente población.
3. Mutación: Simula la mutación de los genes, esto para escapar de posibles óptimos locales.

Como se puede inducir hasta este punto, el espacio de búsqueda de los algoritmos genéticos son los genes del individuo, estos genes son vectores de bits que representan una posible solución para el problema dado, la longitud de los genes es proporcional a la posible solución del problema.

### Pseudo-Código
El pseudo-código de un algoritmo genético únicamente toma en cuenta la integración de los operadores genéticos entre si.

```
DEFINE SELECCION:
    PARAMS:
        - P: Población en la que se realizara el proceso de selección.
        - FA: Función de Aptitud.
        - DECO: Funcion de decodificación del genoma.
        - LS: Longitudes de las secciones del genoma.

    LET decodificaciones <- DECO(P, LS)
    LET aptitudes <- FA(decodificaciones)
    LET ruleta <- aptitudes / SUM(aptitudes)

    LET parejas <- EMPTY LIST
    FOR EACH individuo IN P DO
        LET pareja <- WEIGHT_SELECT(ruleta)
        parejas.push(
            individuo,
            pareja
        )

    RETURN parejas

DEFINE CRUCE:
    PARAMS:
        - P: Población en la que se realizara el proceso de selección.
        - Pa: Parejas de los cruces.
        - LS: Longitudes de las secciones del genoma.

    LET nueva_pobacion <- NEW POBLACION()

    FOR EACH pareja in Pa DO
        LET pa_1, pa_2 <- pareja

        LET hijo_1 <- NEW INDIVIDUO()
        LET hijo_2 <- NEW INDIVIDUO()

        nueva_seccion_1, nueva_seccion_2 = CRUZAR_GENOMA(
            pa_1.genoma,
            pa_2.genoma, 
            LS
        )
        hijo_1.add_to_genoma(nueva_seccion_1)
        hijo_2.add_to_genoma(nueva_seccion_2)

        nueva_poblacion.add(hijo_1)
        nueva_poblacion.add(hijo_2)

    RETURN nueva_poblacion

DEFINE CRUZAR_GENOMA:
    PARAMS:
        - g_1: Genoma 1
        - g_2: Genoma 2
        - LS: Longitudes de las secciones del genoma.
        
    LET nuevo_g_1 = GENOMA()
    LET nuevo_g_2 = GENOMA()

    LET i = 0
    FOR long_seccion IN LS DO
        LET punto_medio = i + ROUND(long_seccion / 2)
        nuevo_g_1.add(
            g_1[i:punto_medio],
            g_2[punto_medio:i+long_seccion]
        )
        nuevo_g_2.add(
            g_2[i:punto_medio],
            g_1[punto_medio:i+long_seccion]
        )
        i += long_seccion

    RETURN nuevo_g_1, nuevo_g_2

DEFINE MUTACION:
    PARAMS:
        - P: Población en la que se realizara el proceso de mutacion.
        - PM: Probabilidad de mutación del individuo.
        - PMg: Probabilidad de mutación de cada gen.

    FOR EACH individuo IN P DO
        IF RANDOM() <= PM DO
            FOR EACH gen IN individuo.genoma DO
                IF RANDOM() <= PMg DO
                    gen.flip()

DEFINE AG:
    PARAMS:
        - Pi: Población inicial de algoritmo genetico.
        - LS: Longitudes de las secciones del genoma.
        - FA: Función de aptitud.
        - PM: Probabilidad de mutación del individuo.
        - PMg: Probabilidad de mutación de cada gen.
        - MAXi: Maxima cantidad de iteraciones.
        - DECO: Función de decodificación del genoma.

    LET longitud_genoma = SUM(LS)

    LET poblacion <- POBLACION_INICIAL(
        Pi,
        longitud_genoma
    )

    LET i = 0
    while i < MAXi DO
        LET parejas <- SELECCION(
            poblacion,
            FA,
            DECO,
            LS
        )

        LET nueva_poblacion <- CRUCE(
            poblacion,
            parejas,
            LS,
        )

        MUTACION(
            nueva_poblacion,
            PM,
            PMg,
        )

        poblacion = nueva_pobacion
        i ++
```
