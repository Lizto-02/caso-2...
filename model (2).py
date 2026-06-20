"""
model.py
---------
Modelo del Problema del Agente Viajero (TSP) para el Caso 2.

Formulación (equivalente a la usada en AMPL):

    Conjuntos:
        N = {0,...,n-1}            puntos relevantes (0 = depósito)
        A = {(i,j) : i != j}       arcos posibles

    Parámetros:
        d[i,j]                     distancia entre i y j (matriz Desde-Hasta)

    Variables:
        x[i,j] in {0,1}            1 si la ruta va directo de i a j

    Función objetivo:
        min  sum_{(i,j) in A} d[i,j] * x[i,j]

    Restricciones:
        (1) Salida única:   sum_j x[i,j] = 1   para todo i en N
        (2) Entrada única:  sum_i x[i,j] = 1   para todo j en N
        (3) Eliminación de subciclos (DFJ, agregadas de forma perezosa):
            sum_{i,j in S} x[i,j] <= |S| - 1     para todo S subconjunto propio de N, 2<=|S|<=n-1

El solver usa PuLP + CBC (gratuito, sin licencia) y agrega las restricciones (3)
de manera iterativa (lazy constraints): resuelve, detecta sub-rutas (subtours)
desconectadas del depósito y les agrega su restricción, hasta obtener un único
ciclo que pasa por todos los nodos (la ruta óptima).
"""

import itertools
import pulp

# ---------------------------------------------------------------------------
# 1. Datos: los 13 puntos relevantes y la matriz Desde-Hasta (ya reducida)
# ---------------------------------------------------------------------------

LABELS = [0, 3, 10, 21, 22, 35, 40, 41, 47, 65, 71, 76, 77]  # 0 = depósito

DIST_MATRIX = [
    [0, 29, 18, 16, 24, 32, 17, 28, 29, 27, 27, 33, 34],
    [29, 0, 38, 40, 29, 48, 15, 57, 27, 53, 55, 24, 44],
    [18, 38, 0, 31, 42, 49, 22, 27, 21, 19, 36, 50, 16],
    [16, 40, 31, 0, 21, 18, 32, 25, 45, 30, 16, 34, 47],
    [24, 29, 42, 21, 0, 20, 30, 46, 46, 48, 36, 14, 56],
    [32, 48, 49, 18, 20, 0, 45, 40, 60, 47, 25, 33, 65],
    [17, 15, 22, 32, 30, 45, 0, 44, 16, 39, 45, 32, 30],
    [28, 57, 27, 25, 46, 40, 44, 0, 48, 11, 17, 59, 41],
    [29, 27, 21, 45, 46, 60, 16, 48, 0, 40, 54, 48, 18],
    [27, 53, 19, 30, 48, 47, 39, 11, 40, 0, 26, 60, 30],
    [27, 55, 36, 16, 36, 25, 45, 17, 54, 26, 0, 50, 52],
    [33, 24, 50, 34, 14, 33, 32, 59, 48, 60, 50, 0, 62],
    [34, 44, 16, 47, 56, 65, 30, 41, 18, 30, 52, 62, 0],
]

N = len(LABELS)  # 13


def model_size(n: int):
    """Devuelve (num_variables, num_restricciones_basicas) para n puntos.

    num_variables            = n*(n-1)                  (arcos dirigidos i != j)
    restricciones basicas    = 2n                        (salida unica + entrada unica)
    (las de eliminacion de subciclos se agregan de forma perezosa, no se cuentan
     todas porque crecen exponencialmente: hasta 2^n - n - 2)
    """
    variables = n * (n - 1)
    basicas = 2 * n
    subciclos_max = 2 ** n - n - 2
    return variables, basicas, subciclos_max


# ---------------------------------------------------------------------------
# 2. Solver MILP con eliminación de subciclos perezosa (estilo DFJ)
# ---------------------------------------------------------------------------

def _find_subtours(x_vals, n):
    """A partir de los valores de x[i,j], devuelve la lista de sub-rutas (cada
    una como lista de índices). Si hay una sola ruta de longitud n, es la
    solución final (no hay subtours)."""
    succ = {}
    for i in range(n):
        for j in range(n):
            if i != j and x_vals[i][j] > 0.5:
                succ[i] = j

    visited = set()
    subtours = []
    for start in range(n):
        if start in visited:
            continue
        tour = []
        cur = start
        while cur not in visited:
            visited.add(cur)
            tour.append(cur)
            cur = succ[cur]
        subtours.append(tour)
    return subtours


def solve_tsp(dist_matrix=None, labels=None, max_iter=200, msg=False):
    """Resuelve el TSP exacto (mínima distancia) sobre la matriz dada.

    Retorna un diccionario con:
        distancia_total, ruta (lista de etiquetas), arcos (lista de tuplas),
        variables, restricciones_basicas, restricciones_subciclo_agregadas,
        status
    """
    if dist_matrix is None:
        dist_matrix = DIST_MATRIX
    if labels is None:
        labels = LABELS

    n = len(labels)
    nodes = range(n)

    prob = pulp.LpProblem("TSP", pulp.LpMinimize)

    x = {
        (i, j): pulp.LpVariable(f"x_{i}_{j}", cat="Binary")
        for i in nodes for j in nodes if i != j
    }

    # Objetivo
    prob += pulp.lpSum(dist_matrix[i][j] * x[i, j] for i in nodes for j in nodes if i != j)

    # (1) Salida única
    for i in nodes:
        prob += pulp.lpSum(x[i, j] for j in nodes if j != i) == 1, f"salida_{i}"

    # (2) Entrada única
    for j in nodes:
        prob += pulp.lpSum(x[i, j] for i in nodes if i != j) == 1, f"entrada_{j}"

    subciclo_count = 0
    for iteration in range(max_iter):
        prob.solve(pulp.PULP_CBC_CMD(msg=msg))

        x_vals = [[pulp.value(x[i, j]) if i != j else 0 for j in nodes] for i in nodes]
        subtours = _find_subtours(x_vals, n)

        if len(subtours) == 1:
            # Solución factible: un solo ciclo que visita todos los nodos
            order = subtours[0]
            # Asegurar que la ruta inicie en el depósito (índice 0)
            start_idx = order.index(0)
            order = order[start_idx:] + order[:start_idx]
            ruta_labels = [labels[k] for k in order] + [labels[order[0]]]
            arcos = [(labels[order[k]], labels[order[k + 1]]) for k in range(len(order) - 1)]
            return {
                "distancia_total": pulp.value(prob.objective),
                "ruta": ruta_labels,
                "arcos": arcos,
                "variables": len(x),
                "restricciones_basicas": 2 * n,
                "restricciones_subciclo_agregadas": subciclo_count,
                "status": pulp.LpStatus[prob.status],
                "iteraciones": iteration + 1,
            }

        # Agregar una restricción de eliminación de subciclo por cada subtour
        # estricto (DFJ): sum_{i,j in S} x[i,j] <= |S| - 1
        for tour in subtours:
            if len(tour) < n:
                subciclo_count += 1
                prob += (
                    pulp.lpSum(x[i, j] for i in tour for j in tour if i != j)
                    <= len(tour) - 1
                ), f"subciclo_{subciclo_count}"

    raise RuntimeError("No se alcanzó una solución libre de subciclos en max_iter iteraciones")


if __name__ == "__main__":
    resultado = solve_tsp(msg=False)
    print("Distancia total:", resultado["distancia_total"])
    print("Ruta:", " -> ".join(map(str, resultado["ruta"])))
    print("Variables:", resultado["variables"])
    print("Restricciones básicas:", resultado["restricciones_basicas"])
    print("Restricciones de subciclo agregadas:", resultado["restricciones_subciclo_agregadas"])
