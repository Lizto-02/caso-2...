"""
app.py
-------
Streamlit app — Caso 2: Ruteo de un vehículo de distribución (TSP)

Resuelve el TSP exacto sobre la matriz Desde-Hasta (13 puntos: depósito + 12
clientes) usando un modelo MILP (PuLP/CBC) equivalente al formulado en AMPL,
y responde las preguntas de las Partes I, II y III del caso.
"""

import pandas as pd
import streamlit as st

from model import DIST_MATRIX, LABELS, N, model_size, solve_tsp

st.set_page_config(page_title="Caso 2 · TSP Ruteo de distribución", layout="wide")

st.title("🚚 Caso 2 · Ruteo de un vehículo de distribución (TSP)")
st.caption(
    "Depósito (nodo 0) + 12 clientes · matriz de distancias ya reducida · "
    "modelo exacto de Programación Lineal Entera (equivalente al formulado en AMPL)"
)

tab_matriz, tab_modelo, tab_resultado = st.tabs(
    ["📋 Matriz de distancias", "🧮 Formulación del modelo", "✅ Resolver TSP"]
)

# ---------------------------------------------------------------------------
# Tab 1: Matriz de distancias
# ---------------------------------------------------------------------------
with tab_matriz:
    st.subheader("Matriz Desde–Hasta (13×13)")
    df = pd.DataFrame(DIST_MATRIX, index=LABELS, columns=LABELS)
    st.dataframe(df, use_container_width=True)
    st.caption("Distancias en km · d(i,j) = d(j,i) · diagonal = 0 · fila/columna 0 = depósito.")

# ---------------------------------------------------------------------------
# Tab 2: Formulación del modelo
# ---------------------------------------------------------------------------
with tab_modelo:
    st.subheader("Formulación matemática (TSP exacto)")
    st.markdown(
        r"""
**Conjuntos**
- $N = \{0,1,\dots,n-1\}$: puntos relevantes (0 = depósito), $n=13$.

**Parámetros**
- $d_{ij}$: distancia entre $i$ y $j$ (matriz Desde-Hasta).

**Variables**
- $x_{ij} \in \{0,1\}$: 1 si la ruta va directamente de $i$ a $j$, $\; i \neq j$.

**Función objetivo**
$$\min \sum_{i \neq j} d_{ij}\, x_{ij}$$

**Restricciones**
1. Salida única: $\displaystyle\sum_{j \neq i} x_{ij} = 1 \quad \forall i \in N$
2. Entrada única: $\displaystyle\sum_{i \neq j} x_{ij} = 1 \quad \forall j \in N$
3. Eliminación de subciclos (DFJ, agregadas de forma perezosa):
$$\sum_{i,j \in S} x_{ij} \le |S| - 1 \qquad \forall\, S \subset N,\; 2 \le |S| \le n-1$$
        """
    )
    variables, basicas, subciclos_max = model_size(N)
    c1, c2, c3 = st.columns(3)
    c1.metric("Variables binarias", variables)
    c2.metric("Restricciones básicas", basicas)
    c3.metric("Restricciones de subciclo (cota máx.)", f"{subciclos_max:,}")
    st.caption(
        "Las restricciones de subciclo no se generan todas de entrada (crecen "
        "exponencialmente); el solver las agrega solo cuando aparece una subruta, "
        "lo cual basta para llegar al óptimo en pocas iteraciones."
    )

# ---------------------------------------------------------------------------
# Tab 3: Resolver
# ---------------------------------------------------------------------------
with tab_resultado:
    st.subheader("Ejecutar el modelo")
    if st.button("▶️ Resolver TSP", type="primary"):
        with st.spinner("Resolviendo modelo MILP (CBC)..."):
            resultado = solve_tsp()
        st.session_state["resultado"] = resultado

    if "resultado" in st.session_state:
        r = st.session_state["resultado"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Distancia total óptima", f"{r['distancia_total']:.0f} km")
        c2.metric("Status", r["status"])
        c3.metric("Iteraciones (subciclos)", r["iteraciones"])

        st.markdown("**Secuencia de visita (ruta óptima):**")
        st.success(" → ".join(map(str, r["ruta"])))

        st.markdown("**Arcos seleccionados:**")
        arcos_df = pd.DataFrame(r["arcos"], columns=["Desde", "Hasta"])
        arcos_df["Distancia (km)"] = [
            DIST_MATRIX[LABELS.index(i)][LABELS.index(j)] for i, j in r["arcos"]
        ]
        st.dataframe(arcos_df, use_container_width=True, hide_index=True)
    else:
        st.info("Presione el botón para correr el solver y obtener la ruta óptima.")


st.divider()
st.caption("Modelo TSP exacto · PuLP + CBC (sin licencia, equivalente al formulado en AMPL) · Caso 2")
