import streamlit as st
import plotly.graph_objects as go
import sqlite3
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import ast
import json
# =========================================================
# CONFIG INICIAL
# =========================================================
st.set_page_config(page_title="TopoSwitch – Plataforma Interativa", layout="wide")

st.title("⚡ Plataforma Interativa – TopoSwitch (IEEE-123 Bus)")

BASE_DIR = Path(__file__).parent

DB_MODO1 = BASE_DIR / "ieee123_isolamento.db"
DB_MODO2 = BASE_DIR / "ieee123_duasfontes.db"
# =========================================================
# FUNÇÕES DE ACESSO A BANCO
# =========================================================
def connect(db_path):
    return sqlite3.connect(db_path)

@st.cache_data(show_spinner=False)
def load_table(db_path, query, many=True):
    conn = connect(db_path)
    cur = conn.cursor()
    cur.execute(query)
    rows = cur.fetchall()
    conn.close()
    return rows if many else rows[0]

@st.cache_data(show_spinner=False)
def load_coords(db_path):
    rows = load_table(db_path, "SELECT bus, x, y FROM coords")
    return {str(b): (float(x), float(y)) for b, x, y in rows}

@st.cache_data(show_spinner=False)
def load_topology(db_path):
    rows = load_table(
        db_path,
        "SELECT line, from_bus, to_bus, is_switch, norm FROM topology"
    )
    topo = []
    for line, f, t, sw, norm in rows:
        topo.append(dict(
            line=str(line),
            from_bus=str(f),
            to_bus=str(t),
            is_switch=bool(sw),
            norm=str(norm),
        ))
    return topo


@st.cache_data(show_spinner=False)
def load_loads(db_path):
    try:
        rows = load_table(db_path, "SELECT bus, kw FROM loads")
        return {str(b): float(kw) for b, kw in rows}
    except:
        return {}

# =========================================================
# FUNÇÕES DO MODO 1 – Isolamento Real (uma Fonte)
# =========================================================

@st.cache_data(show_spinner=False)
def load_vao_map(db_path):
    rows = load_table(db_path, "SELECT u_bus, v_bus, nf, kw, n_barras FROM vao_map")
    dados = []
    for u, v, nf, kw, n in rows:
        dados.append(dict(
            u=str(u),
            v=str(v),
            nf=str(nf),
            kw=float(kw),
            n_barras=int(n)
        ))
    return dados

@st.cache_data(show_spinner=False)
def load_nf_map(db_path):
    rows = load_table(db_path, "SELECT nf, barras_isoladas, kw, n_barras FROM nf_map")
    nf_dict = {}

    for nf, barras_str, kw, n in rows:
        try:
            barras = set(ast.literal_eval(barras_str))
        except:
            barras = {x.strip() for x in barras_str.split(",") if x.strip()}

        nf_dict[str(nf)] = dict(
            barras=barras,
            kw=float(kw),
            n_barras=int(n),
        )
    return nf_dict

def identificar_vaos(seq):
    vals = [b.strip() for b in seq if b.strip()]
    vaos = []
    for i in range(0, len(vals), 2):
        if i + 1 < len(vals):
            vaos.append((vals[i], vals[i + 1]))
    return vaos

def melhor_nf_para_vao(u, v, vao_map):
    ops = [
        linha for linha in vao_map
        if (linha["u"] == u and linha["v"] == v)
        or (linha["u"] == v and linha["v"] == u)
    ]
    if not ops:
        return None
    ops.sort(key=lambda x: (x["kw"], x["n_barras"]))
    return ops[0]

def impacto_consolidado_modo1(lista_nf, loads, nf_map_data):
    total_barras = set()
    for nf in lista_nf:
        if nf in nf_map_data:
            total_barras |= nf_map_data[nf]["barras"]
    kw_total = sum(loads.get(b, 0.0) for b in total_barras)
    return kw_total, len(total_barras), sorted(list(total_barras))

# =========================================================
# FUNÇÕES DO MODO 2 – Duas Fontes (NF–NA–NF)
# =========================================================

@st.cache_data(show_spinner=False)
def load_nfnanf(db_path):
    rows = load_table(
        db_path,
        "SELECT nf1, na, nf_block, kw_off, n_manobras, buses_off FROM nf_na_nf"
    )
    combos = []
    for nf1, na, nfb, kw, nm, bjson in rows:
        try:
            blist = json.loads(bjson)
        except:
            blist = []
        combos.append(dict(
            nf1=str(nf1),
            na=str(na) if na else None,
            nf_block=str(nfb) if nfb else None,
            kw_off=float(kw),
            n_manobras=int(nm),
            buses_off=blist,
        ))
    return combos

# =========================================================
# PLOT – rede
# =========================================================

def plot_network(coords, topo, edges_highlight=None, buses_highlight=None):
    """Plot básico da rede."""
    fig = go.Figure()

    # linhas
    x, y = [], []
    for el in topo:
        u, v = el["from_bus"], el["to_bus"]
        if u not in coords or v not in coords:
            continue
        x += [coords[u][0], coords[v][0], None]
        y += [coords[u][1], coords[v][1], None]

    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines",
        line=dict(color="#ccc", width=1),
        name="Linhas"
    ))

    # nós
    fig.add_trace(go.Scatter(
        x=[coords[b][0] for b in coords],
        y=[coords[b][1] for b in coords],
        text=list(coords.keys()),
        mode="markers+text",
        textposition="top center",
        marker=dict(size=5, color="#1f77b4"),
        name="Barras"
    ))

    # linhas destacadas
    if edges_highlight:
        ex, ey = [], []
        for (u, v) in edges_highlight:
            if u in coords and v in coords:
                ex += [coords[u][0], coords[v][0], None]
                ey += [coords[u][1], coords[v][1], None]
        fig.add_trace(go.Scatter(
            x=ex, y=ey, mode="lines",
            line=dict(color="black", width=4),
            name="Trecho"
        ))

    # barras destacadas
    if buses_highlight:
        fig.add_trace(go.Scatter(
            x=[coords[b][0] for b in buses_highlight if b in coords],
            y=[coords[b][1] for b in buses_highlight if b in coords],
            mode="markers",
            marker=dict(size=10, color="red"),
            name="Barras isoladas"
        ))

    fig.update_layout(
        height=600,
        showlegend=True,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig

# =========================================================
# INTERFACE
# =========================================================

tab1, tab2 = st.tabs(["🔌 MODO 1 – Fonte Única", "⚡ MODO 2 – Duas Fontes"])

# ---------------------------------------------------------
# MODO 1 – Isolamento Real
# ---------------------------------------------------------

with tab1:

    st.header("🔌 Modo 1 – Isolamento Real (Uma Fonte)")

    if not DB_MODO1.exists():
        st.error("Banco ieee123_isolamento.db não encontrado.")
        st.stop()

    coords = load_coords(DB_MODO1)
    topo = load_topology(DB_MODO1)
    vao_map = load_vao_map(DB_MODO1)
    loads = load_loads(DB_MODO1)
    nf_map_data = load_nf_map(DB_MODO1)

    st.subheader("Mapa Base")
    st.plotly_chart(plot_network(coords, topo), use_container_width=True)

    # ---- entrada vão simples ----
    lista_barras = sorted(coords.keys(), key=lambda x: int(x) if x.isdigit() else x)

    col1, col2 = st.columns(2)
    u = col1.selectbox("Barra U", lista_barras)
    v = col2.selectbox("Barra V", lista_barras)

    if st.button("Processar Vão (Modo 1)"):
        info = melhor_nf_para_vao(u, v, vao_map)
        if info is None:
            st.error("Nenhuma NF encontrada para esse vão!")
        else:
            st.success(f"NF ótima: **{info['nf']}** | {info['kw']:.1f} kW desligados")

            # Highlight
            fig = plot_network(
                coords,
                topo,
                edges_highlight=[(u, v)],
                buses_highlight=[],
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ---- trecho multi-vãos ----
    st.subheader("Trecho Multi-Vãos")

    entrada = st.text_input("Sequência (U1,V1,U2,V2,...)", "60,62,63,64")
    if st.button("Processar Trecho (Modo 1)"):

        seq = [x.strip() for x in entrada.split(",")]
        vaos = identificar_vaos(seq)

        infos = []
        for u, v in vaos:
            i = melhor_nf_para_vao(u, v, vao_map)
            if i:
                infos.append(i)

        st.write("NF por vãos:", infos)

        lista_nf = [i["nf"] for i in infos]
        kw_tot, nb, barras = impacto_consolidado_modo1(lista_nf, loads, nf_map_data)

        st.success(f"Impacto consolidado: **{kw_tot:.1f} kW**, {nb} barras desligadas")

# ---------------------------------------------------------
# MODO 2 – Duas Fontes NF–NA–NF
# ---------------------------------------------------------

with tab2:

    st.header("⚡ Modo 2 – Duas Fontes (NF – NA – NF Bloqueio)")

    if not DB_MODO2.exists():
        st.error("Banco ieee123_duasfontes.db não encontrado.")
        st.stop()

    coords2 = load_coords(DB_MODO2)
    topo2 = load_topology(DB_MODO2)
    loads2 = load_loads(DB_MODO2)
    combos = load_nfnanf(DB_MODO2)

    st.subheader("Mapa Base (Modo 2)")
    st.plotly_chart(plot_network(coords2, topo2), use_container_width=True)

    # filtros
    nf_list = sorted({c["nf1"] for c in combos})
    na_list = sorted({c["na"] for c in combos if c["na"]})
    nfb_list = sorted({c["nf_block"] for c in combos if c["nf_block"]})

    st.subheader("Filtrar Combinações NF–NA–NF")

    c1, c2, c3 = st.columns(3)
    sel_nf = c1.selectbox("NF isoladora", ["(todas)"] + nf_list)
    sel_na = c2.selectbox("NA", ["(todas)"] + na_list)
    sel_nfb = c3.selectbox("NF bloqueio", ["(todas)"] + nfb_list)

    filtrado = combos
    if sel_nf != "(todas)":
        filtrado = [c for c in filtrado if c["nf1"] == sel_nf]
    if sel_na != "(todas)":
        filtrado = [c for c in filtrado if c["na"] == sel_na]
    if sel_nfb != "(todas)":
        filtrado = [c for c in filtrado if c["nf_block"] == sel_nfb]

    st.write(f"{len(filtrado)} combinações encontradas.")

    if filtrado:
        filtrado_sorted = sorted(filtrado, key=lambda x: (x["kw_off"], x["n_manobras"]))
        best = filtrado_sorted[0]

        st.success(
            f"Melhor cenário: **NF1={best['nf1']}**, "
            f"**NA={best['na']}**, **NF_bloq={best['nf_block']}**  \n"
            f"⛔ **{best['kw_off']:.1f} kW** desligados, "
            f"{best['n_manobras']} manobras"
        )

        st.subheader("Mapa do Cenário Selecionado")

        st.plotly_chart(
            plot_network(
                coords2,
                topo2,
                buses_highlight=best["buses_off"]
            ),
            use_container_width=True
        )

        st.subheader("Linha do Tempo da Manobra")
        st.markdown(f"1️⃣ Abrir NF isoladora **{best['nf1']}**")
        if best["na"]:
            st.markdown(f"2️⃣ Fechar NA **{best['na']}**")
        if best["nf_block"]:
            st.markdown(f"3️⃣ Abrir NF de bloqueio **{best['nf_block']}**")
        st.markdown("🟢 Após manutenção → fechar na ordem inversa.")
