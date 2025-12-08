import streamlit as st
import plotly.graph_objects as go
import sqlite3
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import ast


# =========================================================
# CONFIGURAÇÃO INICIAL
# =========================================================
st.set_page_config(page_title="TopoSwitch – IEEE123", layout="wide")
st.title("⚡ TopoSwitch – Plataforma Interativa (IEEE-123)")


BASE_DIR = Path(__file__).parent

DB_MODO1 = BASE_DIR / "ieee123_isolamento.db"
DB_MODO2 = BASE_DIR / "ieee123_duasfontes.db"


# =========================================================
# FUNÇÕES AUXILIARES – BANCO
# =========================================================
def get_conn(db_path: Path) -> sqlite3.Connection:
    return sqlite3.connect(db_path)


@st.cache_data
def carregar_coords(db_path: Path) -> Dict[str, Tuple[float, float]]:
    conn = get_conn(db_path)
    cur = conn.cursor()
    try:
        cur.execute("SELECT bus, x, y FROM coords")
        rows = cur.fetchall()
    except:
        rows = []
    conn.close()
    return {str(b): (float(x), float(y)) for b, x, y in rows}


@st.cache_data
def carregar_topologia(db_path: Path):
    conn = get_conn(db_path)
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT line, from_bus, to_bus, is_switch, norm
            FROM topology
        """)
        rows = cur.fetchall()
    except:
        rows = []
    conn.close()

    topo = []
    for l, u, v, is_sw, norm in rows:
        topo.append(dict(
            line=str(l),
            from_bus=str(u),
            to_bus=str(v),
            is_switch=bool(is_sw),
            norm=str(norm)
        ))
    return topo


@st.cache_data
def carregar_loads(db_path: Path):
    conn = get_conn(db_path)
    cur = conn.cursor()
    try:
        cur.execute("SELECT bus, kw FROM loads")
        rows = cur.fetchall()
    except:
        rows = []
    conn.close()
    return {str(b): float(kw) for b, kw in rows}


@st.cache_data
def carregar_vao_map_modo1(db_path: Path):
    """u_bus, v_bus, nf, kw, n_barras"""
    conn = get_conn(db_path)
    cur = conn.cursor()
    try:
        cur.execute("SELECT u_bus, v_bus, nf, kw, n_barras FROM vao_map")
        rows = cur.fetchall()
    except:
        rows = []
    conn.close()

    vao_map = []
    for u, v, nf, kw, n in rows:
        vao_map.append(dict(
            u_bus=str(u),
            v_bus=str(v),
            nf=str(nf),
            kw=float(kw),
            n_barras=int(n)
        ))
    return vao_map


@st.cache_data
def carregar_nf_map(db_path: Path):
    """NF isolada → lista de barras isoladas."""
    conn = get_conn(db_path)
    cur = conn.cursor()
    try:
        cur.execute("SELECT nf, barras_isoladas, kw, n_barras FROM nf_map")
        rows = cur.fetchall()
    except:
        rows = []
    conn.close()

    nf_map = {}
    for nf, barras_str, kw, n in rows:
        try:
            lista = ast.literal_eval(barras_str)
            barras = {str(b).strip() for b in lista}
        except:
            barras = set()

        nf_map[str(nf)] = dict(
            barras=barras,
            kw=float(kw),
            n_barras=int(n),
        )

    return nf_map


# -------------------------------
# MODO 2 – NF, NA, NF–BLOCK ótimas
# -------------------------------
@st.cache_data
def carregar_modo2_map(db_path: Path):
    """
    Tabela: duasfontes_map
    Campos:
        u_bus, v_bus,
        nf1, na, nf_block,
        kw, n_manobras, barras_desligadas
    """
    conn = get_conn(db_path)
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT u_bus, v_bus, nf1, na, nf_block, kw, n_manobras, barras_desligadas
            FROM duasfontes_map
        """)
        rows = cur.fetchall()
    except:
        rows = []
    conn.close()

    data = []
    for u, v, nf1, na, nf_block, kw, nm, barras_str in rows:
        try:
            barras = ast.literal_eval(barras_str)
            barras = {str(b) for b in barras}
        except:
            barras = set()

        data.append(dict(
            u=str(u),
            v=str(v),
            nf1=str(nf1),
            na=str(na) if na else None,
            nf_block=str(nf_block) if nf_block else None,
            kw=float(kw),
            n_manobras=int(nm),
            barras=barras,
        ))
    return data



# =========================================================
# FUNÇÕES DE PROCESSAMENTO
# =========================================================
def identificar_vaos(bloco: List[str]) -> List[Tuple[str, str]]:
    vaos = []
    for i in range(0, len(bloco), 2):
        if i + 1 < len(bloco):
            vaos.append((bloco[i], bloco[i+1]))
    return vaos


def buscar_nf_modo1(u, v, vao_map):
    cand = [r for r in vao_map if (r["u_bus"], r["v_bus"]) in [(u, v), (v, u)]]
    if not cand:
        return None
    cand.sort(key=lambda r: (r["kw"], r["n_barras"]))
    return cand[0]


def buscar_modo2(u, v, map2):
    cand = [r for r in map2 if (r["u"], r["v"]) in [(u, v), (v, u)]]
    if not cand:
        return None
    cand.sort(key=lambda r: (r["kw"], r["n_manobras"]))
    return cand[0]


def impacto_consolidado(nfs: List[str], loads: Dict[str, float], nf_map):
    barras = set()
    for nf in nfs:
        if nf in nf_map:
            barras |= nf_map[nf]["barras"]
    kw = sum(loads.get(b, 0.0) for b in barras)
    return kw, len(barras), sorted(list(barras))


# =========================================================
# PLOTAGEM
# =========================================================
def plot_topologia(coords, topo, cor_barras=None, cor_linhas=None):
    edge_x = []
    edge_y = []

    for el in topo:
        u = el["from_bus"]
        v = el["to_bus"]
        if u in coords and v in coords:
            edge_x += [coords[u][0], coords[v][0], None]
            edge_y += [coords[u][1], coords[v][1], None]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(color="#bbb", width=1),
        hoverinfo="none",
    ))

    node_x = [coords[b][0] for b in coords]
    node_y = [coords[b][1] for b in coords]
    labels = list(coords.keys())

    if cor_barras:
        colors = [cor_barras.get(b, "#1f77b4") for b in coords]
    else:
        colors = "#1f77b4"

    fig.add_trace(go.Scatter(
        x=node_x,
        y=node_y,
        text=labels,
        mode="markers+text",
        textposition="top center",
        marker=dict(size=7, color=colors),
    ))

    fig.update_layout(
        height=650,
        showlegend=False,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=5, r=5, t=30, b=5),
    )

    return fig



# =========================================================
# INTERFACE
# =========================================================

st.sidebar.header("⚙️ Seleção do Modo de Operação")
modo = st.sidebar.radio("Modo:", ["Fonte Única", "Duas Fontes"])

db_path = DB_MODO1 if modo == "Fonte Única" else DB_MODO2

coords = carregar_coords(db_path)
topo = carregar_topologia(db_path)
loads = carregar_loads(db_path)

if modo == "Fonte Única":
    vao_map = carregar_vao_map_modo1(db_path)
    nf_map = carregar_nf_map(db_path)
else:
    modo2_map = carregar_modo2_map(db_path)


st.success(f"📦 Banco carregado: `{db_path.name}`")

st.subheader("🗺️ Topologia Base IEEE-123")
base_fig = plot_topologia(coords, topo)
st.plotly_chart(base_fig, use_container_width=True)



# ---------------------------------------------------------
# 🧩 Entrada do trecho
# ---------------------------------------------------------
st.markdown("---")
st.subheader("Seleção do trecho (blocos U-V)")

entrada = st.text_input(
    "Informe barras separadas por vírgula (ex: 60,62,63,64):",
    value="60,62"
)

if st.button("Processar"):
    barras = [b.strip() for b in entrada.split(",") if b.strip()]

    if len(barras) < 2:
        st.error("Informe ao menos duas barras.")
        st.stop()

    vaos = identificar_vaos(barras)
    st.write("Vãos identificados:", vaos)

    resultados = []
    nf_usadas = []

    if modo == "Fonte Única":
        for u, v in vaos:
            info = buscar_nf_modo1(u, v, vao_map)
            if info:
                resultados.append(info)
                nf_usadas.append(info["nf"])

    else:  # MODO 2
        for u, v in vaos:
            info = buscar_modo2(u, v, modo2_map)
            if info:
                resultados.append(info)
                nf_usadas.append(info["nf1"])

    if not resultados:
        st.error("Nenhum registro encontrado no banco para os vãos informados.")
        st.stop()

    st.markdown("### 🔎 Resultados por Vão")

    st.table(resultados)

    # Plot colorido
    barras_afetadas = set()
    for r in resultados:
        if modo == "Fonte Única":
            barras_afetadas |= nf_map[r["nf"]]["barras"]
        else:
            barras_afetadas |= r["barras"]

    cor_barras = {b: "red" for b in barras_afetadas}

    fig2 = plot_topologia(coords, topo, cor_barras=cor_barras)
    st.plotly_chart(fig2, use_container_width=True)

    # Impacto consolidado
    st.subheader("⚡ Impacto Consolidado")

    if modo == "Fonte Única":
        kw_total, nb, barras_final = impacto_consolidado(nf_usadas, loads, nf_map)
    else:
        kw_total = sum(loads.get(b, 0.0) for b in barras_afetadas)
        nb = len(barras_afetadas)
        barras_final = sorted(list(barras_afetadas))

    st.success(f"🔌 Carga total desligada: **{kw_total:.1f} kW**")
    st.info(f"Total de barras isoladas: **{nb}**")

    with st.expander("Ver barras isoladas"):
        st.write(barras_final)

    # Linha do tempo da manobra
    st.subheader("📜 Linha do Tempo da Manobra")

    if modo == "Fonte Única":
        for nf in nf_usadas:
            st.markdown(f"➡️ Abrir NF **{nf}**")
    else:
        for r in resultados:
            st.markdown(
                f"""
                🔹 Para o vão **{r['u']}-{r['v']}**  
                ➡️ Abrir NF isoladora **{r['nf1']}**  
                ➡️ Fechar NA **{r['na']}**  
                ➡️ Abrir NF bloqueio **{r['nf_block']}**  
                """
            )

    st.success("Após concluir a manutenção, **fechar as chaves na ordem inversa**.")
