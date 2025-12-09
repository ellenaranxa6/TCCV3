import streamlit as st
import plotly.graph_objects as go
import sqlite3
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import ast
import json

# =========================================================
# CONFIGURAÇÃO INICIAL
# =========================================================

st.set_page_config(
    page_title="TopoSwitch – IEEE 123 Bus",
    layout="wide"
)

st.title("⚡ TopoSwitch – IEEE 123 Bus (Isolamento Real)")

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "ieee123_manobras.db"  # banco único Modo 1 + Modo 2


# =========================================================
# AUXILIARES BANCO
# =========================================================

def get_connection() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


@st.cache_data(show_spinner=False)
def listar_tabelas() -> List[str]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    )
    rows = [r[0] for r in cur.fetchall()]
    conn.close()
    return rows


@st.cache_data(show_spinner=False)
def carregar_coords() -> Dict[str, Tuple[float, float]]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT bus, x, y FROM coords")
    rows = cur.fetchall()
    conn.close()
    return {str(b): (float(x), float(y)) for b, x, y in rows}


@st.cache_data(show_spinner=False)
def carregar_topologia() -> List[Dict]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT line, from_bus, to_bus, is_switch, norm FROM topology"
    )
    rows = cur.fetchall()
    conn.close()
    topo = []
    for line, f, t, is_sw, norm in rows:
        topo.append(
            dict(
                line=str(line),
                from_bus=str(f),
                to_bus=str(t),
                is_switch=bool(is_sw),
                norm=str(norm),
            )
        )
    return topo


@st.cache_data(show_spinner=False)
def carregar_loads() -> Dict[str, float]:
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT bus, kw FROM loads")
        rows = cur.fetchall()
    except sqlite3.OperationalError:
        rows = []
    conn.close()
    return {str(b): float(kw) for b, kw in rows}


@st.cache_data(show_spinner=False)
def carregar_vao_map() -> List[Dict]:
    """
    Tabela vao_map(u_bus, v_bus, nf, kw, n_barras) – usada no Modo 1.
    """
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT u_bus, v_bus, nf, kw, n_barras FROM vao_map")
        rows = cur.fetchall()
    except sqlite3.OperationalError:
        rows = []
    conn.close()

    vao_map = []
    for u, v, nf, kw, n in rows:
        vao_map.append(
            dict(
                u_bus=str(u),
                v_bus=str(v),
                nf=str(nf),
                kw=float(kw),
                n_barras=int(n),
            )
        )
    return vao_map


@st.cache_data(show_spinner=False)
def carregar_nf_map() -> Dict[str, Dict]:
    """
    nf_map(nf, barras_isoladas TEXT, kw REAL, n_barras INTEGER)
    – usado no Modo 1 para impacto consolidado.
    """
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT nf, barras_isoladas, kw, n_barras FROM nf_map")
        rows = cur.fetchall()
    except sqlite3.OperationalError:
        rows = []
    conn.close()

    nf_dict: Dict[str, Dict] = {}
    for nf, barras_str, kw, n in rows:
        barras_set = set()
        if barras_str:
            try:
                lista = ast.literal_eval(barras_str)
                for b in lista:
                    barras_set.add(str(b).strip())
            except Exception:
                for b in str(barras_str).replace("[", "").replace("]", "").replace('"', "").split(","):
                    b = b.strip()
                    if b:
                        barras_set.add(b)

        nf_dict[str(nf)] = {
            "barras": barras_set,
            "kw": float(kw),
            "n_barras": int(n),
        }
    return nf_dict


@st.cache_data(show_spinner=False)
def carregar_nf_na_nf() -> List[Dict]:
    """
    Tabela nf_na_nf(id, nf1, na, nf_block, buses_off, lines_off,
                    kw_off, vmin_pu, vmax_pu, max_loading, n_manobras, switch_states)

    Usada no Modo 2 – Duas fontes.
    """
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT id, nf1, na, nf_block, buses_off, lines_off,
                   kw_off, vmin_pu, vmax_pu, max_loading, n_manobras, switch_states
            FROM nf_na_nf
            """
        )
        rows = cur.fetchall()
    except sqlite3.OperationalError:
        rows = []
    conn.close()

    data: List[Dict] = []
    for row in rows:
        (
            rid,
            nf1,
            na,
            nf_block,
            buses_off_str,
            lines_off_str,
            kw_off,
            vmin_pu,
            vmax_pu,
            max_loading,
            n_manobras,
            switch_states_str,
        ) = row

        def parse_list(s) -> List[str]:
            if s is None:
                return []
            try:
                v = ast.literal_eval(s)
                return [str(x) for x in v]
            except Exception:
                return [x.strip() for x in str(s).replace("[", "").replace("]", "").replace('"', "").split(",") if x.strip()]

        def parse_dict(s) -> Dict[str, str]:
            if s is None:
                return {}
            try:
                v = json.loads(s)
                return {str(k): str(vv) for k, vv in v.items()}
            except Exception:
                try:
                    v = ast.literal_eval(s)
                    if isinstance(v, dict):
                        return {str(k): str(vv) for k, vv in v.items()}
                except Exception:
                    return {}

        data.append(
            dict(
                id=int(rid),
                nf1=str(nf1) if nf1 is not None else None,
                na=str(na) if na is not None else None,
                nf_block=str(nf_block) if nf_block is not None else None,
                buses_off=parse_list(buses_off_str),
                lines_off=parse_list(lines_off_str),
                kw_off=float(kw_off),
                vmin_pu=float(vmin_pu),
                vmax_pu=float(vmax_pu),
                max_loading=float(max_loading),
                n_manobras=int(n_manobras),
                switch_states=parse_dict(switch_states_str),
            )
        )

    return data


# =========================================================
# FUNÇÕES MODO 1 – FONTE ÚNICA
# =========================================================

def identificar_vaos_blocos(lista_barras: List[str]) -> List[Tuple[str, str]]:
    """
    Converte uma sequência de barras em pares (vãos)
    ex: [60,62,63,64,65,66,60,67] -> [(60,62),(63,64),(65,66),(60,67)]
    """
    vaos: List[Tuple[str, str]] = []
    for i in range(0, len(lista_barras), 2):
        if i + 1 < len(lista_barras):
            u = lista_barras[i].strip()
            v = lista_barras[i + 1].strip()
            if u and v:
                vaos.append((u, v))
    return vaos


def buscar_nf_para_vao(u: str, v: str, vao_map: List[Dict]) -> Optional[Dict]:
    """
    Procura no vao_map a NF ótima para o vão (u, v),
    indiferente à ordem (u,v) ou (v,u).
    """
    candidatos = [
        reg for reg in vao_map
        if (reg["u_bus"] == u and reg["v_bus"] == v)
        or (reg["u_bus"] == v and reg["v_bus"] == u)
    ]
    if not candidatos:
        return None
    candidatos.sort(key=lambda r: (r["kw"], r["n_barras"]))
    return candidatos[0]


def obter_barras_unicas(vaos: List[Tuple[str, str]]) -> List[str]:
    s = set()
    for u, v in vaos:
        s.add(u)
        s.add(v)
    return sorted(s, key=lambda x: (x.isdigit(), int(x) if x.isdigit() else x))


def impacto_consolidado_modo1(
    lista_nf: List[str],
    loads: Dict[str, float],
    nf_map_data: Dict[str, Dict],
) -> Tuple[float, int, List[str]]:
    """
    Soma impacto de várias NFs sem dupla contagem de barras.
    """
    barras_afetadas = set()
    for nf in lista_nf:
        reg = nf_map_data.get(nf)
        if reg:
            barras_afetadas |= reg["barras"]

    kw_total = sum(loads.get(b, 0.0) for b in barras_afetadas)
    barras_ordenadas = sorted(
        barras_afetadas,
        key=lambda x: (x.isdigit(), int(x) if x.isdigit() else x),
    )
    return kw_total, len(barras_afetadas), barras_ordenadas


# =========================================================
# FUNÇÕES MODO 2 – DUAS FONTES
# =========================================================

def normalizar_barras(lista: List[str]) -> List[str]:
    out = []
    for b in lista:
        s = str(b).strip()
        if s:
            out.append(s)
    return out


def find_options_modo2(
    trecho_barras: List[str],
    nf_na_nf_rows: List[Dict],
) -> List[Dict]:
    """
    Replica a lógica do script do Colab a partir da tabela nf_na_nf:

    1. NF candidata precisa, sozinha (na=None, nf_block=None), desligar o trecho.
    2. Cenários (NF1 + NA + NF_block) só entram se o trecho estiver desligado
       e se kw_off < kw_off_base_nf1 (uso de NA precisa compensar).
    3. Ordenação: (kw_off, n_manobras, kw_off_base_nf1)
    """
    trecho_set = set(normalizar_barras(trecho_barras))
    if not trecho_set:
        return []

    # --- Mapear base de cada NF1 (somente NF aberta, sem NA / NF_bloq)
    base_por_nf: Dict[str, Dict] = {}
    for row in nf_na_nf_rows:
        if row["na"] is None and row["nf_block"] is None and row["nf1"] is not None:
            base_por_nf[row["nf1"]] = row

    # --- NF que realmente isolam o trecho sozinhas
    nf_validas = []
    for nf1, base in base_por_nf.items():
        buses_off = set(normalizar_barras(base["buses_off"]))
        if trecho_set.issubset(buses_off):
            nf_validas.append(nf1)

    if not nf_validas:
        return []

    candidates: List[Dict] = []

    for row in nf_na_nf_rows:
        nf1 = row["nf1"]
        if nf1 not in nf_validas:
            continue

        buses_off = set(normalizar_barras(row["buses_off"]))
        if not trecho_set.issubset(buses_off):
            continue

        base = base_por_nf[nf1]
        kw_base = base["kw_off"]

        # Se usa NA e/ou NF_bloq, só aceita se reduzir carga desligada
        if (row["na"] is not None or row["nf_block"] is not None) and row["kw_off"] >= kw_base:
            continue

        cand = row.copy()
        cand["kw_off_base_nf"] = kw_base
        candidates.append(cand)

    if not candidates:
        return []

    # Ordena como no Colab
    candidates_sorted = sorted(
        candidates,
        key=lambda o: (o["kw_off"], o["n_manobras"], o["kw_off_base_nf"]),
    )

    # Remove duplicatas NF1/NA/NF_bloq
    uniq = []
    seen = set()
    for opt in candidates_sorted:
        key = (opt["nf1"], opt["na"], opt["nf_block"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(opt)

    return uniq


# =========================================================
# PLOT – MAPA BASE + RÓTULO DAS LINHAS
# =========================================================

def construir_mapa_base(
    coords: Dict[str, Tuple[float, float]],
    topo: List[Dict],
    show_line_labels: bool = True,
) -> go.Figure:
    """
    Desenha a topologia base com Plotly:
      - linhas em cinza claro
      - barras em azul
      - opcionalmente, rótulos das linhas no meio do segmento
    """
    edge_x = []
    edge_y = []

    for el in topo:
        u = el["from_bus"]
        v = el["to_bus"]
        if u in coords and v in coords:
            x0, y0 = coords[u]
            x1, y1 = coords[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

    node_x = [coords[b][0] for b in coords]
    node_y = [coords[b][1] for b in coords]
    node_text = list(coords.keys())

    fig = go.Figure()

    # Linhas
    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(color="#D3D3D3", width=1),
            hoverinfo="none",
            name="Linhas",
        )
    )

    # Barras
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_text,
            textposition="top center",
            marker=dict(size=6, color="#1f77b4"),
            name="Barras",
            hovertemplate="Barra %{text}<extra></extra>",
        )
    )

    # Rótulos das linhas
    if show_line_labels:
        label_x = []
        label_y = []
        label_text = []
        for el in topo:
            line = el["line"]
            u = el["from_bus"]
            v = el["to_bus"]
            if u in coords and v in coords:
                x0, y0 = coords[u]
                x1, y1 = coords[v]
                xm = (x0 + x1) / 2.0
                ym = (y0 + y1) / 2.0
                label_x.append(xm)
                label_y.append(ym)
                label_text.append(str(line))

        fig.add_trace(
            go.Scatter(
                x=label_x,
                y=label_y,
                mode="text",
                text=label_text,
                textposition="middle center",
                textfont=dict(color="#555555", size=7),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.update_layout(
        height=600,
        showlegend=True,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )

    return fig


# =========================================================
# PLOT – MODO 1 (VÃOS + NFs)
# =========================================================

def plotar_mapa_modo1(
    coords,
    topo,
    vaos: List[Tuple[str, str]],
    info_vaos: List[Dict],
) -> go.Figure:
    """
    Mapa para o Modo 1:
      - vãos selecionados em preto
      - NFs em vermelho tracejado
      - rótulo das linhas visível
    """
    fig = construir_mapa_base(coords, topo, show_line_labels=True)

    # Destaque dos vãos
    destaque_edge_x = []
    destaque_edge_y = []
    for u, v in vaos:
        if u in coords and v in coords:
            x0, y0 = coords[u]
            x1, y1 = coords[v]
            destaque_edge_x += [x0, x1, None]
            destaque_edge_y += [y0, y1, None]

    if destaque_edge_x:
        fig.add_trace(
            go.Scatter(
                x=destaque_edge_x,
                y=destaque_edge_y,
                mode="lines",
                line=dict(color="black", width=4),
                name="Trecho selecionado (vãos)",
                hoverinfo="none",
            )
        )

    # NF associadas aos vãos
    topo_por_line = {el["line"]: el for el in topo}
    nf_edges_x = []
    nf_edges_y = []
    nf_labels_x = []
    nf_labels_y = []
    nf_labels_text = []

    for info in info_vaos:
        nf = info["nf"]
        if nf in topo_por_line:
            el = topo_por_line[nf]
            u = el["from_bus"]
            v = el["to_bus"]
            if u in coords and v in coords:
                x0, y0 = coords[u]
                x1, y1 = coords[v]
                nf_edges_x += [x0, x1, None]
                nf_edges_y += [y0, y1, None]
                nf_labels_x.append((x0 + x1) / 2)
                nf_labels_y.append((y0 + y1) / 2)
                nf_labels_text.append(nf)

    if nf_edges_x:
        fig.add_trace(
            go.Scatter(
                x=nf_edges_x,
                y=nf_edges_y,
                mode="lines",
                line=dict(color="red", width=3, dash="dash"),
                name="Chaves NF de manobra",
                hoverinfo="none",
            )
        )

    if nf_labels_x:
        fig.add_trace(
            go.Scatter(
                x=nf_labels_x,
                y=nf_labels_y,
                mode="text",
                text=nf_labels_text,
                textposition="middle center",
                textfont=dict(color="red", size=10),
                showlegend=False,
            )
        )

    return fig


# =========================================================
# PLOT – MODO 2 (DUAS FONTES)
# =========================================================

def plotar_mapa_modo2(
    coords: Dict[str, Tuple[float, float]],
    topo: List[Dict],
    trecho_barras: List[str],
    option: Dict,
) -> go.Figure:
    """
    Mapa para o Modo 2 (duas fontes):
      - barras energizadas (verde) x desligadas (vermelho)
      - trecho alvo em amarelo
      - NF1 em vermelho
      - NA em ciano
      - NF_bloq em roxo
      - rótulo das linhas
    """
    fig = construir_mapa_base(coords, topo, show_line_labels=True)

    all_buses = set(coords.keys())
    buses_off = set(normalizar_barras(option.get("buses_off", [])))
    trecho_set = set(normalizar_barras(trecho_barras))
    buses_on = all_buses - buses_off

    # Barras desligadas
    off_x = [coords[b][0] for b in buses_off if b in coords]
    off_y = [coords[b][1] for b in buses_off if b in coords]
    if off_x:
        fig.add_trace(
            go.Scatter(
                x=off_x,
                y=off_y,
                mode="markers",
                marker=dict(size=8, color="red"),
                name="Barras desligadas",
                hovertemplate="Barra %{x},%{y}<extra></extra>",
            )
        )

    # Barras energizadas
    on_x = [coords[b][0] for b in buses_on if b in coords]
    on_y = [coords[b][1] for b in buses_on if b in coords]
    if on_x:
        fig.add_trace(
            go.Scatter(
                x=on_x,
                y=on_y,
                mode="markers",
                marker=dict(size=7, color="green"),
                name="Barras energizadas",
                hoverinfo="skip",
            )
        )

    # Trecho alvo (amarelo)
    trecho_x = [coords[b][0] for b in trecho_set if b in coords]
    trecho_y = [coords[b][1] for b in trecho_set if b in coords]
    if trecho_x:
        fig.add_trace(
            go.Scatter(
                x=trecho_x,
                y=trecho_y,
                mode="markers",
                marker=dict(size=10, color="yellow", line=dict(color="black", width=1)),
                name="Trecho alvo (barras)",
                hoverinfo="skip",
            )
        )

    # Mapa linha -> elemento topo
    topo_por_line = {el["line"]: el for el in topo}

    def add_special_line(line_name: Optional[str], color: str, label: str):
        if line_name is None:
            return
        ln = str(line_name)
        if ln not in topo_por_line:
            return
        el = topo_por_line[ln]
        u = el["from_bus"]
        v = el["to_bus"]
        if u in coords and v in coords:
            x0, y0 = coords[u]
            x1, y1 = coords[v]
            fig.add_trace(
                go.Scatter(
                    x=[x0, x1],
                    y=[y0, y1],
                    mode="lines",
                    line=dict(color=color, width=4),
                    name=label,
                    hoverinfo="none",
                )
            )

    add_special_line(option.get("nf1"), "red", "NF isoladora (NF1)")
    add_special_line(option.get("na"), "cyan", "NA de restabelecimento")
    add_special_line(option.get("nf_block"), "purple", "NF de bloqueio")

    fig.update_layout(
        height=650,
        margin=dict(l=10, r=10, t=30, b=10),
    )

    return fig


# =========================================================
# CARREGAR DADOS & STATUS
# =========================================================

st.sidebar.header("📂 Status do banco de dados")

if not DB_PATH.exists():
    st.sidebar.error(f"Banco `{DB_PATH.name}` não encontrado na pasta do app.")
    st.stop()

tabelas = listar_tabelas()
st.sidebar.write("Banco:", f"`{DB_PATH.name}`")
st.sidebar.write("Tabelas encontradas:", ", ".join(tabelas))

coords = carregar_coords()
topo = carregar_topologia()
loads = carregar_loads()
vao_map = carregar_vao_map()
nf_map_data = carregar_nf_map()
nf_na_nf_rows = carregar_nf_na_nf()

if not coords or not topo:
    st.error("Banco encontrado, mas `coords` ou `topology` está vazio.")
    st.stop()

# Campo do operador (nome usado em relatórios)
nome_operador = st.text_input("👤 Nome do operador responsável", value="Ellen")
st.info(f"Usuário: **{nome_operador}**")

st.markdown("---")

# =========================================================
# MAPA BASE
# =========================================================

st.subheader("🗺️ Mapa base da rede IEEE-123 Bus (com nomes das linhas)")
fig_base = construir_mapa_base(coords, topo, show_line_labels=True)
st.plotly_chart(fig_base, use_container_width=True)

st.markdown("---")

# =========================================================
# TABS – MODO 1 / MODO 2
# =========================================================

tab1, tab2 = st.tabs(["🔌 Modo 1 – Fonte única", "⚡ Modo 2 – Duas fontes"])


# ---------------------------------------------------------
# TAB 1 – MODO 1 (Fonte única)
# ---------------------------------------------------------
with tab1:
    st.subheader("🔧 Modo 1 – Isolamento por NFs (fonte única)")

    if not vao_map:
        st.warning("Tabela `vao_map` não encontrada ou vazia no banco.")
    else:
        st.markdown("### 🎯 Vão simples (U–V)")

        lista_barras = sorted(
            coords.keys(),
            key=lambda x: (x.isdigit(), int(x) if x.isdigit() else x),
        )

        col_u, col_v = st.columns(2)
        with col_u:
            u_simples = st.selectbox("Barra U", lista_barras, key="modo1_u")
        with col_v:
            v_simples = st.selectbox("Barra V", lista_barras, key="modo1_v")

        if st.button("Calcular NF para vão simples", key="btn_modo1_simples"):
            vao_simples = (u_simples, v_simples)
            info = buscar_nf_para_vao(u_simples, v_simples, vao_map)

            st.markdown("#### 🔎 Resultado – Vão simples")

            if info is None:
                st.error(
                    f"Não há NF cadastrada no banco para o vão {u_simples} – {v_simples}."
                )
            else:
                st.success(
                    f"**Melhor NF:** `{info['nf']}`  |  "
                    f"**Carga interrompida (NF isolada):** {info['kw']:.1f} kW  |  "
                    f"**Barras isoladas:** {info['n_barras']}"
                )

                fig_vao = plotar_mapa_modo1(
                    coords,
                    topo,
                    vaos=[vao_simples],
                    info_vaos=[info],
                )
                st.plotly_chart(fig_vao, use_container_width=True)

                st.markdown("**Sequência de manobra sugerida:**")
                st.markdown(
                    f"1️⃣ Abrir NF **{info['nf']}** para isolar o vão **{u_simples} – {v_simples}**.  \n"
                    f"✅ Demais chaves permanecem no estado nominal."
                )

            st.markdown("---")

        # -------- TRECHO COM MÚLTIPLOS VÃOS --------
        st.markdown("### 🧩 Trecho com múltiplos vãos (entrada em blocos de 2 barras)")

        entrada_seq = st.text_input(
            "Sequência de barras (ex: 60,62,63,64,65,66,60,67)",
            value="60,62,63,64,65,66,60,67",
            key="modo1_seq",
        )

        if st.button("Processar trecho (Modo 1)", key="btn_modo1_multi"):
            barras_raw = [b.strip() for b in entrada_seq.split(",") if b.strip()]
            if len(barras_raw) < 2:
                st.error("Informe pelo menos duas barras.")
            else:
                vaos = identificar_vaos_blocos(barras_raw)
                if not vaos:
                    st.error("Nenhum vão pôde ser formado com a sequência informada.")
                else:
                    st.markdown("#### 🔍 Vãos identificados (blocos de 2 barras):")
                    st.write(vaos)

                    info_vaos = []
                    nao_encontrados = []

                    for u, v in vaos:
                        info = buscar_nf_para_vao(u, v, vao_map)
                        if info is None:
                            nao_encontrados.append((u, v))
                        else:
                            info_vaos.append(
                                dict(
                                    u=u,
                                    v=v,
                                    nf=info["nf"],
                                    kw=info["kw"],
                                    n_barras=info["n_barras"],
                                )
                            )

                    if nao_encontrados:
                        st.warning(
                            "Não foram encontrados registros para os seguintes vãos: "
                            + ", ".join([f"{u}-{v}" for u, v in nao_encontrados])
                        )

                    if info_vaos:
                        st.markdown("#### ✅ NF de manobra por vão (impacto individual)")

                        df_data = [
                            {
                                "Vão (U–V)": f"{d['u']} - {d['v']}",
                                "NF ótima": d["nf"],
                                "kW interrompidos (NF isolada)": d["kw"],
                                "Barras isoladas (NF isolada)": d["n_barras"],
                            }
                            for d in info_vaos
                        ]
                        st.table(df_data)

                        fig_multi = plotar_mapa_modo1(
                            coords,
                            topo,
                            vaos=[(d["u"], d["v"]) for d in info_vaos],
                            info_vaos=info_vaos,
                        )
                        st.markdown("#### 🗺️ Mapa com trecho e NFs destacadas")
                        st.plotly_chart(fig_multi, use_container_width=True)

                        # Impacto consolidado
                        st.markdown("#### ⚡ Impacto consolidado da manobra (sem dupla contagem)")

                        lista_nf_ordenada: List[str] = []
                        for d in info_vaos:
                            if d["nf"] not in lista_nf_ordenada:
                                lista_nf_ordenada.append(d["nf"])

                        if not nf_map_data:
                            st.warning(
                                "Tabela `nf_map` não está disponível no banco. "
                                "Não foi possível calcular o impacto consolidado."
                            )
                        else:
                            kw_total, n_barras_unicas, barras_ordenadas = impacto_consolidado_modo1(
                                lista_nf_ordenada, loads, nf_map_data
                            )

                            st.success(
                                f"**Carga total interrompida:** {kw_total:.1f} kW  \n"
                                f"**Barras desenergizadas únicas:** {n_barras_unicas}"
                            )
                            with st.expander("Ver barras desenergizadas únicas"):
                                st.write(barras_ordenadas)

                        st.markdown("#### 📜 Linha de tempo de manobra (sequência sugerida)")
                        for i, nf in enumerate(lista_nf_ordenada, start=1):
                            vaos_nf = [
                                f"{d['u']}-{d['v']}"
                                for d in info_vaos
                                if d["nf"] == nf
                            ]
                            st.markdown(
                                f"{i}️⃣ Abrir NF **{nf}** para isolar os vãos: "
                                + ", ".join(vaos_nf)
                            )

                        st.markdown(
                            "✅ Após conclusão da manutenção, **fechar as NFs na ordem inversa**, "
                            "conforme os procedimentos operacionais da distribuidora."
                        )


# ---------------------------------------------------------
# TAB 2 – MODO 2 (Duas fontes)
# ---------------------------------------------------------
with tab2:
    st.subheader("⚡ Modo 2 – Manobras com duas fontes (NF–NA–NF)")

    if not nf_na_nf_rows:
        st.warning("Tabela `nf_na_nf` não encontrada ou vazia no banco.")
    else:
        st.markdown("### 🎯 Trecho alvo (barras) – Modo 2")

        entrada_barras_m2 = st.text_input(
            "Lista de barras do trecho a desenergizar (ex: 69,70 ou 69,70,71)",
            value="69,70",
            key="modo2_barras",
        )

        if st.button("Buscar opções de manobra (Modo 2)", key="btn_modo2_buscar"):
            barras_raw = [b.strip() for b in entrada_barras_m2.split(",") if b.strip()]
            trecho_barras = normalizar_barras(barras_raw)

            st.write(f"Trecho considerado (barras): {trecho_barras}")

            options = find_options_modo2(trecho_barras, nf_na_nf_rows)

            if not options:
                st.error("Nenhum cenário de manobra encontrado que desligue exatamente esse trecho.")
            else:
                import pandas as pd

                df_opts = pd.DataFrame(
                    [
                        dict(
                            Opção=i,
                            NF_isoladora=row["nf1"],
                            NA=row["na"] or "•",
                            NF_bloqueio=row["nf_block"] or "•",
                            Carga_desligada_kW=row["kw_off"],
                            N_manobras=row["n_manobras"],
                            Vmin_pu=row["vmin_pu"],
                            Vmax_pu=row["vmax_pu"],
                            Carregamento_max_pu=row["max_loading"],
                            Impacto_NF_isoladora_kW=row["kw_off_base_nf"],
                        )
                        for i, row in enumerate(options, start=1)
                    ]
                )

                st.markdown("### ✅ TOP opções de manobra (ordenadas por menor carga desligada)")
                st.dataframe(df_opts, use_container_width=True)

                max_op = len(options)

                # Inicializa o estado se não existir
                if "modo2_opcao" not in st.session_state:
                    st.session_state["modo2_opcao"] = 1
                    
                opcao_det = st.selectbox(
                    "Escolha o número da opção para detalhar:",
                    options=list(range(1, max_op + 1)),
                    index=st.session_state["modo2_opcao"] - 1,
                    key="modo2_opcao_select"
                )
                    
                # Armazena o valor escolhido no estado
                st.session_state["modo2_opcao"] = opcao_det
                    
                idx = opcao_det - 1
                opt_sel = options[idx]

                st.markdown(f"#### 🔍 Detalhamento da opção {opcao_det}")

                st.write(
                    f"- **NF isoladora (NF1):** `{opt_sel['nf1']}`  \n"
                    f"- **NA de restabelecimento:** `{opt_sel['na'] or '-'}`  \n"
                    f"- **NF de bloqueio:** `{opt_sel['nf_block'] or '-'}`  \n"
                    f"- **Carga desligada:** `{opt_sel['kw_off']:.4f} kW`  \n"
                    f"- **Nº de manobras:** `{opt_sel['n_manobras']}`  \n"
                    f"- **Vmin/Vmax [pu]:** `{opt_sel['vmin_pu']:.4f} / {opt_sel['vmax_pu']:.4f}`  \n"
                    f"- **Carregamento máximo [pu]:** `{opt_sel['max_loading']:.4f}`  \n"
                    f"- **Impacto NF isoladora (NF1 sozinha):** `{opt_sel['kw_off_base_nf']:.4f} kW`  \n"
                    f"- **Usuário:** `{nome_operador}`"
                )

                st.markdown("#### 🗺️ Mapa da manobra (Modo 2)")
                fig_m2 = plotar_mapa_modo2(coords, topo, trecho_barras, opt_sel)
                st.plotly_chart(fig_m2, use_container_width=True)

                # Exibir listas de barras/linhas desligadas
                with st.expander("Ver barras desligadas e linhas desligadas da opção selecionada"):
                    st.write("**Barras desligadas:**", normalizar_barras(opt_sel.get("buses_off", [])))
                    st.write("**Linhas desligadas:**", normalizar_barras(opt_sel.get("lines_off", [])))

                st.markdown(
                    f"📄 *Resumo da manobra (Modo 2) – opção {opcao_det} – Usuário: {nome_operador}*"
                )
