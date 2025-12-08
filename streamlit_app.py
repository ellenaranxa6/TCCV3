import streamlit as st
import plotly.graph_objects as go
import sqlite3
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import ast

# =========================================================
# CONFIGURAÇÃO INICIAL
# =========================================================
st.set_page_config(
    page_title="Plataforma Interativa – Manobras IEEE 123 Bus",
    layout="wide"
)

st.title("⚡ Plataforma Interativa – Manobras IEEE 123 Bus")

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "ieee123_manobras.db"


# =========================================================
# FUNÇÕES AUXILIARES – BANCO E CARREGAMENTO
# =========================================================
def get_connection() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


@st.cache_data(show_spinner=False)
def listar_tabelas() -> List[str]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
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
def carregar_topologia():
    """
    Lê tabela topology(line, from_bus, to_bus, is_switch, norm).
    """
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
def carregar_loads():
    """Tabela loads(bus, kw)."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT bus, kw FROM loads")
    rows = cur.fetchall()
    conn.close()
    return {str(b): float(kw) for b, kw in rows}


@st.cache_data(show_spinner=False)
def carregar_nf_map():
    """
    Tabela nf_map(nf, barras_isoladas TEXT, kw REAL, n_barras INTEGER)
    para o MODO 1.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT nf, barras_isoladas, kw, n_barras FROM nf_map"
    )
    rows = cur.fetchall()
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
def carregar_vao_map():
    """
    Tabela vao_map(u_bus, v_bus, nf, kw, n_barras) – MODO 1.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT u_bus, v_bus, nf, kw, n_barras FROM vao_map"
    )
    rows = cur.fetchall()
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
def carregar_nf_na_nf():
    """
    Tabela nf_na_nf(nf1, na, nf_block, buses_off, lines_off,
                    kw_off, vmin_pu, vmax_pu, max_loading,
                    n_manobras, switch_states) – MODO 2.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT nf1, na, nf_block,
               buses_off, lines_off,
               kw_off, vmin_pu, vmax_pu,
               max_loading, n_manobras,
               switch_states
        FROM nf_na_nf
        """
    )
    rows = cur.fetchall()
    conn.close()

    registros = []

    for nf1, na, nf_block, buses_str, lines_str, kw, vmin, vmax, load, nman, sw_str in rows:
        try:
            buses = list(ast.literal_eval(buses_str)) if buses_str else []
        except Exception:
            buses = []

        try:
            lines = list(ast.literal_eval(lines_str)) if lines_str else []
        except Exception:
            lines = []

        try:
            states = dict(ast.literal_eval(sw_str)) if sw_str else {}
        except Exception:
            states = {}

        registros.append(
            dict(
                nf1=str(nf1),
                na=str(na) if na else None,
                nf_block=str(nf_block) if nf_block else None,
                buses_off=[str(b) for b in buses],
                lines_off=[str(l) for l in lines],
                kw_off=float(kw),
                vmin_pu=float(vmin),
                vmax_pu=float(vmax),
                max_loading=float(load),
                n_manobras=int(nman),
                switch_states=states,
            )
        )

    return registros


# =========================================================
# FUNÇÕES DE PROCESSAMENTO – MODO 1
# =========================================================
def identificar_vaos_blocos(lista_barras: List[str]) -> List[Tuple[str, str]]:
    """
    Converte lista de barras em pares disjuntos:
    [60,62,63,64,65,66,60,67] ->
    [(60,62), (63,64), (65,66), (60,67)]
    """
    vaos = []
    for i in range(0, len(lista_barras), 2):
        if i + 1 < len(lista_barras):
            u = lista_barras[i].strip()
            v = lista_barras[i + 1].strip()
            if u and v:
                vaos.append((u, v))
    return vaos


def buscar_nf_para_vao(
    u: str,
    v: str,
    vao_map: List[Dict]
) -> Optional[Dict]:
    """
    Procura no vao_map a NF ótima para o vão (u, v),
    considerando que o usuário pode informar em qualquer ordem.
    """
    candidatos = [
        registro for registro in vao_map
        if (registro["u_bus"] == u and registro["v_bus"] == v)
        or (registro["u_bus"] == v and registro["v_bus"] == u)
    ]
    if not candidatos:
        return None

    candidatos.sort(key=lambda r: (r["kw"], r["n_barras"]))
    return candidatos[0]


def impacto_consolidado(lista_nf: List[str],
                        loads: Dict[str, float],
                        nf_map_data: Dict[str, Dict]) -> Tuple[float, int, List[str]]:
    """
    Impacto consolidado de uma sequência de NFs (modo 1):
      - união das barras isoladas por todas as NFs;
      - soma dos kW por barra.
    """
    barras_afetadas = set()
    for nf in lista_nf:
        reg = nf_map_data.get(nf)
        if not reg:
            continue
        barras_afetadas |= reg["barras"]

    kw_total = sum(loads.get(b, 0.0) for b in barras_afetadas)
    barras_ordenadas = sorted(
        barras_afetadas,
        key=lambda x: (x.isdigit(), int(x) if x.isdigit() else x)
    )
    return kw_total, len(barras_afetadas), barras_ordenadas


# =========================================================
# FUNÇÕES DE PROCESSAMENTO – MODO 2
# =========================================================
def trecho_barras_por_linhas(
    linhas: List[str],
    topo: List[Dict]
) -> Tuple[set, List[Tuple[str, str]]]:
    """
    Dado uma lista de linhas (ex: ["l70","l71"]),
    retorna:
      - conjunto de barras presentes no trecho;
      - lista de vãos (from_bus, to_bus) para plot.
    """
    buses = set()
    vaos = []
    topo_por_line = {el["line"].lower(): el for el in topo}

    for ln in linhas:
        key = ln.strip().lower()
        el = topo_por_line.get(key)
        if not el:
            continue
        u = str(el["from_bus"])
        v = str(el["to_bus"])
        buses.add(u)
        buses.add(v)
        vaos.append((u, v))

    return buses, vaos


def get_kw_base_nf(nf1: str, combos: List[Dict]) -> Optional[float]:
    """
    Encontra o cenário base (apenas NF1, sem NA e sem NF_block)
    para recuperar o impacto da NF isolada.
    """
    for reg in combos:
        if reg["nf1"] == nf1 and reg["na"] is None and reg["nf_block"] is None:
            return reg["kw_off"]
    return None


def encontrar_opcoes_modo2(
    trecho_buses: set,
    combos: List[Dict],
    max_options: int = 5,
) -> List[Dict]:
    """
    Usa o banco nf_na_nf:
      - filtra cenários que desligam todas as barras do trecho;
      - ordena por (kw_off, n_manobras, kw_base_nf).
    """
    if not trecho_buses:
        return []

    valid = []
    for reg in combos:
        buses_off = set(reg["buses_off"])
        if not trecho_buses.issubset(buses_off):
            continue

        kw_base = get_kw_base_nf(reg["nf1"], combos)
        reg = reg.copy()
        reg["kw_base_nf1"] = kw_base if kw_base is not None else reg["kw_off"]
        valid.append(reg)

    valid_sorted = sorted(
        valid,
        key=lambda r: (r["kw_off"], r["n_manobras"], r["kw_base_nf1"])
    )

    # remove duplicatas por (nf1, na, nf_block)
    seen = set()
    out = []
    for reg in valid_sorted:
        key = (reg["nf1"], reg["na"], reg["nf_block"])
        if key in seen:
            continue
        seen.add(key)
        out.append(reg)
        if len(out) >= max_options:
            break

    return out


# =========================================================
# FUNÇÕES DE PLOT – MAPA BASE + DESTAQUES
# =========================================================
def construir_mapa_base(coords, topo):
    """
    Retorna uma figura Plotly com a topologia base (sem destaques),
    mostrando:
      - barras com nome;
      - linhas em cinza;
      - nomes das linhas em fonte pequena, centralizados em cada segmento.
    """
    fig = go.Figure()

    # --- Linhas base ---
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

    # --- Nós (barras) ---
    node_x = [coords[b][0] for b in coords]
    node_y = [coords[b][1] for b in coords]
    node_text = list(coords.keys())

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

    # --- Rótulos das linhas (nomes das linhas) ---
    label_x = []
    label_y = []
    label_text = []

    for el in topo:
        ln = el["line"]
        u = el["from_bus"]
        v = el["to_bus"]
        if u not in coords or v not in coords:
            continue

        x0, y0 = coords[u]
        x1, y1 = coords[v]
        xm = (x0 + x1) / 2
        ym = (y0 + y1) / 2

        dx = x1 - x0
        dy = y1 - y0

        # desloca um pouco o texto para não sobrepor a barra
        if abs(dx) >= abs(dy):
            # linha mais horizontal -> desloca um pouco no eixo Y
            ym = ym + 10
        else:
            # linha mais vertical -> desloca um pouco no eixo X
            xm = xm + 10

        label_x.append(xm)
        label_y.append(ym)
        label_text.append(ln)

    fig.add_trace(
        go.Scatter(
            x=label_x,
            y=label_y,
            mode="text",
            text=label_text,
            textposition="middle center",
            textfont=dict(size=8, color="gray"),
            showlegend=False,
            hoverinfo="none",
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


def plotar_mapa_modo1(
    coords,
    topo,
    vaos: List[Tuple[str, str]],
    info_vaos: List[Dict],
):
    """
    Mapa para o MODO 1:
      - vãos selecionados em preto;
      - NFs associadas em vermelho tracejado;
      - rótulos de linhas e barras do mapa base.
    """
    fig = construir_mapa_base(coords, topo)

    # --- Destaque dos vãos selecionados ---
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

    topo_por_line = {el["line"]: el for el in topo}

    # --- Destaque das NF associadas ---
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


def plotar_mapa_modo2(
    coords,
    topo,
    vaos: List[Tuple[str, str]],
    scenario: Dict,
):
    """
    Mapa para o MODO 2:
      - vãos do trecho em preto;
      - NF1 (isoladora) em vermelho;
      - NA em ciano;
      - NF_block em laranja;
      - labels de linhas e barras do mapa base.
    """
    fig = construir_mapa_base(coords, topo)
    topo_por_line = {el["line"]: el for el in topo}

    # --- Trecho (vãos selecionados) ---
    edge_x = []
    edge_y = []

    for u, v in vaos:
        if u in coords and v in coords:
            x0, y0 = coords[u]
            x1, y1 = coords[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

    if edge_x:
        fig.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line=dict(color="black", width=4),
                name="Trecho selecionado (linhas)",
                hoverinfo="none",
            )
        )

    # --- NF1 ---
    nf1 = scenario["nf1"]
    if nf1 in topo_por_line:
        el = topo_por_line[nf1]
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
                    line=dict(color="red", width=4),
                    name=f"NF1 (isoladora): {nf1}",
                    hoverinfo="none",
                )
            )

    # --- NA ---
    na = scenario["na"]
    if na and na in topo_por_line:
        el = topo_por_line[na]
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
                    line=dict(color="cyan", width=4, dash="dot"),
                    name=f"NA (restabelecimento): {na}",
                    hoverinfo="none",
                )
            )

    # --- NF_block ---
    nf_block = scenario["nf_block"]
    if nf_block and nf_block in topo_por_line:
        el = topo_por_line[nf_block]
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
                    line=dict(color="orange", width=4, dash="dash"),
                    name=f"NF bloqueio: {nf_block}",
                    hoverinfo="none",
                )
            )

    return fig


# =========================================================
# CARREGAMENTO DO BANCO E STATUS
# =========================================================
st.sidebar.header("📂 Banco de dados")

if not DB_PATH.exists():
    st.sidebar.error(f"Banco `{DB_PATH.name}` não encontrado na pasta do app.")
    st.stop()

tabelas = listar_tabelas()
st.sidebar.write("Banco:", f"`{DB_PATH.name}`")
st.sidebar.write("COORDS:", "✅" if "coords" in tabelas else "❌")
st.sidebar.write("TOPOLOGY:", "✅" if "topology" in tabelas else "❌")
st.sidebar.write("LOADS:", "✅" if "loads" in tabelas else "❌")
st.sidebar.write("VAO_MAP:", "✅" if "vao_map" in tabelas else "❌")
st.sidebar.write("NF_MAP:", "✅" if "nf_map" in tabelas else "❌")
st.sidebar.write("NF_NA_NF:", "✅" if "nf_na_nf" in tabelas else "❌")

coords = carregar_coords()
topo = carregar_topologia()
loads = carregar_loads()
vao_map = carregar_vao_map() if "vao_map" in tabelas else []
nf_map_data = carregar_nf_map() if "nf_map" in tabelas else {}
nf_na_nf_data = carregar_nf_na_nf() if "nf_na_nf" in tabelas else []

if not coords or not topo:
    st.error("Banco encontrado, mas `coords` ou `topology` estão vazios.")
    st.stop()

# Campo para nome do operador (para aparecer nos relatórios)
st.sidebar.subheader("👨‍💼 Operador")
nome_operador = st.sidebar.text_input(
    "Nome do operador:",
    value="",
    help="Esse nome será exibido nos relatórios de manobra."
)


# =========================================================
# DESCRIÇÃO RÁPIDA
# =========================================================
st.markdown(
    """
Ferramenta de apoio à **manobra de desligamento programado** em redes de distribuição,
baseada no alimentador teste **IEEE-123 Bus**.

Toda a inteligência (impacto de chaves NF, combinações NF–NA–NF, barras/linhas desenergizadas,
tensões e carregamento máximo) foi calculada previamente em **OpenDSS + Python (Colab)**
e armazenada em um banco **SQLite** único (`ieee123_manobras.db`).

A plataforma possui dois modos:

- 🟢 **Modo 1 – Fonte única:** usa as tabelas `vao_map` e `nf_map`  
- 🟣 **Modo 2 – Duas fontes (NF–NA–NF):** usa a tabela `nf_na_nf`  
"""
)

st.markdown("---")


# =========================================================
# MAPA BASE ESTÁTICO
# =========================================================
st.subheader("🗺️ Mapa Base da Rede (IEEE-123 Bus) – Barras + Linhas (com nomes)")
fig_base = construir_mapa_base(coords, topo)
st.plotly_chart(fig_base, use_container_width=True)

st.markdown("---")


# =========================================================
# SELEÇÃO DE MODO
# =========================================================
tab_modo1, tab_modo2 = st.tabs(
    ["🟢 Modo 1 – Fonte única", "🟣 Modo 2 – Duas fontes"]
)


# =========================================================
# MODO 1 – FONTE ÚNICA (VAO_MAP + NF_MAP)
# =========================================================
with tab_modo1:
    st.markdown("### 🟢 Modo 1 – Fonte única (NF ótima por vão)")

    if not vao_map or not nf_map_data:
        st.warning(
            "Tabelas `vao_map` e/ou `nf_map` não disponíveis no banco. "
            "Modo 1 indisponível."
        )
    else:
        st.sidebar.subheader("🔧 Modo 1 – Vão simples (U–V)")
        lista_barras = sorted(
            coords.keys(),
            key=lambda x: (x.isdigit(), int(x) if x.isdigit() else x),
        )

        u_simples = st.sidebar.selectbox(
            "Barra U (Modo 1)", lista_barras, key="u_simples_m1"
        )
        v_simples = st.sidebar.selectbox(
            "Barra V (Modo 1)", lista_barras, key="v_simples_m1"
        )

        if st.sidebar.button("Confirmar vão simples (Modo 1)"):
            vao_simples = (u_simples, v_simples)
            info = buscar_nf_para_vao(u_simples, v_simples, vao_map)

            st.subheader("🔎 Resultado – Vão simples (Modo 1)")

            if nome_operador:
                st.info(f"Operador responsável: **{nome_operador}**")

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

        # --------- TRECHO MULTI-VÃOS ---------
        st.markdown("### 🧩 Trecho com múltiplos vãos (entrada em blocos de 2 barras)")

        entrada_seq = st.text_input(
            "Sequência de barras (ex: 60,62,63,64,65,66,60,67) – Modo 1",
            value="60,62,63,64,65,66,60,67",
        )

        if st.button("Processar Trecho (Multi-Vãos) – Modo 1"):
            barras_raw = [b.strip() for b in entrada_seq.split(",") if b.strip()]
            if len(barras_raw) < 2:
                st.error("Informe pelo menos duas barras.")
            else:
                vaos = identificar_vaos_blocos(barras_raw)

                if not vaos:
                    st.error("Nenhum vão pôde ser formado com a sequência informada.")
                else:
                    st.markdown("### 🔍 Vãos identificados (blocos de 2 barras):")
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

                    if nome_operador:
                        st.info(f"Operador responsável: **{nome_operador}**")

                    if nao_encontrados:
                        st.warning(
                            "Não foram encontrados registros para os seguintes vãos: "
                            + ", ".join([f"{u}-{v}" for u, v in nao_encontrados])
                        )

                    if info_vaos:
                        st.markdown("### ✅ NF de manobra por vão (impacto individual)")

                        df_data = [
                            {
                                "Vão (U-V)": f"{d['u']} - {d['v']}",
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
                        st.markdown("### 🗺️ Mapa com trecho e NFs destacadas")
                        st.plotly_chart(fig_multi, use_container_width=True)

                        # Impacto consolidado
                        st.markdown(
                            "### ⚡ Impacto consolidado da manobra (sem dupla contagem)"
                        )

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
                            kw_total, n_barras_unicas, barras_ordenadas = impacto_consolidado(
                                lista_nf_ordenada, loads, nf_map_data
                            )

                            st.success(
                                f"**Carga total interrompida:** {kw_total:.1f} kW  \n"
                                f"**Barras desenergizadas únicas:** {n_barras_unicas}"
                            )

                            with st.expander("Ver barras desenergizadas únicas"):
                                st.write(barras_ordenadas)

                        st.markdown("### 📜 Linha de tempo de manobra (sequência sugerida)")
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


# =========================================================
# MODO 2 – DUAS FONTES (NF–NA–NF)
# =========================================================
with tab_modo2:
    st.markdown("### 🟣 Modo 2 – Duas fontes (combinações NF–NA–NF)")

    if not nf_na_nf_data:
        st.warning(
            "Tabela `nf_na_nf` não disponível no banco. "
            "Modo 2 indisponível."
        )
    else:
        st.markdown(
            """
No **Modo 2**, o banco `nf_na_nf` armazena o resultado de todas as combinações
de chaves **NF–NA–NF**, incluindo:

- barras desligadas (`buses_off`)  
- linhas desligadas (`lines_off`)  
- carga total interrompida (`kw_off`)  
- tensão mínima/máxima (`vmin_pu`, `vmax_pu`)  
- carregamento máximo (`max_loading`)  
- número de manobras (`n_manobras`)  
- estados das chaves (`switch_states`)  
"""
        )

        st.markdown("#### 🎯 Entrada do trecho (por linhas Lxx, swx, l108...)")
        exemplo_txt = "L70"  # exemplo simples
        linhas_input = st.text_input(
            "Linhas do trecho (ex: L70 ou L70,L71):",
            value=exemplo_txt,
            help="Use nomes de linhas exatamente como no mapa (Lxx, swx, l108, etc.).",
        )

        if st.button("Processar trecho – Modo 2"):
            # Normaliza nomes de linhas
            linhas_trecho = [
                ln.strip() for ln in linhas_input.split(",") if ln.strip()
            ]

            if not linhas_trecho:
                st.error("Informe pelo menos uma linha.")
            else:
                # Determina conjunto de barras do trecho e vãos para plot
                trecho_buses, vaos_m2 = trecho_barras_por_linhas(
                    linhas_trecho, topo
                )

                if not trecho_buses:
                    st.error(
                        "Nenhuma linha informada foi encontrada na tabela de topologia."
                    )
                else:
                    st.markdown("#### 🔍 Trecho considerado (Modo 2)")
                    st.write(f"**Linhas do trecho:** {linhas_trecho}")
                    st.write(f"**Barras do trecho:** {sorted(trecho_buses)}")

                    if nome_operador:
                        st.info(f"Operador responsável: **{nome_operador}**")

                    # Busca opções no banco nf_na_nf
                    opcoes = encontrar_opcoes_modo2(
                        trecho_buses, nf_na_nf_data, max_options=5
                    )

                    if not opcoes:
                        st.error(
                            "Nenhuma combinação NF–NA–NF encontrada que desligue "
                            "exatamente todas as barras do trecho informado."
                        )
                    else:
                        st.markdown(
                            "### ✅ TOP opções de manobra (ordenadas por menor carga desligada)"
                        )

                        # Tabela resumida
                        tabela_opcoes = []
                        for i, opt in enumerate(opcoes, start=1):
                            tabela_opcoes.append(
                                {
                                    "Opção": i,
                                    "NF isoladora": opt["nf1"],
                                    "NA restabelecimento": opt["na"] or "-",
                                    "NF bloqueio": opt["nf_block"] or "-",
                                    "Carga desligada [kW]": opt["kw_off"],
                                    "Nº manobras": opt["n_manobras"],
                                    "Vmin [pu]": opt["vmin_pu"],
                                    "Vmax [pu]": opt["vmax_pu"],
                                    "Carregamento máx. [pu]": opt["max_loading"],
                                    "Impacto NF isoladora (kW)": opt["kw_base_nf1"],
                                }
                            )

                        st.table(tabela_opcoes)

                        # Escolha de uma opção para detalhar
                        idx_escolha = st.number_input(
                            "Escolha o número da opção para detalhar (1, 2, 3, ...):",
                            min_value=1,
                            max_value=len(opcoes),
                            value=1,
                            step=1,
                        )
                        opt_sel = opcoes[idx_escolha - 1]

                        st.markdown("### 📜 Detalhamento da opção escolhida")

                        st.write(
                            f"- **NF isoladora:** `{opt_sel['nf1']}`  \n"
                            f"- **NA restabelecimento:** `{opt_sel['na'] or '-'}`  \n"
                            f"- **NF bloqueio:** `{opt_sel['nf_block'] or '-'}`  \n"
                            f"- **Carga desligada:** `{opt_sel['kw_off']:.2f} kW`  \n"
                            f"- **Nº manobras:** `{opt_sel['n_manobras']}`  \n"
                            f"- **Impacto NF isoladora (sem NA):** `{opt_sel['kw_base_nf1']:.2f} kW`  \n"
                            f"- **Vmin / Vmax [pu]:** `{opt_sel['vmin_pu']:.3f} / {opt_sel['vmax_pu']:.3f}`  \n"
                            f"- **Carregamento máximo [pu]:** `{opt_sel['max_loading']:.3f}`  \n"
                        )

                        if nome_operador:
                            st.write(f"👤 **Operador:** {nome_operador}")

                        # Mapa da opção
                        st.markdown("### 🗺️ Mapa da opção escolhida (NF–NA–NF)")
                        fig_m2 = plotar_mapa_modo2(
                            coords,
                            topo,
                            vaos_m2,
                            opt_sel,
                        )
                        st.plotly_chart(fig_m2, use_container_width=True)

                        # Listagem de barras e linhas desligadas
                        st.markdown("#### 📌 Barras desenergizadas no cenário:")
                        st.write(sorted(set(opt_sel["buses_off"])))

                        st.markdown("#### 📌 Linhas desligadas no cenário:")
                        st.write(sorted(set(opt_sel["lines_off"])))

                        # Linha do tempo simplificada
                        st.markdown("### ⏱️ Linha de tempo de manobra – Modo 2")
                        st.markdown(
                            f"1️⃣ Abrir NF isoladora **{opt_sel['nf1']}** para iniciar o isolamento do trecho."
                        )
                        if opt_sel["na"]:
                            st.markdown(
                                f"2️⃣ Fechar NA **{opt_sel['na']}** para restabelecer o maior número possível de cargas."
                            )
                            if opt_sel["nf_block"]:
                                st.markdown(
                                    f"3️⃣ Abrir NF de bloqueio **{opt_sel['nf_block']}** "
                                    "para evitar dupla alimentação em trechos da rede."
                                )
                        elif opt_sel["nf_block"]:
                            # Caso raro: NF_block sem NA
                            st.markdown(
                                f"2️⃣ Abrir NF de bloqueio **{opt_sel['nf_block']}** "
                                "para garantir seletividade da manobra."
                            )

                        st.markdown(
                            "✅ Após conclusão da manutenção, **retornar as chaves ao estado nominal**, "
                            "conforme procedimento operacional da distribuidora."
                        )
