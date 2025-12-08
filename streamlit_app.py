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
    page_title="Isolamento Real IEEE-123 Bus – Modo 1 & Modo 2",
    layout="wide"
)

st.title("⚡ Plataforma Interativa – Isolamento Real IEEE 123 Bus")

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "ieee123_manobras.db"

# Lista de chaves NF/NA (para exibição e lógica de negócio)
NF_LIST = ["sw1", "sw2", "sw3", "sw4", "sw5", "sw6"]
NA_LIST = ["sw7", "sw8", "l108"]


# =========================================================
# FUNÇÕES AUXILIARES – BANCO
# =========================================================
def get_connection() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


@st.cache_data(show_spinner=False)
def carregar_coords() -> Dict[str, Tuple[float, float]]:
    """Lê tabela coords(bus, x, y) do banco."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT bus, x, y FROM coords")
    rows = cur.fetchall()
    conn.close()
    return {str(b): (float(x), float(y)) for b, x, y in rows}


@st.cache_data(show_spinner=False)
def carregar_topologia():
    """
    Lê tabela topology(line, from_bus, to_bus, is_switch, norm)
    do banco.
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
def carregar_vao_map():
    """
    Lê tabela vao_map(u_bus, v_bus, nf, kw, n_barras).
    Um registro por vão contendo a NF ótima (MODO 1).
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
def carregar_loads():
    """Tabela loads(bus, kw)."""
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
def carregar_nf_map():
    """
    Tabela nf_map(nf, barras_isoladas TEXT, kw REAL, n_barras INTEGER).

    barras_isoladas está salva como uma string de lista Python,
    ex.: '["33", "61", "18", ...]'.
    """
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT nf, barras_isoladas, kw, n_barras FROM nf_map"
        )
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
                for b in (
                    str(barras_str)
                    .replace("[", "")
                    .replace("]", "")
                    .replace('"', "")
                    .split(",")
                ):
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
def carregar_nf_na_nf():
    """
    Tabela nf_na_nf:
      id, nf1, na, nf_block,
      buses_off, lines_off,
      kw_off, vmin_pu, vmax_pu, max_loading,
      n_manobras, switch_states
    Usada no MODO 2 (duas fontes).
    """
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT id, nf1, na, nf_block,
                   buses_off, lines_off,
                   kw_off, vmin_pu, vmax_pu, max_loading,
                   n_manobras, switch_states
            FROM nf_na_nf
            """
        )
        rows = cur.fetchall()
    except sqlite3.OperationalError:
        rows = []
    conn.close()

    registros = []

    def norm_opt(x):
        if x is None:
            return None
        xs = str(x).strip()
        if xs.lower() in ("none", "null", ""):
            return None
        return xs

    for (
        _id,
        nf1,
        na,
        nf_block,
        buses_str,
        lines_str,
        kw_off,
        vmin_pu,
        vmax_pu,
        max_loading,
        n_manobras,
        switch_states,
    ) in rows:
        # parse listas
        buses_list: List[str] = []
        lines_list: List[str] = []
        if buses_str:
            try:
                buses_list = [str(b).strip() for b in ast.literal_eval(buses_str)]
            except Exception:
                buses_list = [
                    s.strip()
                    for s in str(buses_str)
                    .replace("[", "")
                    .replace("]", "")
                    .replace('"', "")
                    .split(",")
                    if s.strip()
                ]
        if lines_str:
            try:
                lines_list = [str(l).strip() for l in ast.literal_eval(lines_str)]
            except Exception:
                lines_list = [
                    s.strip()
                    for s in str(lines_str)
                    .replace("[", "")
                    .replace("]", "")
                    .replace('"', "")
                    .split(",")
                    if s.strip()
                ]

        registros.append(
            dict(
                id=int(_id),
                nf1=str(nf1),
                na=norm_opt(na),
                nf_block=norm_opt(nf_block),
                buses_off=set(buses_list),
                lines_off=set(lines_list),
                kw_off=float(kw_off),
                vmin_pu=float(vmin_pu),
                vmax_pu=float(vmax_pu),
                max_loading=float(max_loading),
                n_manobras=int(n_manobras),
                switch_states=str(switch_states) if switch_states is not None else "",
            )
        )

    return registros


def listar_tabelas() -> List[str]:
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    rows = [r[0] for r in cur.fetchall()]
    conn.close()
    return rows


# =========================================================
# FUNÇÕES – PROCESSAMENTO MODO 1
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


def buscar_nf_para_vao(u: str, v: str, vao_map: List[Dict]) -> Optional[Dict]:
    """
    Procura no vao_map a NF ótima para o vão (u, v),
    considerando que o usuário pode informar em qualquer ordem.
    """
    candidatos = [
        registro
        for registro in vao_map
        if (registro["u_bus"] == u and registro["v_bus"] == v)
        or (registro["u_bus"] == v and registro["v_bus"] == u)
    ]
    if not candidatos:
        return None

    # menor kW depois menos barras
    candidatos.sort(key=lambda r: (r["kw"], r["n_barras"]))
    return candidatos[0]


def obter_barras_unicas(vaos: List[Tuple[str, str]]) -> List[str]:
    """Retorna a lista de barras únicas presentes em uma lista de vãos."""
    s = set()
    for u, v in vaos:
        s.add(u)
        s.add(v)
    return sorted(s, key=lambda x: (x.isdigit(), int(x) if x.isdigit() else x))


def impacto_consolidado(
    lista_nf: List[str],
    loads: Dict[str, float],
    nf_map_data: Dict[str, Dict],
) -> Tuple[float, int, List[str]]:
    """
    Calcula o impacto consolidado de uma manobra que envolve
    várias NFs, **sem dupla contagem** de carga:

      - união das barras isoladas por todas as NFs;
      - soma dos kW por barra (usando tabela loads).
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
        key=lambda x: (x.isdigit(), int(x) if x.isdigit() else x),
    )
    return kw_total, len(barras_afetadas), barras_ordenadas


# =========================================================
# FUNÇÕES – PROCESSAMENTO MODO 2 (NF–NA–NF via BANCO)
# =========================================================
def encontrar_linhas_por_barras(
    topo: List[Dict], u: str, v: str
) -> List[str]:
    """
    Retorna lista de nomes de linhas (campo 'line' em topology)
    que ligam as barras u e v (em qualquer ordem).
    """
    resultado = []
    for el in topo:
        bu = el["from_bus"]
        bv = el["to_bus"]
        if (bu == u and bv == v) or (bu == v and bv == u):
            resultado.append(el["line"])
    return resultado


def montar_trecho_buses_de_linhas(
    topo: List[Dict], linhas: List[str]
) -> Tuple[set, List[Tuple[str, str]]]:
    """
    A partir de uma lista de nomes de linhas (ex: ["l70","l71"]),
    retorna:
      - conjunto de barras envolvidas (trecho_buses)
      - lista de pares (from_bus, to_bus) para plot.
    """
    trecho_buses = set()
    vaos = []
    topo_por_line = {el["line"]: el for el in topo}
    for ln in linhas:
        el = topo_por_line.get(ln)
        if not el:
            continue
        u = el["from_bus"]
        v = el["to_bus"]
        trecho_buses.add(u)
        trecho_buses.add(v)
        vaos.append((u, v))
    return trecho_buses, vaos


def construir_nf_base_from_nf_na_nf(nf_na_nf_regs: List[Dict]) -> Dict[str, Dict]:
    """
    Constrói mapa:
      nf -> cenário base (apenas NF1 aberta, sem NA e sem NF_block)
    a partir da tabela nf_na_nf.
    """
    base: Dict[str, Dict] = {}
    for reg in nf_na_nf_regs:
        nf1 = reg["nf1"]
        na = reg["na"]
        nf_block = reg["nf_block"]
        if na is None and nf_block is None:
            # cenário NF isolada
            if nf1 not in base or reg["kw_off"] < base[nf1]["kw_off"]:
                base[nf1] = reg
    return base


def buscar_opcoes_modo2(
    trecho_buses: set,
    nf_na_nf_regs: List[Dict],
) -> List[Dict]:
    """
    Replica a lógica do script de duas fontes usando
    o banco nf_na_nf:

      1) Encontra NF que isolam o trecho (buses_off da NF sozinha
         contém todas as barras do trecho).
      2) Para cada NF candidata:
         - cenário base (NF1 apenas)
         - cenários com NA (+ opcional NF_block) que:
             * mantêm o trecho desenergizado
             * reduzem a carga desligada (kw_off < kw_off_base_nf1)
      3) Ordena por:
           kw_off, n_manobras, kw_off_base_nf1
    """
    if not trecho_buses:
        return []

    nf_base = construir_nf_base_from_nf_na_nf(nf_na_nf_regs)
    if not nf_base:
        return []

    # indexar por nf1 para acesso rápido
    por_nf1: Dict[str, List[Dict]] = {}
    for reg in nf_na_nf_regs:
        nf1 = reg["nf1"]
        por_nf1.setdefault(nf1, []).append(reg)

    opcoes = []

    for nf1, base_reg in nf_base.items():
        base_buses_off = base_reg["buses_off"]
        # checa se NF1 sozinha isola o trecho
        if not trecho_buses.issubset(base_buses_off):
            continue

        kw_off_base = base_reg["kw_off"]

        # Cenário A: NF1 apenas
        scen_base = {
            **base_reg,
            "kw_off_base_nf": kw_off_base,
        }
        opcoes.append(scen_base)

        # Cenários com NA e/ou NF_block
        for reg in por_nf1.get(nf1, []):
            if reg is base_reg:
                continue

            # Mantém trecho desenergizado?
            if not trecho_buses.issubset(reg["buses_off"]):
                continue

            # Precisa melhorar em relação à NF apenas
            if reg["kw_off"] >= kw_off_base:
                continue

            reg2 = {
                **reg,
                "kw_off_base_nf": kw_off_base,
            }
            opcoes.append(reg2)

    # ordena e remove duplicatas por (nf1, na, nf_block)
    opcoes_ordenadas = sorted(
        opcoes,
        key=lambda o: (o["kw_off"], o["n_manobras"], o["kw_off_base_nf"]),
    )

    uniq = []
    seen = set()
    for o in opcoes_ordenadas:
        key = (o["nf1"], o["na"], o["nf_block"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(o)

    return uniq


# =========================================================
# FUNÇÕES DE PLOT – MAPA BASE E DESTAQUES
# =========================================================
def construir_mapa_base(coords: Dict[str, Tuple[float, float]], topo: List[Dict]):
    """
    Retorna uma figura Plotly com a topologia base (sem destaques).
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

    fig.update_layout(
        height=600,
        showlegend=True,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )

    return fig


def adicionar_rotulos_de_linhas(
    fig: go.Figure,
    coords: Dict[str, Tuple[float, float]],
    topo_por_line: Dict[str, Dict],
    linhas: List[str],
    cor_texto: str = "black",
    nome_traco: str = "Linhas",
):
    """
    Adiciona rótulos com o nome das linhas no ponto médio de cada uma,
    com fonte pequena e levemente acima (para não ficar sobreposto).
    """
    xs = []
    ys = []
    texts = []

    for ln in sorted(set(linhas)):
        el = topo_por_line.get(ln)
        if not el:
            continue
        u = el["from_bus"]
        v = el["to_bus"]
        if u not in coords or v not in coords:
            continue
        x0, y0 = coords[u]
        x1, y1 = coords[v]
        xm = (x0 + x1) / 2.0
        ym = (y0 + y1) / 2.0
        xs.append(xm)
        ys.append(ym)
        texts.append(ln)

    if xs:
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="text",
                text=texts,
                textposition="top center",
                textfont=dict(color=cor_texto, size=9),
                showlegend=False,
                name=nome_traco,
                hoverinfo="skip",
            )
        )


def plotar_mapa_modo1(
    coords,
    topo,
    vaos: List[Tuple[str, str]],
    info_vaos: List[Dict],
):
    """
    MODO 1:
      - vãos selecionados (linhas pretas grossas)
      - NFs associadas em vermelho (tracejado)
      - rótulos de linhas (vãos + NFs)
    """
    fig = construir_mapa_base(coords, topo)
    topo_por_line = {el["line"]: el for el in topo}

    # --- Destaque dos vãos (linhas pretas) ---
    edge_x = []
    edge_y = []
    linhas_vaos = []

    for u, v in vaos:
        # tentar achar linha correspondente (se existir)
        for ln, el in topo_por_line.items():
            if (el["from_bus"] == u and el["to_bus"] == v) or (
                el["from_bus"] == v and el["to_bus"] == u
            ):
                linhas_vaos.append(ln)
                if u in coords and v in coords:
                    x0, y0 = coords[u]
                    x1, y1 = coords[v]
                    edge_x += [x0, x1, None]
                    edge_y += [y0, y1, None]
                break

    if edge_x:
        fig.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line=dict(color="black", width=4),
                name="Trecho selecionado (vãos)",
                hoverinfo="none",
            )
        )

    # --- Destaque das NFs ---
    nf_edges_x = []
    nf_edges_y = []
    nf_labels_lines = []

    for info in info_vaos:
        nf = info["nf"]
        el = topo_por_line.get(nf)
        if not el:
            continue
        u = el["from_bus"]
        v = el["to_bus"]
        if u in coords and v in coords:
            x0, y0 = coords[u]
            x1, y1 = coords[v]
            nf_edges_x += [x0, x1, None]
            nf_edges_y += [y0, y1, None]
            nf_labels_lines.append(nf)

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

    # --- Rótulos de linhas (vãos + NFs) ---
    linhas_para_rotulo = set(linhas_vaos) | set(nf_labels_lines)
    adicionar_rotulos_de_linhas(
        fig,
        coords,
        topo_por_line,
        list(linhas_para_rotulo),
        cor_texto="black",
        nome_traco="Rótulos de linhas",
    )

    return fig


def plotar_mapa_modo2(
    coords,
    topo,
    trecho_linhas: List[str],
    opcao: Dict,
):
    """
    MODO 2:
      - Linhas do trecho em preto grosso
      - NF1 em vermelho
      - NA em azul
      - NF_block em laranja
      - Rótulos de linhas relevantes (trecho + NF1 + NA + NF_block)
    """
    fig = construir_mapa_base(coords, topo)
    topo_por_line = {el["line"]: el for el in topo}

    # ---- Linhas do trecho (preto) ----
    edge_x = []
    edge_y = []
    for ln in trecho_linhas:
        el = topo_por_line.get(ln)
        if not el:
            continue
        u = el["from_bus"]
        v = el["to_bus"]
        if u not in coords or v not in coords:
            continue
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
                name="Trecho (linhas selecionadas)",
                hoverinfo="none",
            )
        )

    # ---- NF1 (vermelho) ----
    nf1 = opcao["nf1"]
    nf1_x = []
    nf1_y = []
    if nf1 in topo_por_line:
        el = topo_por_line[nf1]
        u = el["from_bus"]
        v = el["to_bus"]
        if u in coords and v in coords:
            x0, y0 = coords[u]
            x1, y1 = coords[v]
            nf1_x += [x0, x1, None]
            nf1_y += [y0, y1, None]

    if nf1_x:
        fig.add_trace(
            go.Scatter(
                x=nf1_x,
                y=nf1_y,
                mode="lines",
                line=dict(color="red", width=4),
                name=f"NF isoladora ({nf1})",
                hoverinfo="none",
            )
        )

    # ---- NA (azul) ----
    na = opcao["na"]
    na_x = []
    na_y = []
    if na and na in topo_por_line:
        el = topo_por_line[na]
        u = el["from_bus"]
        v = el["to_bus"]
        if u in coords and v in coords:
            x0, y0 = coords[u]
            x1, y1 = coords[v]
            na_x += [x0, x1, None]
            na_y += [y0, y1, None]

    if na_x:
        fig.add_trace(
            go.Scatter(
                x=na_x,
                y=na_y,
                mode="lines",
                line=dict(color="blue", width=4, dash="dot"),
                name=f"NA restabelecimento ({na})",
                hoverinfo="none",
            )
        )

    # ---- NF bloqueio (laranja) ----
    nf_block = opcao["nf_block"]
    nb_x = []
    nb_y = []
    if nf_block and nf_block in topo_por_line:
        el = topo_por_line[nf_block]
        u = el["from_bus"]
        v = el["to_bus"]
        if u in coords and v in coords:
            x0, y0 = coords[u]
            x1, y1 = coords[v]
            nb_x += [x0, x1, None]
            nb_y += [y0, y1, None]

    if nb_x:
        fig.add_trace(
            go.Scatter(
                x=nb_x,
                y=nb_y,
                mode="lines",
                line=dict(color="orange", width=4, dash="dash"),
                name=f"NF bloqueio ({nf_block})",
                hoverinfo="none",
            )
        )

    # ---- Rótulos de linhas (trecho + NF1 + NA + NF_block) ----
    linhas_rotulo = set(trecho_linhas)
    linhas_rotulo.add(nf1)
    if na:
        linhas_rotulo.add(na)
    if nf_block:
        linhas_rotulo.add(nf_block)

    adicionar_rotulos_de_linhas(
        fig,
        coords,
        topo_por_line,
        list(linhas_rotulo),
        cor_texto="black",
        nome_traco="Rótulos de linhas (M2)",
    )

    return fig


# =========================================================
# CARREGAMENTO DO BANCO E STATUS
# =========================================================
st.sidebar.header("📂 Dados carregados")

if not DB_PATH.exists():
    st.sidebar.error("Banco ieee123_isolamento.db não encontrado na pasta do app.")
    st.stop()

tabelas = listar_tabelas()
st.sidebar.write("Banco:", f"`{DB_PATH.name}`")
st.sidebar.write("TOPOLOGY:", "✅" if "topology" in tabelas else "❌")
st.sidebar.write("COORDS:", "✅" if "coords" in tabelas else "❌")
st.sidebar.write("LOADS:", "✅" if "loads" in tabelas else "❌")
st.sidebar.write("VAO_MAP:", "✅" if "vao_map" in tabelas else "❌")
st.sidebar.write("NF_MAP:", "✅" if "nf_map" in tabelas else "❌")
st.sidebar.write("NF_NA_NF:", "✅" if "nf_na_nf" in tabelas else "❌")

coords = carregar_coords()
topo = carregar_topologia()
vao_map = carregar_vao_map()
loads = carregar_loads()
nf_map_data = carregar_nf_map()
nf_na_nf_regs = carregar_nf_na_nf()

if not coords or not topo:
    st.error("Banco encontrado, mas alguma tabela essencial está vazia.")
    st.stop()

# =========================================================
# CAMPO: NOME DO OPERADOR
# =========================================================
st.sidebar.markdown("---")
nome_operador = st.sidebar.text_input(
    "👤 Nome do operador (para relatório):",
    value="",
    help="Este nome aparecerá nos relatórios exibidos na tela.",
)


# =========================================================
# DESCRIÇÃO GERAL
# =========================================================
st.markdown(
    """
Ferramenta de apoio à manobra de **desligamento programado** em redes de distribuição,
baseada no alimentador teste **IEEE-123 Bus**.

Toda a inteligência de manobra foi calculada anteriormente em
**OpenDSS + Python (Colab)** e gravada em um banco **SQLite** (`ieee123_isolamento.db`).

A plataforma possui dois modos:

- 🟢 **MODO 1 – Fonte Única**: melhor NF para isolar vãos (U–V) com banco `vao_map` / `nf_map`  
- 🟣 **MODO 2 – Duas Fontes**: melhor combinação **NF–NA–NF** com base no banco `nf_na_nf`
"""
)

st.markdown("---")

# =========================================================
# MAPA BASE
# =========================================================
st.subheader("🗺️ Mapa Base da Rede (IEEE-123 Bus)")
fig_base = construir_mapa_base(coords, topo)
st.plotly_chart(fig_base, use_container_width=True)

st.markdown("---")

# =========================================================
# MODO 1 – FONTE ÚNICA
# =========================================================
st.header("🟢 MODO 1 – Isolamento com Fonte Única (NF por Vão)")

if nome_operador:
    st.markdown(f"**Operador:** `{nome_operador}`")

lista_barras = sorted(
    coords.keys(),
    key=lambda x: (x.isdigit(), int(x) if x.isdigit() else x),
)

# --------- VÃO SIMPLES ---------
st.subheader("🔧 Vão simples (U–V)")

col_u, col_v = st.columns(2)
with col_u:
    u_simples = st.selectbox("Barra U", lista_barras, key="u_simples")
with col_v:
    v_simples = st.selectbox("Barra V", lista_barras, key="v_simples")

if st.button("Confirmar vão simples (Modo 1)"):
    vao_simples = (u_simples, v_simples)
    info = buscar_nf_para_vao(u_simples, v_simples, vao_map)

    st.subheader("🔎 Resultado – Vão simples (Modo 1)")

    if info is None:
        st.error(f"Não há NF cadastrada no banco para o vão {u_simples} – {v_simples}.")
    else:
        st.success(
            f"**Operador:** `{nome_operador or 'N/D'}`  \n"
            f"**Vão:** `{u_simples} – {v_simples}`  \n"
            f"**Melhor NF:** `{info['nf']}`  |  "
            f"**Carga interrompida (NF isolada):** {info['kw']:.1f} kW  |  "
            f"**Barras isoladas:** {info['n_barras']}"
        )

        fig_vao = plotar_mapo_modo1 := plotar_mapa_modo1(
            coords,
            topo,
            vaos=[vao_simples],
            info_vaos=[info],
        )
        st.plotly_chart(fig_vao, use_container_width=True)

        st.markdown("**Sequência de manobra sugerida (Modo 1):**")
        st.markdown(
            f"1️⃣ Abrir NF **{info['nf']}** para isolar o vão **{u_simples} – {v_simples}**.  \n"
            f"✅ Demais chaves permanecem no estado nominal."
        )

    st.markdown("---")

# --------- TRECHO MULTI-VÃOS ---------
st.subheader("🧩 Trecho com múltiplos vãos (entrada em blocos de 2 barras) – MODO 1")

entrada_seq = st.text_input(
    "Sequência de barras (ex: 60,62,63,64,65,66,60,67)",
    value="60,62,63,64,65,66,60,67",
)

if st.button("Processar trecho (MODO 1 – multi-vãos)"):
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

            if nao_encontrados:
                st.warning(
                    "Não foram encontrados registros para os seguintes vãos: "
                    + ", ".join([f"{u}-{v}" for u, v in nao_encontrados])
                )

            if info_vaos:
                st.markdown("### ✅ NF de manobra por vão (impacto individual – MODO 1)")

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
                st.markdown("### 🗺️ Mapa com trecho e NFs destacadas (MODO 1)")
                st.plotly_chart(fig_multi, use_container_width=True)

                st.markdown("### ⚡ Impacto consolidado da manobra (sem dupla contagem – MODO 1)")
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
                        f"**Operador:** `{nome_operador or 'N/D'}`  \n"
                        f"**Carga total interrompida:** {kw_total:.1f} kW  \n"
                        f"**Barras desenergizadas únicas:** {n_barras_unicas}"
                    )
                    with st.expander("Ver barras desenergizadas únicas"):
                        st.write(barras_ordenadas)

                st.markdown("### 📜 Linha de tempo de manobra (MODO 1)")

                for i, nf in enumerate(lista_nf_ordenada, start=1):
                    vaos_nf = [
                        f"{d['u']}-{d['v']}" for d in info_vaos if d["nf"] == nf
                    ]
                    st.markdown(
                        f"{i}️⃣ Abrir NF **{nf}** para isolar os vãos: "
                        + ", ".join(vaos_nf)
                    )

                st.markdown(
                    "✅ Após conclusão da manutenção, **fechar as NFs na ordem inversa**, "
                    "conforme os procedimentos operacionais da distribuidora."
                )

st.markdown("---")

# =========================================================
# MODO 2 – DUAS FONTES (NF–NA–NF)
# =========================================================
st.header("🟣 MODO 2 – Manutenção com Duas Fontes (NF–NA–NF via banco)")

if nome_operador:
    st.markdown(f"**Operador:** `{nome_operador}`")

if not nf_na_nf_regs:
    st.warning(
        "Tabela `nf_na_nf` não encontrada ou vazia. "
        "O MODO 2 depende desse banco pré-calculado."
    )
else:
    st.markdown(
        """
**Entrada do MODO 2 é feita por LINHAS** do trecho a ser desligado.

A plataforma procura, no banco `nf_na_nf`, as combinações **NF–NA–NF** que:

- isolam todas as barras do trecho (todas as barras do trecho ∈ `buses_off`), e  
- reduzem a carga interrompida em relação à NF isoladora sozinha, quando há NA envolvida.
"""
    )

    # Lista de linhas "normais" para facilitar (pode incluir tudo, se preferir)
    linhas_disponiveis = sorted(
        {el["line"] for el in topo if not el["is_switch"]},
        key=lambda x: (x.startswith("l"), x),
    )

    trecho_linhas_sel = st.multiselect(
        "Selecione as LINHAS do trecho de manutenção (MODO 2):",
        options=linhas_disponiveis,
    )

    if st.button("Buscar opções de manobra (MODO 2 – NF–NA–NF)"):
        if not trecho_linhas_sel:
            st.error("Selecione pelo menos uma linha para o trecho.")
        else:
            trecho_buses, vaos_m2 = montar_trecho_buses_de_linhas(
                topo, trecho_linhas_sel
            )

            st.markdown(
                f"**Trecho (barras) considerado (MODO 2):** `{sorted(trecho_buses)}`"
            )

            opcoes = buscar_opcoes_modo2(
                trecho_buses=trecho_buses,
                nf_na_nf_regs=nf_na_nf_regs,
            )

            if not opcoes:
                st.error(
                    "❌ Nenhuma combinação NF–NA–NF encontrada no banco que isole "
                    "esse trecho com as condições pré-calculadas."
                )
            else:
                st.markdown("### ✅ TOP opções de manobra (ordenadas por menor carga desligada – MODO 2)")
                top3 = opcoes[:3]

                tabela_top3 = [
                    {
                        "Opção": i,
                        "NF isoladora": opt["nf1"],
                        "NA restabelecimento": opt["na"] or "-",
                        "NF bloqueio": opt["nf_block"] or "-",
                        "Carga desligada [kW]": f"{opt['kw_off']:.1f}",
                        "Nº manobras": opt["n_manobras"],
                        "Impacto NF isoladora (kW)": f"{opt['kw_off_base_nf']:.1f}",
                        "Vmin_pu": f"{opt['vmin_pu']:.3f}",
                        "Vmax_pu": f"{opt['vmax_pu']:.3f}",
                        "Max loading (pu)": f"{opt['max_loading']:.3f}",
                    }
                    for i, opt in enumerate(top3, start=1)
                ]
                st.table(tabela_top3)

                if len(opcoes) > 3:
                    with st.expander("Ver TODAS as opções válidas (MODO 2)"):
                        tabela_all = [
                            {
                                "Opção": i,
                                "NF isoladora": opt["nf1"],
                                "NA restabelecimento": opt["na"] or "-",
                                "NF bloqueio": opt["nf_block"] or "-",
                                "Carga desligada [kW]": f"{opt['kw_off']:.1f}",
                                "Nº manobras": opt["n_manobras"],
                                "Impacto NF isoladora (kW)": f"{opt['kw_off_base_nf']:.1f}",
                                "Vmin_pu": f"{opt['vmin_pu']:.3f}",
                                "Vmax_pu": f"{opt['vmax_pu']:.3f}",
                                "Max loading (pu)": f"{opt['max_loading']:.3f}",
                            }
                            for i, opt in enumerate(opcoes, start=1)
                        ]
                        st.table(tabela_all)

                st.markdown("### 📜 Relatório resumido (MODO 2)")
                for i, opt in enumerate(top3, start=1):
                    st.markdown(
                        f"**Opção {i} – Operador `{nome_operador or 'N/D'}`**  \n"
                        f"- NF isoladora: **{opt['nf1']}**  \n"
                        f"- NA de restabelecimento: **{opt['na'] or '-'}**  \n"
                        f"- NF de bloqueio: **{opt['nf_block'] or '-'}**  \n"
                        f"- Carga desligada: **{opt['kw_off']:.1f} kW**  \n"
                        f"- Nº de manobras: **{opt['n_manobras']}**  \n"
                        f"- Impacto NF isoladora (sem NA): **{opt['kw_off_base_nf']:.1f} kW**  \n"
                        f"- Vmin/Vmax (pu): **{opt['vmin_pu']:.3f} / {opt['vmax_pu']:.3f}**  \n"
                        f"- Carregamento máximo (pu): **{opt['max_loading']:.3f}**"
                    )

                    fig_op = plotar_mapa_modo2(
                        coords,
                        topo,
                        trecho_linhas_sel,
                        opt,
                    )
                    st.plotly_chart(fig_op, use_container_width=True)
