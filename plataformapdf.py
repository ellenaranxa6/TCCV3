import streamlit as st
import plotly.graph_objects as go
import sqlite3
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import ast
import io
from datetime import datetime

# PDF
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm


# =========================================================
# CONFIGURAÇÃO INICIAL
# =========================================================
st.set_page_config(
    page_title="Isolamento Real IEEE-123 Bus",
    layout="wide"
)

st.title("⚡ Plataforma Interativa – Isolamento Real IEEE 123 Bus")

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "ieee123_manobras.db"


# =========================================================
# FUNÇÕES – PDF
# =========================================================
def gerar_relatorio_pdf(
    operador: str,
    modo: str,
    trecho: str,
    opcao_escolhida: Dict,
    barras_afetadas: List[str],
    linhas_afetadas: List[str],
) -> io.BytesIO:
    """
    Gera PDF com informações da manobra escolhida.
    Retorna um buffer (BytesIO) pronto para download.
    """
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)

    largura, altura = letter
    margem = 2 * cm
    y = altura - margem

    p.setFont("Helvetica-Bold", 16)
    p.drawString(margem, y, "Relatório de Manobra – Sistema IEEE 123 Bus")
    y -= 30

    p.setFont("Helvetica", 11)
    p.drawString(margem, y, f"Operador(a): {operador}")
    y -= 15
    p.drawString(margem, y, f"Data/Hora: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    y -= 25

    p.setFont("Helvetica-Bold", 13)
    p.drawString(margem, y, f"Modo de operação: {modo}")
    y -= 20

    p.setFont("Helvetica", 11)
    p.drawString(margem, y, f"Trecho selecionado: {trecho}")
    y -= 25

    p.setFont("Helvetica-Bold", 12)
    p.drawString(margem, y, "Opção escolhida:")
    y -= 18

    p.setFont("Helvetica", 11)

    def escreve_linha(txt: str):
        nonlocal y
        p.drawString(margem, y, txt)
        y -= 15
        if y < 60:
            p.showPage()
            y = altura - margem
            p.setFont("Helvetica", 11)

    escreve_linha(f"NF isoladora: {opcao_escolhida.get('nf1', '-')}")
    escreve_linha(f"NA: {opcao_escolhida.get('na', '-')}")
    escreve_linha(f"NF bloqueio: {opcao_escolhida.get('nf_block', '-')}")
    escreve_linha(f"Carga desligada: {opcao_escolhida.get('kw_off', 0.0):.1f} kW")
    escreve_linha(f"Nº de manobras: {opcao_escolhida.get('n_manobras', '-')}")

    if "kw_off_base_nf" in opcao_escolhida:
        escreve_linha(
            f"Impacto NF isoladora (sem NA): {opcao_escolhida['kw_off_base_nf']:.1f} kW"
        )

    if "vmin_pu" in opcao_escolhida and "vmax_pu" in opcao_escolhida:
        escreve_linha(
            f"Tensão mínima / máxima (pu): "
            f"{opcao_escolhida['vmin_pu']:.3f} / {opcao_escolhida['vmax_pu']:.3f}"
        )
    if "max_loading" in opcao_escolhida:
        escreve_linha(f"Carregamento máximo: {opcao_escolhida['max_loading']:.3f} pu")

    y -= 10
    p.setFont("Helvetica-Bold", 12)
    p.drawString(margem, y, "Barras afetadas:")
    y -= 18

    p.setFont("Helvetica", 10)
    for b in barras_afetadas:
        escreve_linha(f"- Barra {b}")

    y -= 10
    p.setFont("Helvetica-Bold", 12)
    p.drawString(margem, y, "Linhas afetadas:")
    y -= 18

    p.setFont("Helvetica", 10)
    for l in linhas_afetadas:
        escreve_linha(f"- Linha {l}")

    p.showPage()
    p.save()
    buffer.seek(0)
    return buffer


# =========================================================
# FUNÇÕES AUXILIARES – BANCO
# =========================================================
def get_connection() -> sqlite3.Connection:
    return sqlite3.connect(DB_PATH)


@st.cache_data(show_spinner=False)
def carregar_coords() -> Dict[str, Tuple[float, float]]:
    """Tabela coords(bus, x, y)."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT bus, x, y FROM coords")
    rows = cur.fetchall()
    conn.close()
    return {str(b): (float(x), float(y)) for b, x, y in rows}


@st.cache_data(show_spinner=False)
def carregar_topologia():
    """
    Tabela topology(line, from_bus, to_bus, is_switch, norm).
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
    Para MODO 1 – Tabela vao_map(u_bus, v_bus, nf, kw, n_barras).
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
    Para MODO 1 consolidação: nf_map(nf, barras_isoladas TEXT, kw REAL, n_barras INTEGER)
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
    Para MODO 2 – Tabela nf_na_nf:
    (id, nf1, na, nf_block, buses_off, lines_off, kw_off,
     vmin_pu, vmax_pu, max_loading, n_manobras, switch_states)
    """
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT id, nf1, na, nf_block, buses_off, lines_off, "
            "kw_off, vmin_pu, vmax_pu, max_loading, n_manobras, switch_states "
            "FROM nf_na_nf"
        )
        rows = cur.fetchall()
    except sqlite3.OperationalError:
        rows = []
    conn.close()

    registros = []
    for (
        id_,
        nf1,
        na,
        nf_block,
        buses_str,
        lines_str,
        kw_off,
        vmin,
        vmax,
        max_load,
        n_man,
        switch_states,
    ) in rows:
        # Parse listas de barras/linhas
        def parse_list(s):
            if not s:
                return []
            try:
                return [str(x).strip() for x in ast.literal_eval(s)]
            except Exception:
                s2 = (
                    str(s)
                    .replace("[", "")
                    .replace("]", "")
                    .replace('"', "")
                    .replace("'", "")
                )
                return [x.strip() for x in s2.split(",") if x.strip()]

        buses_list = parse_list(buses_str)
        lines_list = parse_list(lines_str)

        registros.append(
            dict(
                id=int(id_),
                nf1=str(nf1),
                na=None if na in (None, "", "None") else str(na),
                nf_block=None
                if nf_block in (None, "", "None")
                else str(nf_block),
                buses_off=set(buses_list),
                lines_off=set(lines_list),
                kw_off=float(kw_off),
                vmin_pu=float(vmin),
                vmax_pu=float(vmax),
                max_loading=float(max_load),
                n_manobras=int(n_man),
                switch_states=str(switch_states),
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
    vao_map: List[Dict],
) -> Optional[Dict]:
    """
    Procura no vao_map a NF ótima para o vão (u, v),
    considerando ordem (u,v) ou (v,u).
    """
    candidatos = [
        registro
        for registro in vao_map
        if (registro["u_bus"] == u and registro["v_bus"] == v)
        or (registro["u_bus"] == v and registro["v_bus"] == u)
    ]
    if not candidatos:
        return None

    candidatos.sort(key=lambda r: (r["kw"], r["n_barras"]))
    return candidatos[0]


def impacto_consolidado(
    lista_nf: List[str],
    loads: Dict[str, float],
    nf_map_data: Dict[str, Dict],
) -> Tuple[float, int, List[str]]:
    """
    Impacto consolidado de várias NFs (sem dupla contagem de carga):
      - união das barras isoladas por todas as NFs;
      - soma dos kW por barra (tabela loads).
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
# FUNÇÕES DE PROCESSAMENTO – MODO 2
# =========================================================
def normalizar_linha_nome(s: str) -> str:
    s = s.strip()
    s = s.lower()
    if s.startswith("line."):
        s = s[5:]
    return s


def trecho_from_lines(
    line_names: List[str],
    topo: List[Dict],
) -> Tuple[set, List[str]]:
    """
    A partir de nomes de linhas (Lxx, lxx, line.lxx), retorna:
      - conjunto de barras do trecho
      - lista de nomes normalizados das linhas.
    """
    line_names_norm = [normalizar_linha_nome(x) for x in line_names]
    topo_by_line = {el["line"].lower(): el for el in topo}

    trecho_buses = set()
    trecho_lines_norm = []

    for ln in line_names_norm:
        el = topo_by_line.get(ln)
        if not el:
            continue
        trecho_buses.add(str(el["from_bus"]))
        trecho_buses.add(str(el["to_bus"]))
        trecho_lines_norm.append(ln)

    return trecho_buses, trecho_lines_norm


def filtrar_combos_para_trecho(
    trecho_buses: set,
    combos: List[Dict],
) -> List[Dict]:
    """
    Seleciona combos nf_na_nf que desligam TODO o trecho (todas as barras do trecho).
    Critério: trecho_buses ⊆ buses_off.
    Ordena por (kw_off, n_manobras).
    """
    if not trecho_buses:
        return []

    candidatos = []
    for reg in combos:
        if trecho_buses.issubset(reg["buses_off"]):
            candidatos.append(reg)

    candidatos.sort(key=lambda r: (r["kw_off"], r["n_manobras"]))
    return candidatos


# =========================================================
# FUNÇÕES DE PLOT – MAPA
# =========================================================
def construir_mapa_base(coords, topo):
    """
    Figura Plotly com topologia base (barras + linhas),
    mostrando também o nome das linhas na posição média de cada trecho.
    """
    edge_x = []
    edge_y = []

    # Linhas base (cinza)
    for el in topo:
        u = el["from_bus"]
        v = el["to_bus"]
        if u in coords and v in coords:
            x0, y0 = coords[u]
            x1, y1 = coords[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

    # Nós
    node_x = [coords[b][0] for b in coords]
    node_y = [coords[b][1] for b in coords]
    node_text = list(coords.keys())

    fig = go.Figure()

    # Linhas base
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

    # Rótulos das linhas (nome das linhas)
    label_x = []
    label_y = []
    label_text = []

    for el in topo:
        line_name = el["line"]
        u = el["from_bus"]
        v = el["to_bus"]
        if u in coords and v in coords:
            x0, y0 = coords[u]
            x1, y1 = coords[v]
            xm = (x0 + x1) / 2
            ym = (y0 + y1) / 2
            label_x.append(xm)
            label_y.append(ym)
            label_text.append(line_name)

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
        height=650,
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
    Modo 1:
      - Vãos selecionados (linhas pretas grossas)
      - NF associadas em vermelho tracejado
      - nome das linhas visível no mapa
    """
    fig = construir_mapa_base(coords, topo)

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

    # NF destacadas
    nf_edges_x = []
    nf_edges_y = []
    nf_labels_x = []
    nf_labels_y = []
    nf_labels_text = []

    topo_por_line = {el["line"].lower(): el for el in topo}

    for info in info_vaos:
        nf = info["nf"].lower()
        el = topo_por_line.get(nf)
        if el:
            u = el["from_bus"]
            v = el["to_bus"]
            if u in coords and v in coords:
                x0, y0 = coords[u]
                x1, y1 = coords[v]
                nf_edges_x += [x0, x1, None]
                nf_edges_y += [y0, y1, None]
                nf_labels_x.append((x0 + x1) / 2)
                nf_labels_y.append((y0 + y1) / 2)
                nf_labels_text.append(el["line"])

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
                textfont=dict(color="red", size=9),
                showlegend=False,
            )
        )

    return fig


def plotar_mapa_modo2(
    coords,
    topo,
    trecho_lines: List[str],
    opcao: Dict,
):
    """
    Modo 2:
      - Trecho (linhas informadas) em preto
      - NF isoladora (nf1) em vermelho
      - NA em azul
      - NF bloqueio em laranja
      - nome das linhas visível
    """
    fig = construir_mapa_base(coords, topo)
    topo_por_line = {el["line"].lower(): el for el in topo}

    # Trecho selecionado
    trecho_edge_x = []
    trecho_edge_y = []

    for ln in trecho_lines:
        ln_norm = normalizar_linha_nome(ln)
        el = topo_por_line.get(ln_norm)
        if el:
            u = el["from_bus"]
            v = el["to_bus"]
            if u in coords and v in coords:
                x0, y0 = coords[u]
                x1, y1 = coords[v]
                trecho_edge_x += [x0, x1, None]
                trecho_edge_y += [y0, y1, None]

    if trecho_edge_x:
        fig.add_trace(
            go.Scatter(
                x=trecho_edge_x,
                y=trecho_edge_y,
                mode="lines",
                line=dict(color="black", width=4),
                name="Trecho selecionado (linhas)",
                hoverinfo="none",
            )
        )

    # NF1 isoladora
    nf1 = opcao.get("nf1")
    if nf1:
        nf1_norm = normalizar_linha_nome(nf1)
        el = topo_por_line.get(nf1_norm)
        if el:
            u = el["from_bus"]
            v = el["to_bus"]
            if u in coords and v in coords:
                x0, y0 = coords[u]
                x1, y1 = coords[v]
                fig.add_trace(
                    go.Scatter(
                        x=[x0, x1],
                        y=[y0, y1],
                        mode="lines+text",
                        line=dict(color="red", width=4),
                        text=[el["line"], ""],
                        textposition="middle center",
                        textfont=dict(color="red", size=9),
                        name="NF isoladora",
                        hoverinfo="none",
                    )
                )

    # NA
    na = opcao.get("na")
    if na:
        na_norm = normalizar_linha_nome(na)
        el = topo_por_line.get(na_norm)
        if el:
            u = el["from_bus"]
            v = el["to_bus"]
            if u in coords and v in coords:
                x0, y0 = coords[u]
                x1, y1 = coords[v]
                fig.add_trace(
                    go.Scatter(
                        x=[x0, x1],
                        y=[y0, y1],
                        mode="lines+text",
                        line=dict(color="blue", width=4, dash="dash"),
                        text=[el["line"], ""],
                        textposition="middle center",
                        textfont=dict(color="blue", size=9),
                        name="NA (restabelecimento)",
                        hoverinfo="none",
                    )
                )

    # NF bloqueio
    nf_block = opcao.get("nf_block")
    if nf_block:
        nf_block_norm = normalizar_linha_nome(nf_block)
        el = topo_por_line.get(nf_block_norm)
        if el:
            u = el["from_bus"]
            v = el["to_bus"]
            if u in coords and v in coords:
                x0, y0 = coords[u]
                x1, y1 = coords[v]
                fig.add_trace(
                    go.Scatter(
                        x=[x0, x1],
                        y=[y0, y1],
                        mode="lines+text",
                        line=dict(color="orange", width=4, dash="dot"),
                        text=[el["line"], ""],
                        textposition="middle center",
                        textfont=dict(color="orange", size=9),
                        name="NF bloqueio",
                        hoverinfo="none",
                    )
                )

    return fig


# =========================================================
# CARREGAMENTO DO BANCO E STATUS
# =========================================================
st.sidebar.header("📂 Dados carregados")

if not DB_PATH.exists():
    st.sidebar.error("Banco ieee123_manobras.db não encontrado na pasta do app.")
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
vao_map = carregar_vao_map()
loads = carregar_loads()
nf_map_data = carregar_nf_map()
nf_na_nf_data = carregar_nf_na_nf()

if not coords or not topo:
    st.error("Banco encontrado, mas alguma tabela essencial está vazia.")
    st.stop()

# Nome do operador
st.sidebar.subheader("👤 Operador(a)")
operador_nome = st.sidebar.text_input("Nome do operador(a)", value="Ellen")


# =========================================================
# DESCRIÇÃO RÁPIDA
# =========================================================
st.markdown(
    """
Ferramenta de apoio à manobra de **desligamento programado** em redes de distribuição,
baseada no alimentador teste **IEEE-123 Bus**.

Toda a inteligência de manobra foi calculada previamente em **OpenDSS + Python (Colab)** e
armazenada no banco **SQLite** `ieee123_manobras.db`.

Esta aplicação possui dois modos:

- 🟢 **Modo 1 – Fonte única**: usa apenas a fonte A e chaves NF para isolar vãos programados.  
- 🟠 **Modo 2 – Duas fontes (NF–NA–NF)**: considera a entrada da Fonte B pela chave L108,
  com fechamento de NA e NF de bloqueio para reduzir carga desligada, respeitando o fluxo de carga real.
"""
)

st.markdown("---")


# =========================================================
# MAPA BASE ESTÁTICO
# =========================================================
st.subheader("🗺️ Mapa Base da Rede (IEEE-123 Bus)")
fig_base = construir_mapa_base(coords, topo)
st.plotly_chart(fig_base, use_container_width=True)

st.markdown("---")


# =========================================================
# SELETOR DE MODO
# =========================================================
modo = st.radio(
    "Selecione o modo de operação:",
    ["Modo 1 - Fonte única", "Modo 2 - Duas fontes (NF–NA–NF)"],
    horizontal=True,
)


# =========================================================
# MODO 1 – FONTE ÚNICA
# =========================================================
if modo == "Modo 1 - Fonte única":
    st.header("🟢 MODO 1 – Fonte única (NF isoladora)")

    st.sidebar.subheader("🔧 Vão simples (U-V)")

    lista_barras = sorted(
        coords.keys(),
        key=lambda x: (x.isdigit(), int(x) if x.isdigit() else x),
    )

    u_simples = st.sidebar.selectbox("Barra U", lista_barras, key="u_simples_m1")
    v_simples = st.sidebar.selectbox("Barra V", lista_barras, key="v_simples_m1")

    if st.sidebar.button("Confirmar vão simples (Modo 1)"):
        vao_simples = (u_simples, v_simples)
        info = buscar_nf_para_vao(u_simples, v_simples, vao_map)

        st.subheader("🔎 Resultado – Vão simples (Modo 1)")

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

            # PDF para Modo 1 – vão simples
            barras_afetadas = nf_map_data.get(info["nf"], {}).get("barras", set())
            linhas_afetadas = [
                el["line"]
                for el in topo
                if el["from_bus"] in barras_afetadas
                or el["to_bus"] in barras_afetadas
            ]

            if st.button("Gerar relatório em PDF (vão simples)"):
                trecho_str = f"{u_simples}-{v_simples}"
                opcao_pdf = {
                    "nf1": info["nf"],
                    "na": "-",
                    "nf_block": "-",
                    "kw_off": info["kw"],
                    "n_manobras": 1,
                    "kw_off_base_nf": info["kw"],
                }
                buffer = gerar_relatorio_pdf(
                    operador_nome or "N/D",
                    "Modo 1 - Fonte única (vão simples)",
                    trecho_str,
                    opcao_pdf,
                    sorted(list(barras_afetadas)),
                    sorted(linhas_afetadas),
                )
                st.download_button(
                    label="📄 Baixar PDF",
                    data=buffer,
                    file_name="Relatorio_Modo1_VaoSimples.pdf",
                    mime="application/pdf",
                )

        st.markdown("---")

    # ---- TRECHO MULTI-VÃOS ----
    st.subheader("🧩 Trecho com múltiplos vãos (entrada em blocos de 2 barras)")

    entrada_seq = st.text_input(
        "Sequência de barras (ex: 60,62,63,64,65,66,60,67)",
        value="60,62,63,64,65,66,60,67",
        key="entrada_multi_m1",
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

                    kw_total, n_barras_unicas, barras_ordenadas = impacto_consolidado(
                        lista_nf_ordenada, loads, nf_map_data
                    )

                    if not nf_map_data:
                        st.warning(
                            "Tabela `nf_map` não está disponível no banco. "
                            "Não foi possível calcular o impacto consolidado."
                        )
                    else:
                        st.success(
                            f"**Carga total interrompida:** {kw_total:.1f} kW  \n"
                            f"**Barras desenergizadas únicas:** {n_barras_unicas}"
                        )

                        with st.expander("Ver barras desenergizadas únicas"):
                            st.write(barras_ordenadas)

                    st.markdown(
                        "### 📜 Linha de tempo de manobra (sequência sugerida)"
                    )
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

                    # PDF para Modo 1 – multi-vãos
                    if st.button("Gerar relatório em PDF (trecho multi-vãos)"):
                        barras_tot = set()
                        for nf in lista_nf_ordenada:
                            reg = nf_map_data.get(nf)
                            if reg:
                                barras_tot |= reg["barras"]

                        linhas_tot = [
                            el["line"]
                            for el in topo
                            if el["from_bus"] in barras_tot
                            or el["to_bus"] in barras_tot
                        ]

                        trecho_str = ", ".join(
                            [f"{u}-{v}" for (u, v) in vaos]
                        )
                        opcao_pdf = {
                            "nf1": " / ".join(lista_nf_ordenada),
                            "na": "-",
                            "nf_block": "-",
                            "kw_off": kw_total,
                            "n_manobras": len(lista_nf_ordenada),
                            "kw_off_base_nf": kw_total,
                        }
                        buffer = gerar_relatorio_pdf(
                            operador_nome or "N/D",
                            "Modo 1 - Fonte única (multi-vãos)",
                            trecho_str,
                            opcao_pdf,
                            sorted(list(barras_tot)),
                            sorted(linhas_tot),
                        )
                        st.download_button(
                            label="📄 Baixar PDF",
                            data=buffer,
                            file_name="Relatorio_Modo1_MultiVaos.pdf",
                            mime="application/pdf",
                        )


# =========================================================
# MODO 2 – DUAS FONTES (NF–NA–NF)
# =========================================================
else:
    st.header("🟠 MODO 2 – Duas fontes (NF–NA–NF) com Fonte B via L108")

    st.markdown(
        """
No **Modo 2**, são consideradas manobras do tipo:

- NF isoladora (Fonte A)
- NA de restabelecimento (SW7, SW8 ou L108)
- NF de bloqueio (se necessário)

As combinações já foram pré-simuladas no OpenDSS (sem paralelismo entre fontes e
com verificação de fluxo de carga), e os resultados estão na tabela `nf_na_nf`.
"""
    )

    st.subheader("🧩 Trecho a ser desligado (por LINHAS)")

    entrada_linhas_m2 = st.text_input(
        "Informe as linhas do trecho (ex: L70 ou L70,L71,L72)",
        value="L70",
        key="entrada_linhas_m2",
    )

    linhas_trecho_raw = [
        s.strip()
        for s in entrada_linhas_m2.replace(";", ",").split(",")
        if s.strip()
    ]

    if st.button("Processar Trecho – Modo 2"):
        if not linhas_trecho_raw:
            st.error("Informe pelo menos uma linha.")
        else:
            trecho_buses, trecho_lines_norm = trecho_from_lines(
                linhas_trecho_raw, topo
            )

            if not trecho_buses:
                st.error(
                    "Nenhuma barra do trecho pôde ser identificada a partir das linhas informadas."
                )
            else:
                st.markdown(
                    f"**Barras do trecho considerado:** {sorted(list(trecho_buses))}"
                )

                # Filtra combos da tabela nf_na_nf
                combos_trecho = filtrar_combos_para_trecho(
                    trecho_buses, nf_na_nf_data
                )

                if not combos_trecho:
                    st.error(
                        "Nenhuma combinação NF–NA–NF do banco desliga exatamente esse trecho."
                    )
                else:
                    st.success(
                        f"Foram encontradas **{len(combos_trecho)}** combinações válidas para o trecho."
                    )

                    # Lista de opções para seleção
                    opcoes_labels = []
                    for i, c in enumerate(combos_trecho, start=1):
                        label = (
                            f"Opção {i}: NF={c['nf1']}  |  "
                            f"NA={c['na'] or '-'}  |  "
                            f"NF_bloq={c['nf_block'] or '-'}  |  "
                            f"kw_off={c['kw_off']:.1f}  |  "
                            f"manobras={c['n_manobras']}"
                        )
                        opcoes_labels.append(label)

                    st.markdown("### ✅ Opções de manobra encontradas (ordenadas):")

                    for lbl in opcoes_labels[:10]:
                        st.write("• " + lbl)

                    idx_escolhida = st.selectbox(
                        "Selecione a opção desejada para detalhamento:",
                        options=list(range(len(combos_trecho))),
                        format_func=lambda i: opcoes_labels[i],
                        key="opcao_m2",
                    )

                    opcao = combos_trecho[idx_escolhida]

                    st.markdown("### 🔍 Detalhamento da opção selecionada")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.write(f"**NF isoladora:** `{opcao['nf1']}`")
                        st.write(f"**NA (restabelecimento):** `{opcao['na'] or '-'}`")
                        st.write(
                            f"**NF de bloqueio:** `{opcao['nf_block'] or '-'}`"
                        )
                        st.write(
                            f"**Carga desligada:** `{opcao['kw_off']:.1f} kW`"
                        )
                        st.write(
                            f"**Nº de manobras:** `{opcao['n_manobras']}`"
                        )

                    with col2:
                        st.write(
                            f"**Vmin (pu):** `{opcao['vmin_pu']:.3f}`  |  "
                            f"**Vmax (pu):** `{opcao['vmax_pu']:.3f}`"
                        )
                        st.write(
                            f"**Carregamento máximo:** `{opcao['max_loading']:.3f} pu`"
                        )

                        st.write("**Barras desligadas (trecho + adjacentes):**")
                        st.write(sorted(list(opcao["buses_off"]))[:30])
                        if len(opcao["buses_off"]) > 30:
                            st.write("...")

                    # Mapa com destaques
                    st.markdown("### 🗺️ Mapa da manobra selecionada")
                    fig_m2 = plotar_mapa_modo2(
                        coords,
                        topo,
                        trecho_lines_norm,
                        opcao,
                    )
                    st.plotly_chart(fig_m2, use_container_width=True)

                    # PDF – Modo 2
                    if st.button("Gerar relatório em PDF (Modo 2)"):
                        trecho_str = ", ".join(linhas_trecho_raw)
                        barras_afetadas = sorted(list(opcao["buses_off"]))
                        linhas_afetadas = sorted(list(opcao["lines_off"]))

                        buffer = gerar_relatorio_pdf(
                            operador_nome or "N/D",
                            "Modo 2 - Duas fontes (NF–NA–NF)",
                            trecho_str,
                            opcao,
                            barras_afetadas,
                            linhas_afetadas,
                        )
                        st.download_button(
                            label="📄 Baixar PDF",
                            data=buffer,
                            file_name="Relatorio_Modo2.pdf",
                            mime="application/pdf",
                        )

::contentReference[oaicite:0]{index=0}
