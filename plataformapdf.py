import streamlit as st
import plotly.graph_objects as go
import sqlite3
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import ast
import pandas as pd
from datetime import date
import io

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader

# =========================================================
# CONFIGURAÇÃO INICIAL
# =========================================================
st.set_page_config(
    page_title="Plataforma Isolamento / Manobras IEEE-123",
    layout="wide"
)

st.title("⚡ Plataforma Interativa – IEEE 123 Bus (Isolamento & Manobras)")

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "ieee123_manobras.db"

# =========================================================
# FUNÇÕES AUXILIARES – BANCO
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
    """Lê tabela coords(bus, x, y) do banco."""
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT bus, x, y FROM coords")
        rows = cur.fetchall()
    except sqlite3.OperationalError:
        rows = []
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
    try:
        cur.execute(
            "SELECT line, from_bus, to_bus, is_switch, norm "
            "FROM topology"
        )
        rows = cur.fetchall()
    except sqlite3.OperationalError:
        rows = []
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
    try:
        cur.execute("SELECT bus, kw FROM loads")
        rows = cur.fetchall()
    except sqlite3.OperationalError:
        rows = []
    conn.close()
    return {str(b): float(kw) for b, kw in rows}


@st.cache_data(show_spinner=False)
def carregar_vao_map():
    """
    Tabela vao_map(u_bus, v_bus, nf, kw, n_barras).
    Usada no MODO 1 (fonte única).
    """
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT u_bus, v_bus, nf, kw, n_barras FROM vao_map"
        )
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
def carregar_nf_map():
    """
    Tabela nf_map(nf, barras_isoladas TEXT, kw REAL, n_barras INTEGER).

    barras_isoladas está salva como uma string de lista Python,
    ex.: '["33", "61", "18", ...]'.

    Usada no MODO 1 para impacto consolidado.
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
def carregar_nf_na_nf() -> pd.DataFrame:
    """
    Tabela nf_na_nf para o MODO 2 (duas fontes):
      nf1, na, nf_block, buses_off, lines_off, kw_off, vmin_pu, vmax_pu,
      max_loading, n_manobras, switch_states.
    """
    conn = get_connection()
    try:
        df = pd.read_sql_query(
            "SELECT id, nf1, na, nf_block, buses_off, lines_off, "
            "kw_off, vmin_pu, vmax_pu, max_loading, n_manobras, switch_states "
            "FROM nf_na_nf",
            conn,
        )
    except Exception:
        conn.close()
        return pd.DataFrame()
    conn.close()

    def _parse_list_column(series: pd.Series) -> pd.Series:
        def _parse_one(x):
            if x is None:
                return []
            s = str(x).strip()
            if not s:
                return []
            try:
                lst = ast.literal_eval(s)
                return [str(v).strip().strip('"').strip("'") for v in lst]
            except Exception:
                # fallback burro
                s2 = s.replace("[", "").replace("]", "").replace('"', "")
                return [p.strip() for p in s2.split(",") if p.strip()]

        return series.apply(_parse_one)

    if "buses_off" in df.columns:
        df["buses_off_list"] = _parse_list_column(df["buses_off"])
    else:
        df["buses_off_list"] = [[] for _ in range(len(df))]

    if "lines_off" in df.columns:
        df["lines_off_list"] = _parse_list_column(df["lines_off"])
    else:
        df["lines_off_list"] = [[] for _ in range(len(df))]

    return df


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


def obter_barras_unicas(vaos: List[Tuple[str, str]]) -> List[str]:
    """Retorna a lista de barras únicas presentes em uma lista de vãos."""
    s = set()
    for u, v in vaos:
        s.add(u)
        s.add(v)
    return sorted(s, key=lambda x: (x.isdigit(), int(x) if x.isdigit() else x))


def impacto_consolidado(lista_nf: List[str],
                        loads: Dict[str, float],
                        nf_map_data: Dict[str, Dict]) -> Tuple[float, int, List[str]]:
    """
    Calcula o impacto consolidado de uma manobra que envolve
    várias NFs, **sem dupla contagem** de carga.
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
# FUNÇÕES PARA MODO 2 – TRECHO A PARTIR DE LINHAS
# =========================================================
def normalizar_nome_linha(raw: str) -> str:
    """
    Converte entradas do usuário em formato do banco:
    Ex.: "L70", "line.l70", "l70" -> "l70".
    """
    if not raw:
        return ""
    s = raw.strip().lower()
    if s.startswith("line."):
        s = s.split(".", 1)[1]
    return s


def get_trecho_buses_from_lines_modo2(
    line_names: List[str],
    topo: List[Dict],
) -> Tuple[set, List[str]]:
    """
    A partir de uma lista de nomes de LINHAS (L70, l70, line.l70),
    retorna:
      - conjunto de barras do trecho
      - lista de nomes de linha normalizados (l70) para destaque.
    """
    buses = set()
    norm_lines = []

    for raw in line_names:
        lname = normalizar_nome_linha(raw)
        if not lname:
            continue
        norm_lines.append(lname)

        for el in topo:
            if el["line"].lower() == lname:
                buses.add(str(el["from_bus"]))
                buses.add(str(el["to_bus"]))

    return buses, norm_lines


def filtrar_combos_para_trecho(
    df_nf_na_nf: pd.DataFrame,
    trecho_buses: set,
) -> pd.DataFrame:
    """
    Retorna subconjunto do df_nf_na_nf com combos que desligam
    TODAS as barras do trecho (trecho_buses ⊆ buses_off_list).
    """
    if df_nf_na_nf.empty or not trecho_buses:
        return pd.DataFrame()

    def trecho_in_buses_off(lst: List[str]) -> bool:
        s = set(str(b) for b in lst)
        return trecho_buses.issubset(s)

    mask = df_nf_na_nf["buses_off_list"].apply(trecho_in_buses_off)
    subset = df_nf_na_nf[mask].copy()

    if subset.empty:
        return subset

    subset = subset.sort_values(by=["kw_off", "n_manobras"]).reset_index(drop=True)
    return subset


def normalize_str_none(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    if s.lower() == "none":
        return None
    return s


# =========================================================
# FUNÇÕES DE PLOT (BASE COM NOME DAS LINHAS)
# =========================================================
def construir_mapa_base(coords, topo, show_bus_labels=True, show_line_labels=True):
    """
    Retorna uma figura Plotly com a topologia base:
      - Linhas em cinza claro
      - Nós em azul
      - Nome das barras (opcional)
      - Nome das linhas (pequeno, no meio do vão)
    """
    edge_x = []
    edge_y = []

    line_label_x = []
    line_label_y = []
    line_label_text = []

    for el in topo:
        u = el["from_bus"]
        v = el["to_bus"]
        if u in coords and v in coords:
            x0, y0 = coords[u]
            x1, y1 = coords[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

            if show_line_labels:
                cx = (x0 + x1) / 2
                cy = (y0 + y1) / 2
                line_label_x.append(cx)
                line_label_y.append(cy)
                line_label_text.append(el["line"])

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

    # Nós (barras)
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers" + ("+text" if show_bus_labels else ""),
            text=node_text if show_bus_labels else None,
            textposition="top center",
            marker=dict(size=6, color="#1f77b4"),
            name="Barras",
            hovertemplate="Barra %{text}<extra></extra>" if show_bus_labels else "Barra<extra></extra>",
        )
    )

    # Rótulos das linhas (texto pequeno)
    if line_label_x:
        fig.add_trace(
            go.Scatter(
                x=line_label_x,
                y=line_label_y,
                mode="text",
                text=line_label_text,
                textposition="middle center",
                textfont=dict(size=8, color="gray"),
                showlegend=False,
                hoverinfo="none",
            )
        )

    fig.update_layout(
        height=650,
        showlegend=True,
        margin=dict(l=10, r=10, t=40, b=10),
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
    Plota o mapa base + destaques:
      - vãos selecionados (linhas pretas grossas)
      - NFs associadas em vermelho tracejado
    (MODO 1 – igual ao app anterior, com rótulo de linhas).
    """
    fig = construir_mapa_base(coords, topo, show_bus_labels=True, show_line_labels=True)

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

    # --- Destaque das NF associadas ---
    nf_edges_x = []
    nf_edges_y = []
    nf_labels_x = []
    nf_labels_y = []
    nf_labels_text = []

    topo_por_line = {el["line"]: el for el in topo}

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
    trecho_lines: List[str],
    option_row: pd.Series,
):
    """
    Plota o mapa para MODO 2:
      - base com linhas + rótulos
      - trecho selecionado em preto
      - NF1 em vermelho
      - NA em verde
      - NF_block em magenta
    """
    fig = construir_mapa_base(coords, topo, show_bus_labels=True, show_line_labels=True)

    # Mapas auxiliares
    topo_por_line = {el["line"].lower(): el for el in topo}

    # 1) Destaque do trecho (linhas selecionadas)
    trecho_edge_x = []
    trecho_edge_y = []
    for lname in trecho_lines:
        el = topo_por_line.get(lname.lower())
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
                name="Trecho (linhas selecionadas)",
                hoverinfo="none",
            )
        )

    # 2) NF1 / NA / NF_block
    nf1 = normalizar_nome_linha(option_row["nf1"]) if "nf1" in option_row else None
    na = normalize_str_none(option_row.get("na"))
    nf_block = normalize_str_none(option_row.get("nf_block"))

    def add_switch_highlight(line_name_low: Optional[str], color: str, label: str):
        if not line_name_low:
            return
        el = topo_por_line.get(line_name_low)
        if not el:
            return
        u = el["from_bus"]
        v = el["to_bus"]
        if u not in coords or v not in coords:
            return
        x0, y0 = coords[u]
        x1, y1 = coords[v]
        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(color=color, width=4, dash="dot"),
                name=label,
                hoverinfo="none",
            )
        )

    if nf1:
        add_switch_highlight(nf1, "red", f"NF isoladora ({nf1})")

    if na:
        add_switch_highlight(normalizar_nome_linha(na), "green", f"NA restabelecimento ({na})")

    if nf_block:
        add_switch_highlight(normalizar_nome_linha(nf_block), "magenta", f"NF bloqueio ({nf_block})")

    fig.update_layout(title="Configuração de manobra – Modo 2 (duas fontes)")

    return fig


# =========================================================
# GERAÇÃO DE PDF – MODO 2
# =========================================================
def gerar_pdf_modo2(
    nome_operador: str,
    trecho_desc: str,
    trecho_barras: List[str],
    trecho_lines: List[str],
    option_row: pd.Series,
    fig: go.Figure,
) -> bytes:
    """
    Gera PDF multi-página com:
      - resumo
      - detalhes
      - gráfico embutido
      - linha do tempo da manobra
    """
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    w, h = A4
    today_str = date.today().isoformat()
    operador = nome_operador.strip() or "Operador"

    # Normaliza dados da opção
    nf1 = str(option_row["nf1"])
    na = normalize_str_none(option_row.get("na"))
    nf_block = normalize_str_none(option_row.get("nf_block"))
    kw_off = float(option_row.get("kw_off", 0.0))
    n_manobras = int(option_row.get("n_manobras", 0))
    vmin_pu = float(option_row.get("vmin_pu", 0.0))
    vmax_pu = float(option_row.get("vmax_pu", 0.0))
    max_loading = float(option_row.get("max_loading", 0.0))

    buses_off = [str(b) for b in option_row.get("buses_off_list", [])]
    lines_off = [str(l) for l in option_row.get("lines_off_list", [])]

    # -----------------------------------------------------
    # Página 1 – Resumo
    # -----------------------------------------------------
    c.setFont("Helvetica-Bold", 16)
    c.drawString(30 * mm, (h - 30 * mm), "Relatório de Manobra – Modo 2 (Duas Fontes)")

    c.setFont("Helvetica", 11)
    y = h - 45 * mm
    c.drawString(30 * mm, y, f"Operador(a): {operador}")
    y -= 6 * mm
    c.drawString(30 * mm, y, f"Data: {today_str}")
    y -= 6 * mm
    c.drawString(30 * mm, y, f"Trecho (barras): {trecho_desc}")
    y -= 6 * mm
    c.drawString(30 * mm, y, f"Linhas do trecho: {', '.join(trecho_lines) if trecho_lines else '-'}")
    y -= 10 * mm

    c.setFont("Helvetica-Bold", 12)
    c.drawString(30 * mm, y, "Resumo da opção selecionada:")
    y -= 8 * mm
    c.setFont("Helvetica", 11)
    c.drawString(35 * mm, y, f"NF isoladora: {nf1}")
    y -= 6 * mm
    c.drawString(35 * mm, y, f"NA de restabelecimento: {na or '-'}")
    y -= 6 * mm
    c.drawString(35 * mm, y, f"NF de bloqueio: {nf_block or '-'}")
    y -= 6 * mm
    c.drawString(35 * mm, y, f"Carga desligada total: {kw_off:.1f} kW")
    y -= 6 * mm
    c.drawString(35 * mm, y, f"Número de manobras: {n_manobras}")
    y -= 10 * mm
    c.drawString(35 * mm, y, f"Tensão mínima (pu): {vmin_pu:.3f}")
    y -= 6 * mm
    c.drawString(35 * mm, y, f"Tensão máxima (pu): {vmax_pu:.3f}")
    y -= 6 * mm
    c.drawString(35 * mm, y, f"Carregamento máximo (pu): {max_loading:.3f}")

    c.showPage()

    # -----------------------------------------------------
    # Página 2 – Listas de barras/linhas e estados
    # -----------------------------------------------------
    c.setFont("Helvetica-Bold", 14)
    c.drawString(30 * mm, (h - 30 * mm), "Detalhamento das barras, linhas e estados")

    y = h - 40 * mm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(30 * mm, y, "Barras desenergizadas:")
    y -= 8 * mm
    c.setFont("Helvetica", 10)

    # Lista de barras (quebra de linha simples)
    text_obj = c.beginText()
    text_obj.setTextOrigin(35 * mm, y)
    text_obj.setLeading(4.2 * mm)
    if buses_off:
        for b in sorted(buses_off, key=lambda x: (x.isdigit(), int(x) if x.isdigit() else x)):
            text_obj.textLine(f"- Barra {b}")
    else:
        text_obj.textLine("- Nenhuma barra desligada (lista vazia no banco).")
    c.drawText(text_obj)

    # Linhas desligadas
    y = y - (len(buses_off) + 2) * 4.2 * mm
    if y < 40 * mm:
        c.showPage()
        y = h - 40 * mm

    c.setFont("Helvetica-Bold", 12)
    c.drawString(30 * mm, y, "Linhas desenergizadas:")
    y -= 8 * mm
    c.setFont("Helvetica", 10)
    text_obj = c.beginText()
    text_obj.setTextOrigin(35 * mm, y)
    text_obj.setLeading(4.2 * mm)
    if lines_off:
        for ln in sorted(lines_off):
            text_obj.textLine(f"- Linha {ln}")
    else:
        text_obj.textLine("- Nenhuma linha desligada (lista vazia no banco).")
    c.drawText(text_obj)

    c.showPage()

    # -----------------------------------------------------
    # Página 3 – Gráfico embutido
    # -----------------------------------------------------
    # Exporta figura Plotly para PNG (requer `kaleido` instalado)
    img_bytes = fig.to_image(format="png", scale=2)
    img = ImageReader(io.BytesIO(img_bytes))
    img_w, img_h = img.getSize()

    max_w = w - 40 * mm
    max_h = h - 60 * mm
    scale = min(max_w / img_w, max_h / img_h)
    draw_w = img_w * scale
    draw_h = img_h * scale

    x = (w - draw_w) / 2
    y = (h - draw_h) / 2

    c.setFont("Helvetica-Bold", 14)
    c.drawString(30 * mm, h - 30 * mm, "Mapa da rede com trecho e chaves destacadas")

    c.drawImage(img, x, y - 10 * mm, draw_w, draw_h, preserveAspectRatio=True, mask="auto")
    c.showPage()

    # -----------------------------------------------------
    # Página 4 – Linha do tempo da manobra
    # -----------------------------------------------------
    c.setFont("Helvetica-Bold", 14)
    c.drawString(30 * mm, h - 30 * mm, "Linha do tempo da manobra – passo a passo")

    y = h - 45 * mm
    c.setFont("Helvetica", 11)

    step = 1
    c.drawString(30 * mm, y, f"{step}️⃣  Abrir NF isoladora {nf1} para isolar o trecho.")
    y -= 7 * mm
    step += 1

    if na:
        c.drawString(30 * mm, y, f"{step}️⃣  Fechar NA {na} para restabelecer carga em parte da rede.")
        y -= 7 * mm
        step += 1

    if nf_block:
        c.drawString(
            30 * mm, y,
            f"{step}️⃣  Abrir NF de bloqueio {nf_block} para impedir dupla alimentação."
        )
        y -= 7 * mm
        step += 1

    c.drawString(
        30 * mm, y,
        f"{step}️⃣  Executar os procedimentos de segurança e realizar a manutenção no trecho isolado."
    )
    y -= 7 * mm
    step += 1

    c.drawString(
        30 * mm, y,
        f"{step}️⃣  Ao término da manutenção, recompor a rede fechando as chaves de forma "
        "coordenada segundo o POP da distribuidora."
    )

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.getvalue()


# =========================================================
# CARREGAMENTO DO BANCO E STATUS
# =========================================================
st.sidebar.header("📂 Dados do banco")

if not DB_PATH.exists():
    st.sidebar.error(f"Banco `{DB_PATH.name}` não encontrado na pasta do app.")
    st.stop()

tabelas = listar_tabelas()
st.sidebar.write("Banco:", f"`{DB_PATH.name}`")
st.sidebar.write("TOPOLOGY:", "✅" if "topology" in tabelas else "❌")
st.sidebar.write("COORDS:", "✅" if "coords" in tabelas else "❌")
st.sidebar.write("LOADS:", "✅" if "loads" in tabelas else "❌")
st.sidebar.write("VAO_MAP (Modo 1):", "✅" if "vao_map" in tabelas else "❌")
st.sidebar.write("NF_MAP (Modo 1):", "✅" if "nf_map" in tabelas else "❌")
st.sidebar.write("NF_NA_NF (Modo 2):", "✅" if "nf_na_nf" in tabelas else "❌")

nome_operador = st.sidebar.text_input("👤 Nome do operador", value="")

coords = carregar_coords()
topo = carregar_topologia()
loads = carregar_loads()
vao_map = carregar_vao_map()
nf_map_data = carregar_nf_map()
df_nf_na_nf = carregar_nf_na_nf()

if not coords or not topo:
    st.error("Banco encontrado, mas `coords` ou `topology` estão vazios.")
    st.stop()

# =========================================================
# DESCRIÇÃO RÁPIDA
# =========================================================
st.markdown(
    """
Ferramenta de apoio à manobra de **desligamento programado** e **manobras com duas fontes**
em redes de distribuição, baseada no alimentador teste **IEEE-123 Bus**.

A inteligência de isolamento e impacto de manobras foi calculada previamente em
**OpenDSS + Python (Colab)** e gravada em um banco **SQLite** (`ieee123_manobras.db`).

- **Modo 1 – Fonte única**: isolamento real por vãos U–V usando chaves NF.  
- **Modo 2 – Duas fontes**: busca de combinações NF–NA–NF pré-simuladas,
  incluindo **impacto em carga, barras, linhas e limites elétricos**.

No Modo 2, é possível gerar um **Relatório em PDF** com:
resumo, tabelas, gráfico do mapa e linha do tempo da manobra.
"""
)

st.markdown("---")

# =========================================================
# MAPA BASE
# =========================================================
st.subheader("🗺️ Mapa Base da Rede (IEEE-123 Bus) – com nomes de barras e linhas")
fig_base = construir_mapa_base(coords, topo, show_bus_labels=True, show_line_labels=True)
st.plotly_chart(fig_base, use_container_width=True)

st.markdown("---")

# =========================================================
# SELEÇÃO DE MODO
# =========================================================
modo = st.radio(
    "Selecione o modo de operação:",
    (
        "Modo 1 – Fonte única (isolamento por vãos)",
        "Modo 2 – Duas fontes (NF–NA–NF)",
    ),
)

# =========================================================
# MODO 1 – FONTE ÚNICA (APP ORIGINAL)
# =========================================================
if modo.startswith("Modo 1"):

    if not vao_map or not nf_map_data:
        st.error(
            "Tabelas `vao_map` e/ou `nf_map` não estão disponíveis no banco. "
            "Modo 1 indisponível neste banco."
        )
    else:
        st.sidebar.subheader("🔧 Vão simples (U-V) – MODO 1")

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
                st.error(f"Não há NF cadastrada no banco para o vão {u_simples} – {v_simples}.")
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

        # -----------------------------
        # TRECHO COM MÚLTIPLOS VÃOS
        # -----------------------------
        st.subheader("🧩 Trecho com múltiplos vãos (Modo 1 – entrada em blocos de 2 barras)")

        entrada_seq = st.text_input(
            "Sequência de barras (ex: 60,62,63,64,65,66,60,67)",
            value="60,62,63,64,65,66,60,67",
            key="entrada_seq_m1",
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
                        st.markdown("### ⚡ Impacto consolidado da manobra (sem dupla contagem)")

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
# MODO 2 – DUAS FONTES (NF–NA–NF + PDF)
# =========================================================
else:
    st.subheader("⚡ Modo 2 – Manobra com Duas Fontes (NF–NA–NF)")

    if df_nf_na_nf.empty:
        st.error(
            "Tabela `nf_na_nf` não encontrada ou vazia no banco. "
            "Não é possível utilizar o Modo 2."
        )
    else:
        # Estado interno para manter resultados entre interações
        if "modo2_state" not in st.session_state:
            st.session_state["modo2_state"] = {}

        st.markdown(
            """
Informe as **LINHAS** do trecho a ser desligado (ex: `L70` ou `L70,L71`).
A plataforma irá buscar, no banco pré-simulado, as combinações NF–NA–NF que
desligam exatamente esse trecho, ordenando pelas de **menor carga desligada**.
"""
        )

        linhas_input = st.text_input(
            "Digite as LINHAS a serem desligadas (ex: L70 ou L70,L71):",
            value="L70",
            key="linhas_modo2",
        )

        if st.button("Calcular manobras – Modo 2"):
            line_names_raw = [s.strip() for s in linhas_input.split(",") if s.strip()]
            trecho_buses, trecho_lines_norm = get_trecho_buses_from_lines_modo2(
                line_names_raw, topo
            )

            st.session_state["modo2_state"] = {
                "line_names_raw": line_names_raw,
                "trecho_buses": trecho_buses,
                "trecho_lines": trecho_lines_norm,
            }

            if not trecho_buses:
                st.warning(
                    "Não foi possível determinar as barras do trecho com as linhas informadas. "
                    "Verifique se os nomes das linhas existem na topologia."
                )
            else:
                st.success(
                    f"Trecho (barras) considerado: {sorted(trecho_buses)}"
                )

                subset = filtrar_combos_para_trecho(df_nf_na_nf, trecho_buses)
                if subset.empty:
                    st.warning(
                        "Nenhuma combinação NF–NA–NF do banco desliga exatamente esse trecho."
                    )
                else:
                    st.session_state["modo2_state"]["options_df"] = subset

        # Se já houver resultados armazenados:
        state = st.session_state.get("modo2_state", {})
        subset = state.get("options_df", None)
        trecho_buses = state.get("trecho_buses", set())
        trecho_lines_norm = state.get("trecho_lines", [])

        if subset is not None and not subset.empty and trecho_buses:
            st.markdown(
                f"**Trecho (barras) considerado:** `{sorted(trecho_buses)}`  \n"
                f"**Linhas do trecho:** `{', '.join(trecho_lines_norm)}`"
            )

            st.markdown("### ✅ Opções de manobra (ordenadas por menor carga desligada):")

            option_labels = []
            for idx, row in subset.reset_index(drop=True).iterrows():
                label = (
                    f"Opção {idx + 1}: "
                    f"NF1={row['nf1']}, "
                    f"NA={normalize_str_none(row['na']) or '-'}, "
                    f"NF_bloq={normalize_str_none(row['nf_block']) or '-'}, "
                    f"Carga desligada={row['kw_off']:.1f} kW, "
                    f"Manobras={row['n_manobras']}"
                )
                option_labels.append(label)

            selected_idx = st.radio(
                "Selecione a opção para detalhar:",
                options=list(range(len(option_labels))),
                format_func=lambda i: option_labels[i],
                index=0,
                key="modo2_radio_idx",
            )

            selected_row = subset.reset_index(drop=True).iloc[selected_idx]

            st.markdown("### 🔍 Detalhamento da opção selecionada")

            nf1 = str(selected_row["nf1"])
            na = normalize_str_none(selected_row.get("na"))
            nf_block = normalize_str_none(selected_row.get("nf_block"))
            kw_off = float(selected_row.get("kw_off", 0.0))
            n_manobras = int(selected_row.get("n_manobras", 0))
            vmin_pu = float(selected_row.get("vmin_pu", 0.0))
            vmax_pu = float(selected_row.get("vmax_pu", 0.0))
            max_loading = float(selected_row.get("max_loading", 0.0))

            st.write(
                f"**NF isoladora:** `{nf1}`  |  "
                f"**NA restabelecimento:** `{na or '-'}`  |  "
                f"**NF bloqueio:** `{nf_block or '-'}`  \n"
                f"**Carga desligada total:** `{kw_off:.1f} kW`  |  "
                f"**Nº de manobras:** `{n_manobras}`  \n"
                f"**Vmin (pu):** `{vmin_pu:.3f}`  |  "
                f"**Vmax (pu):** `{vmax_pu:.3f}`  |  "
                f"**Carregamento máximo (pu):** `{max_loading:.3f}`"
            )

            # Mapa específico da opção
            fig_m2 = plotar_mapa_modo2(
                coords,
                topo,
                trecho_lines_norm,
                selected_row,
            )
            st.markdown("### 🗺️ Mapa da rede para a opção selecionada")
            st.plotly_chart(fig_m2, use_container_width=True)

            # Geração de PDF com gráfico embutido
            st.markdown("### 📄 Relatório em PDF da manobra")

            if not nome_operador.strip():
                st.warning(
                    "Informe o **nome do operador** na barra lateral para habilitar o PDF."
                )
            else:
                trecho_desc = ", ".join(sorted(trecho_buses))
                pdf_bytes = gerar_pdf_modo2(
                    nome_operador=nome_operador,
                    trecho_desc=trecho_desc,
                    trecho_barras=sorted(trecho_buses),
                    trecho_lines=trecho_lines_norm,
                    option_row=selected_row,
                    fig=fig_m2,
                )
                safe_name = nome_operador.strip().replace(" ", "_")
                file_name = f"Relatorio_{safe_name}_{date.today().isoformat()}.pdf"

                st.download_button(
                    "⬇️ Baixar relatório em PDF",
                    data=pdf_bytes,
                    file_name=file_name,
                    mime="application/pdf",
                )

            st.markdown("### 📜 Linha de tempo (Modo 2 – resumo textual)")

            steps = []
            steps.append(f"1️⃣ Abrir NF isoladora **{nf1}** para isolar o trecho.")
            if na:
                steps.append(f"2️⃣ Fechar NA **{na}** para restabelecer carga em parte da rede.")
            if nf_block:
                n_ord = len(steps) + 1
                steps.append(
                    f"{n_ord}️⃣ Abrir NF de bloqueio **{nf_block}** para impedir dupla alimentação."
                )
            n_ord = len(steps) + 1
            steps.append(
                f"{n_ord}️⃣ Executar a manutenção no trecho isolado, conforme os procedimentos de segurança."
            )
            n_ord += 1
            steps.append(
                f"{n_ord}️⃣ Recompor a rede fechando as chaves conforme o POP da distribuidora."
            )

            for s in steps:
                st.markdown(s)
