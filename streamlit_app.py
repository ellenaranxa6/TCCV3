import streamlit as st
import plotly.graph_objects as go
import sqlite3
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set
import ast  # interpretar listas salvas como texto

# =========================================================
# CONFIGURAÇÃO INICIAL
# =========================================================
st.set_page_config(
    page_title="Isolamento Real IEEE-123 Bus",
    layout="wide"
)

st.title("⚡ Plataforma Interativa – Isolamento Real IEEE 123 Bus")

BASE_DIR = Path(__file__).parent

# Bancos de dados
DB_MODO1 = BASE_DIR / "ieee123_isolamento.db"   # Fonte única
DB_MODO2 = BASE_DIR / "ieee123_duasfontes.db"   # Duas fontes


# =========================================================
# FUNÇÕES AUXILIARES – CONEXÕES
# =========================================================
def get_connection_m1() -> sqlite3.Connection:
    return sqlite3.connect(DB_MODO1)


def get_connection_m2() -> sqlite3.Connection:
    return sqlite3.connect(DB_MODO2)


def listar_tabelas(conn: sqlite3.Connection) -> List[str]:
    cur = conn.cursor()
    cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    rows = [r[0] for r in cur.fetchall()]
    return rows


# =========================================================
# =======================  MODO 1  ========================
# ===============  Fonte Única (1 banco)  =================
# =========================================================

@st.cache_data(show_spinner=False)
def carregar_coords_m1() -> Dict[str, Tuple[float, float]]:
    conn = get_connection_m1()
    cur = conn.cursor()
    cur.execute("SELECT bus, x, y FROM coords")
    rows = cur.fetchall()
    conn.close()
    return {str(b): (float(x), float(y)) for b, x, y in rows}


@st.cache_data(show_spinner=False)
def carregar_topologia_m1():
    conn = get_connection_m1()
    cur = conn.cursor()
    cur.execute(
        "SELECT line, from_bus, to_bus, is_switch, norm "
        "FROM topology"
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
def carregar_vao_map_m1():
    conn = get_connection_m1()
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
def carregar_loads_m1():
    conn = get_connection_m1()
    cur = conn.cursor()
    try:
        cur.execute("SELECT bus, kw FROM loads")
        rows = cur.fetchall()
    except sqlite3.OperationalError:
        rows = []
    conn.close()
    return {str(b): float(kw) for b, kw in rows}


@st.cache_data(show_spinner=False)
def carregar_nf_map_m1():
    conn = get_connection_m1()
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


# ---------- Funções de processamento – Modo 1 ----------

def identificar_vaos_blocos(lista_barras: List[str]) -> List[Tuple[str, str]]:
    vaos = []
    for i in range(0, len(lista_barras), 2):
        if i + 1 < len(lista_barras):
            u = lista_barras[i].strip()
            v = lista_barras[i + 1].strip()
            if u and v:
                vaos.append((u, v))
    return vaos


def buscar_nf_para_vao_m1(
    u: str,
    v: str,
    vao_map: List[Dict]
) -> Optional[Dict]:
    candidatos = [
        registro for registro in vao_map
        if (registro["u_bus"] == u and registro["v_bus"] == v)
        or (registro["u_bus"] == v and registro["v_bus"] == u)
    ]
    if not candidatos:
        return None

    candidatos.sort(key=lambda r: (r["kw"], r["n_barras"]))
    return candidatos[0]


def impacto_consolidado_m1(
    lista_nf: List[str],
    loads: Dict[str, float],
    nf_map_data: Dict[str, Dict]
) -> Tuple[float, int, List[str]]:
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


# ---------- Funções de plot – usadas pelos dois modos ----------

def construir_mapa_base(coords, topo):
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


def plotar_mapa_com_trecho_m1(
    coords,
    topo,
    vaos: List[Tuple[str, str]],
    info_vaos: List[Dict],
):
    fig = construir_mapa_base(coords, topo)

    # destaques dos vãos
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

    # NF associadas
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


# ---------- Interface / lógica do Modo 1 ----------

def run_modo1():
    st.markdown("### 🟢 Modo 1 – Fonte Única (banco `ieee123_isolamento.db`)")

    if not DB_MODO1.exists():
        st.error("Banco `ieee123_isolamento.db` não encontrado na pasta do app.")
        return

    coords = carregar_coords_m1()
    topo = carregar_topologia_m1()
    vao_map = carregar_vao_map_m1()
    loads = carregar_loads_m1()
    nf_map_data = carregar_nf_map_m1()

    if not coords or not topo or not vao_map:
        st.error("Banco do Modo 1 encontrado, mas alguma tabela essencial está vazia.")
        return

    # Mapa base
    st.subheader("🗺️ Mapa Base da Rede (Modo 1)")
    fig_base = construir_mapa_base(coords, topo)
    st.plotly_chart(fig_base, use_container_width=True)
    st.markdown("---")

    # ----- Vão simples -----
    st.sidebar.subheader("🔧 Vão simples (U-V) – Modo 1")

    lista_barras = sorted(
        coords.keys(),
        key=lambda x: (x.isdigit(), int(x) if x.isdigit() else x),
    )

    u_simples = st.sidebar.selectbox("Barra U (Modo 1)", lista_barras, key="m1_u")
    v_simples = st.sidebar.selectbox("Barra V (Modo 1)", lista_barras, key="m1_v")

    if st.sidebar.button("Confirmar vão simples (Modo 1)"):
        vao_simples = (u_simples, v_simples)
        info = buscar_nf_para_vao_m1(u_simples, v_simples, vao_map)

        st.subheader("🔎 Resultado – Vão simples (Modo 1)")

        if info is None:
            st.error(f"Não há NF cadastrada no banco para o vão {u_simples} – {v_simples}.")
        else:
            st.success(
                f"**Melhor NF:** `{info['nf']}`  |  "
                f"**Carga interrompida (NF isolada):** {info['kw']:.1f} kW  |  "
                f"**Barras isoladas:** {info['n_barras']}"
            )

            fig_vao = plotar_mapa_com_trecho_m1(
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

    # ----- Multi-vãos -----
    st.subheader("🧩 Trecho com múltiplos vãos (Modo 1)")

    entrada_seq = st.text_input(
        "Sequência de barras (ex: 60,62,63,64,65,66,60,67)",
        value="60,62,63,64,65,66,60,67",
        key="m1_seq",
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
                st.markdown("### 🔍 Vãos identificados (Modo 1):")
                st.write(vaos)

                info_vaos = []
                nao_encontrados = []

                for u, v in vaos:
                    info = buscar_nf_para_vao_m1(u, v, vao_map)
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

                    fig_multi = plotar_mapa_com_trecho_m1(
                        coords,
                        topo,
                        vaos=[(d["u"], d["v"]) for d in info_vaos],
                        info_vaos=info_vaos,
                    )
                    st.markdown("### 🗺️ Mapa com trecho e NFs destacadas (Modo 1)")
                    st.plotly_chart(fig_multi, use_container_width=True)

                    # Impacto consolidado
                    st.markdown("### ⚡ Impacto consolidado da manobra (Modo 1)")

                    lista_nf_ordenada: List[str] = []
                    for d in info_vaos:
                        if d["nf"] not in lista_nf_ordenada:
                            lista_nf_ordenada.append(d["nf"])

                    if not nf_map_data:
                        st.warning(
                            "Tabela `nf_map` não está disponível no banco do Modo 1. "
                            "Não foi possível calcular o impacto consolidado."
                        )
                    else:
                        kw_total, n_barras_unicas, barras_ordenadas = impacto_consolidado_m1(
                            lista_nf_ordenada, loads, nf_map_data
                        )

                        st.success(
                            f"**Carga total interrompida:** {kw_total:.1f} kW  \n"
                            f"**Barras desenergizadas únicas:** {n_barras_unicas}"
                        )

                        with st.expander("Ver barras desenergizadas únicas (Modo 1)"):
                            st.write(barras_ordenadas)

                    st.markdown("### 📜 Linha de tempo de manobra (Modo 1)")

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
# =======================  MODO 2  ========================
# ===============  Duas Fontes (2º banco)  ================
# =========================================================

@st.cache_data(show_spinner=False)
def carregar_coords_m2() -> Dict[str, Tuple[float, float]]:
    conn = get_connection_m2()
    cur = conn.cursor()
    cur.execute("SELECT bus, x, y FROM coords")
    rows = cur.fetchall()
    conn.close()
    return {str(b): (float(x), float(y)) for b, x, y in rows}


@st.cache_data(show_spinner=False)
def carregar_topologia_m2():
    conn = get_connection_m2()
    cur = conn.cursor()
    cur.execute(
        "SELECT line, from_bus, to_bus, is_switch, norm "
        "FROM topology"
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
def carregar_loads_m2():
    conn = get_connection_m2()
    cur = conn.cursor()
    try:
        cur.execute("SELECT bus, kw FROM loads")
        rows = cur.fetchall()
    except sqlite3.OperationalError:
        rows = []
    conn.close()
    return {str(b): float(kw) for b, kw in rows}


@st.cache_data(show_spinner=False)
def carregar_nf_na_nf_m2():
    """
    Tabela nf_na_nf(nf1, na, nf_block, kw_off, n_manobras, buses_off TEXT)

    buses_off está salva como string de lista Python.
    """
    conn = get_connection_m2()
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT nf1, na, nf_block, kw_off, n_manobras, buses_off "
            "FROM nf_na_nf"
        )
        rows = cur.fetchall()
    except sqlite3.OperationalError:
        rows = []
    conn.close()

    registros = []
    for nf1, na, nf_block, kw, n_man, buses_str in rows:
        buses_set: Set[str] = set()
        if buses_str:
            try:
                lista = ast.literal_eval(buses_str)
                for b in lista:
                    buses_set.add(str(b).strip())
            except Exception:
                for b in str(buses_str).replace("[", "").replace("]", "").replace('"', "").split(","):
                    b = b.strip()
                    if b:
                        buses_set.add(b)

        registros.append(
            dict(
                nf1=str(nf1),
                na=str(na) if na is not None else None,
                nf_block=str(nf_block) if nf_block is not None else None,
                kw_off=float(kw),
                n_manobras=int(n_man),
                buses_off=buses_set,
            )
        )
    return registros


def buscar_combos_para_trecho_m2(
    trecho_buses: Set[str],
    combos: List[Dict],
    max_opcoes: int = 3,
) -> List[Dict]:
    """
    Filtra na tabela nf_na_nf os cenários que desligam TODO o trecho
    (trecho_buses ⊆ buses_off) e ordena por menor carga e nº de manobras.
    """
    if not trecho_buses:
        return []

    candidatos = []
    for reg in combos:
        if trecho_buses.issubset(reg["buses_off"]):
            candidatos.append(reg)

    if not candidatos:
        return []

    candidatos_sorted = sorted(
        candidatos,
        key=lambda r: (r["kw_off"], r["n_manobras"])
    )

    # Remove duplicatas exatas (nf1, na, nf_block)
    uniq = []
    seen = set()
    for reg in candidatos_sorted:
        key = (reg["nf1"], reg["na"], reg["nf_block"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(reg)
        if len(uniq) >= max_opcoes:
            break

    return uniq


def plotar_mapa_m2(
    coords,
    topo,
    trecho_buses: Set[str],
    scenario: Dict,
):
    """
    Plota:
      - barras desligadas pelo cenário (vermelho)
      - trecho informado (amarelo)
      - demais barras (verde)
    As linhas ficam em cinza (sem realce de NF/NA para evitar
    problemas de nomenclatura).
    """
    fig = construir_mapa_base(coords, topo)

    # Cores por barra
    node_x = []
    node_y = []
    node_color = []
    node_text = []

    buses_off = scenario["buses_off"]

    for bus, (x, y) in coords.items():
        node_x.append(x)
        node_y.append(y)
        node_text.append(bus)

        if bus in trecho_buses:
            node_color.append("yellow")      # trecho alvo
        elif bus in buses_off:
            node_color.append("red")         # desenergizada
        else:
            node_color.append("green")       # energizada

    fig.data = []  # limpa traces anteriores (edges serão redesenhadas)

    # redesenha linhas
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

    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=node_text,
            textposition="top center",
            marker=dict(size=7, color=node_color),
            name="Barras",
            hovertemplate="Barra %{text}<extra></extra>",
        )
    )

    fig.update_layout(
        height=600,
        showlegend=False,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        title=(
            f"NF1={scenario['nf1']}  |  NA={scenario['na']}  |  "
            f"NF_bloq={scenario['nf_block']}  |  "
            f"Carga desligada={scenario['kw_off']:.1f} kW"
        ),
    )

    return fig


def run_modo2():
    st.markdown("### 🟠 Modo 2 – Duas Fontes (banco `ieee123_duasfontes.db`)")

    if not DB_MODO2.exists():
        st.error("Banco `ieee123_duasfontes.db` não encontrado na pasta do app.")
        return

    coords = carregar_coords_m2()
    topo = carregar_topologia_m2()
    loads = carregar_loads_m2()
    combos = carregar_nf_na_nf_m2()

    if not coords or not topo or not combos:
        st.error("Banco do Modo 2 encontrado, mas alguma tabela essencial está vazia.")
        return

    st.subheader("🗺️ Mapa Base da Rede (Modo 2)")
    fig_base = construir_mapa_base(coords, topo)
    st.plotly_chart(fig_base, use_container_width=True)
    st.markdown("---")

    # Entrada do trecho por BARRAS (simples e robusto)
    st.subheader("🧩 Trecho alvo (barras) – Modo 2")

    entrada_barras = st.text_input(
        "Lista de barras do trecho a desenergizar (ex: 69,70 ou 69,70,71)",
        value="69,70",
        key="m2_barras",
    )

    if st.button("Buscar opções de manobra (Modo 2)"):
        barras_raw = [b.strip() for b in entrada_barras.split(",") if b.strip()]
        trecho_buses = set(barras_raw)

        if not trecho_buses:
            st.error("Informe ao menos uma barra para o trecho.")
            return

        st.write(f"Trecho considerado (barras): {sorted(trecho_buses)}")

        opcoes = buscar_combos_para_trecho_m2(trecho_buses, combos)

        if not opcoes:
            st.error("Nenhum registro encontrado no banco para os vãos/barras informados.")
            return

        st.markdown("### ✅ Top opções de manobra (Modo 2)")

        linhas_tabela = []
        for i, opt in enumerate(opcoes, start=1):
            linhas_tabela.append(
                {
                    "Opção": i,
                    "NF isoladora (NF1)": opt["nf1"],
                    "NA de restabelecimento": opt["na"],
                    "NF bloqueio": opt["nf_block"],
                    "kW desligados": opt["kw_off"],
                    "Nº manobras": opt["n_manobras"],
                }
            )
        st.table(linhas_tabela)

        # Exibe mapas das opções
        st.markdown("### 🗺️ Mapas das opções (barras energizadas x desligadas)")
        for i, opt in enumerate(opcoes, start=1):
            st.markdown(
                f"**Opção {i}** – NF1={opt['nf1']}  |  NA={opt['na']}  |  "
                f"NF_bloq={opt['nf_block']}  |  "
                f"Desligado={opt['kw_off']:.1f} kW"
            )
            fig_opt = plotar_mapa_m2(coords, topo, trecho_buses, opt)
            st.plotly_chart(fig_opt, use_container_width=True)

# =========================================================
# ========================= MAIN ==========================
# =========================================================

st.sidebar.header("🧭 Seleção de Modo")

modo = st.sidebar.radio(
    "Escolha o modo de operação:",
    options=["Modo 1 – Fonte única", "Modo 2 – Duas fontes"],
)

if modo == "Modo 1 – Fonte única":
    run_modo1()
else:
    run_modo2()
