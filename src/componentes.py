"""Pedaços de interface reaproveitados em várias páginas."""
import pandas as pd
import streamlit as st

from src import estado
from src.graficos import ICONES_SITUACAO
from src.indicadores import DIMENSOES, PRESETS_PESOS

ss = st.session_state


# ---------- Formatação no padrão brasileiro ----------

def num(valor, casas: int = 0) -> str:
    """1234.5 -> '1.234,5'."""
    if valor is None or pd.isna(valor):
        return "—"
    texto = f"{valor:,.{casas}f}"
    return texto.replace(",", "X").replace(".", ",").replace("X", ".")


def pct(valor, casas: int = 1) -> str:
    return "—" if valor is None or pd.isna(valor) else f"{num(valor, casas)}%"


def data_br(data) -> str:
    return "—" if data is None or pd.isna(data) else pd.Timestamp(data).strftime("%d/%m/%Y")


def situacao_com_icone(situacao: str) -> str:
    return f"{ICONES_SITUACAO.get(situacao, '')} {situacao}"


# ---------- Pesos da nota ----------

def _aplicar_preset() -> None:
    for chave, peso in PRESETS_PESOS[ss.preset_pesos].items():
        ss[f"peso_{chave}"] = float(peso)


def seletor_pesos() -> dict:
    """Barra lateral: perfil de uso + ajuste fino dos pesos. Devolve {dimensão: peso}."""
    if "preset_pesos" not in ss:
        ss.preset_pesos = "Equilibrado"
        _aplicar_preset()
    with st.sidebar:
        st.subheader("Como calcular a nota")
        st.selectbox("Perfil", list(PRESETS_PESOS), key="preset_pesos", on_change=_aplicar_preset,
                     help="Cada perfil dá mais importância a dimensões diferentes.")
        with st.expander("Ajustar pesos manualmente"):
            for d in DIMENSOES:
                st.slider(d.nome, 0.0, 3.0, step=0.5, key=f"peso_{d.chave}", help=d.descricao)
    return {d.chave: ss[f"peso_{d.chave}"] for d in DIMENSOES}


# ---------- Status dos dados ----------

def status_dados_na_barra_lateral() -> None:
    inicio, fim = estado.periodo_chamados()
    with st.sidebar:
        st.caption(f"📂 **Dados atuais:** {num(len(estado.chamados()))} chamados do 156, "
                   f"de {data_br(inicio)} a {data_br(fim)}. Versão `{ss.versao_dados}`.")


# ---------- Controles dos modelos ----------

def controles_modelo(catalogo: dict, prefixo: str) -> tuple[str, dict]:
    """Escolha do algoritmo + um controle para cada parâmetro (lidos do catálogo)."""
    escolha, ajustes = st.columns([1, 2])
    nome = escolha.selectbox("Algoritmo", list(catalogo), key=f"{prefixo}_algoritmo")
    escolha.caption(catalogo[nome]["descricao"])
    parametros = {}
    definicoes = catalogo[nome]["parametros"]
    if not definicoes:
        ajustes.caption("Este algoritmo não tem parâmetros para ajustar.")
    colunas = ajustes.columns(min(len(definicoes), 3) or 1)
    for p, coluna in zip(definicoes, colunas * 3):
        chave = f"{prefixo}_{nome}_{p['nome']}"
        if p["tipo"] == "int":
            valor = coluna.slider(p["rotulo"], p["min"], p["max"], p["padrao"], key=chave, help=p["ajuda"])
        elif p["tipo"] == "float":
            valor = coluna.number_input(p["rotulo"], p["min"], p["max"], p["padrao"], step=p["passo"],
                                        key=chave, help=p["ajuda"])
        else:
            valor = coluna.selectbox(p["rotulo"], p["opcoes"], index=p["opcoes"].index(p["padrao"]),
                                     key=chave, help=p["ajuda"])
        parametros[p["nome"]] = valor
    return nome, parametros


def aviso_modelo(tipo: str) -> None:
    """Mostra quando o modelo foi treinado e avisa se os dados mudaram desde então."""
    resultado = estado.modelo(tipo)
    if resultado is None:
        st.info("Nenhum modelo treinado ainda nesta sessão. Escolha o algoritmo acima e clique em **Treinar**.")
        return
    texto = (f"Modelo atual: **{resultado.modelo_nome}**, treinado em {resultado.treinado_em:%d/%m/%Y %H:%M} "
             f"com {num(resultado.n_registros)} registros ({data_br(resultado.periodo[0])} a {data_br(resultado.periodo[1])}).")
    if estado.modelo_desatualizado(tipo):
        st.warning("⚠️ Novos dados foram carregados depois deste treino. " + texto + " Clique em **Re-treinar**.")
    else:
        st.success(texto)
