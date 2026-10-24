"""Pedaços de interface reaproveitados em várias páginas."""
import pandas as pd
import streamlit as st

from src import estado
from src.modelos.comum import CARACTERISTICAS, GRUPOS

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
        st.info("Nenhum modelo treinado ainda nesta sessão. Configure acima e clique em **Treinar**.")
        return
    texto = (f"Modelo atual: **{resultado.modelo_nome}**, treinado em {resultado.treinado_em:%d/%m/%Y %H:%M} "
             f"com {num(resultado.n_registros)} registros ({data_br(resultado.periodo[0])} a {data_br(resultado.periodo[1])}).")
    if estado.modelo_desatualizado(tipo):
        st.warning("⚠️ Novos dados foram carregados depois deste treino. " + texto + " Clique em **Re-treinar**.")
    else:
        st.success(texto)


def seletor_caracteristicas(prefixo: str) -> list[str]:
    """Escolha dos grupos de características usados pelo modelo (ligados às hipóteses)."""
    grupos = st.multiselect(
        "Características usadas pelo modelo", list(GRUPOS), default=list(GRUPOS), key=f"{prefixo}_grupos",
        help="Treine só com 'Onde' e depois só com 'O quê' e compare as métricas: o grupo que prevê melhor é o que "
             "mais explica o atraso (H1 x H2).")
    escolhidas = [c for g in grupos for c in GRUPOS[g]] or list(CARACTERISTICAS)
    if not grupos:
        st.caption("Nenhum grupo escolhido: o modelo usará todas as características.")
    return escolhidas


def nomes_caracteristicas(caracteristicas: list[str]) -> str:
    return ", ".join(CARACTERISTICAS[c].lower() for c in caracteristicas)
