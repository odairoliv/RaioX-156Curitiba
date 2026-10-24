"""Atalhos para treinar os modelos com os dados atuais da sessão (usados pelas páginas)."""
import streamlit as st

from src import estado
from src.modelos.classificacao import treinar_classificador
from src.modelos.regressao import treinar_regressao
from src.modelos.registro import salvar

TREINADORES = {"classificacao": treinar_classificador, "regressao": treinar_regressao}


def treinar(tipo: str, modelo_nome: str, parametros: dict, tamanho_amostra: int, caracteristicas: list[str]):
    """Treina o classificador ou o regressor, salva em modelos_salvos/ e guarda na sessão."""
    resultado = TREINADORES[tipo](estado.base(), modelo_nome, parametros, st.session_state.versao_dados,
                                  estado.periodo_chamados(), tamanho_amostra, caracteristicas)
    salvar(resultado)
    estado.guardar_modelo(resultado)
    return resultado


def retreinar_modelos_existentes() -> list[str]:
    """Re-treina, com os dados novos, os modelos já treinados (mesmo algoritmo, parâmetros e características)."""
    nomes = {"classificacao": "vai atrasar?", "regressao": "quantos dias?"}
    feitos = []
    for tipo, anterior in list(st.session_state.modelos.items()):
        treinar(tipo, anterior.modelo_nome, anterior.parametros, anterior.detalhes["tamanho_amostra"],
                anterior.detalhes["caracteristicas"])
        feitos.append(f"{anterior.modelo_nome} ({nomes[tipo]})")
    return feitos
