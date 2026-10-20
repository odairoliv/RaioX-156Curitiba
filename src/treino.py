"""Atalhos para treinar os modelos com os dados atuais da sessão (usados pelas páginas)."""
import streamlit as st

from src import estado
from src.modelos.classificacao import treinar_classificador
from src.modelos.previsao import treinar_previsao
from src.modelos.registro import salvar


def treinar_tendencia(modelo_nome: str, parametros: dict):
    resultado = treinar_previsao(estado.serie(), modelo_nome, parametros, st.session_state.versao_dados)
    salvar(resultado)
    estado.guardar_modelo(resultado)
    return resultado


def treinar_atraso(modelo_nome: str, parametros: dict, tamanho_amostra: int):
    resultado = treinar_classificador(estado.base_atraso(), modelo_nome, parametros,
                                      st.session_state.versao_dados, estado.periodo_chamados(), tamanho_amostra)
    resultado.detalhes["tamanho_amostra"] = tamanho_amostra
    salvar(resultado)
    estado.guardar_modelo(resultado)
    return resultado


def retreinar_modelos_existentes() -> list[str]:
    """Re-treina, com os dados novos, os modelos já treinados (mesmo algoritmo e parâmetros)."""
    retreinados = []
    for tipo, anterior in list(st.session_state.modelos.items()):
        if tipo == "previsao":
            treinar_tendencia(anterior.modelo_nome, anterior.parametros)
        else:
            treinar_atraso(anterior.modelo_nome, anterior.parametros, anterior.detalhes.get("tamanho_amostra", 40_000))
        retreinados.append(f"{anterior.modelo_nome} ({'tendência' if tipo == 'previsao' else 'atraso'})")
    return retreinados
