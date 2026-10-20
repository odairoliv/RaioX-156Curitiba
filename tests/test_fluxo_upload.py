"""Fluxo completo da página "Enviar dados": upload de outubro -> base cresce -> modelo fica
desatualizado -> re-treino automático usa os dados novos.

Roda as mesmas funções que a página chama, com o Streamlit em modo "sem servidor".
"""
import streamlit as st

from src import estado, treino


def test_upload_de_novo_mes_e_retreino(pasta_exemplos):
    estado.iniciar()
    linhas_antes = len(estado.chamados())
    fim_antes = estado.periodo_chamados()[1]
    versao_antes = st.session_state.versao_dados

    treino.treinar_tendencia("Regressão Linear", {})
    assert not estado.modelo_desatualizado("previsao")

    arquivo = pasta_exemplos / "156_outubro_2026.csv"
    resumo = estado.registrar_upload(arquivo.name, arquivo.read_bytes(), adicionar_ao_periodo=True)

    assert resumo["tipo_chave"] == "156"
    assert len(estado.chamados()) == linhas_antes + resumo["linhas_validas"]
    assert estado.periodo_chamados()[1] > fim_antes
    assert st.session_state.versao_dados != versao_antes
    assert estado.modelo_desatualizado("previsao")

    retreinados = treino.retreinar_modelos_existentes()
    assert retreinados and not estado.modelo_desatualizado("previsao")
    assert estado.modelo("previsao").periodo[1] > fim_antes
