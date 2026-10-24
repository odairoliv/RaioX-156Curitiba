"""Fluxo da página "Enviar dados": upload de outubro -> base cresce -> modelos ficam
desatualizados -> re-treino usa os dados novos. Roda as mesmas funções que a página chama."""
import streamlit as st

from src import estado, treino
from src.modelos.comum import CARACTERISTICAS


def test_upload_de_novo_mes_e_retreino(pasta_exemplos):
    estado.iniciar()
    linhas_antes = len(estado.chamados())
    fim_antes = estado.periodo_chamados()[1]
    versao_antes = st.session_state.versao_dados
    treino.treinar("classificacao", "Árvore de Decisão", {"max_depth": 5, "min_samples_leaf": 5,
                                                          "class_weight": "balanced"}, 5_000, list(CARACTERISTICAS))
    treino.treinar("regressao", "Regressão Linear", {}, 5_000, list(CARACTERISTICAS))

    arquivo = pasta_exemplos / "156_outubro_2026.csv"
    resumo = estado.registrar_upload(arquivo.name, arquivo.read_bytes(), adicionar_ao_periodo=True)

    assert len(estado.chamados()) == linhas_antes + resumo["linhas_validas"]
    assert estado.periodo_chamados()[1] > fim_antes
    assert st.session_state.versao_dados != versao_antes
    assert estado.modelo_desatualizado("classificacao") and estado.modelo_desatualizado("regressao")

    assert len(treino.retreinar_modelos_existentes()) == 2
    assert not estado.modelo_desatualizado("classificacao") and not estado.modelo_desatualizado("regressao")
    assert estado.modelo("regressao").periodo[1] > fim_antes
