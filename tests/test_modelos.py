"""Treino, avaliação e previsão dos dois modelos."""
import pytest

from src.modelos.catalogo import MODELOS_CLASSIFICACAO, MODELOS_REGRESSAO, parametros_padrao
from src.modelos.classificacao import prever_chamado, treinar_classificador
from src.modelos.comum import GRUPOS
from src.modelos.regressao import prever_dias, treinar_regressao


@pytest.mark.parametrize("nome", list(MODELOS_CLASSIFICACAO))
def test_todos_os_classificadores_treinam(base, nome):
    resultado = treinar_classificador(base, nome, parametros_padrao(MODELOS_CLASSIFICACAO, nome), "t", (None, None),
                                      tamanho_amostra=8_000)
    assert 0 <= resultado.metricas["f1"] <= 1
    assert 0 <= prever_chamado(resultado.pipeline, base.iloc[0].to_dict()) <= 1


@pytest.mark.parametrize("nome", list(MODELOS_REGRESSAO))
def test_todos_os_regressores_treinam_e_superam_o_chute(base, nome):
    resultado = treinar_regressao(base, nome, parametros_padrao(MODELOS_REGRESSAO, nome), "t", (None, None),
                                  tamanho_amostra=20_000)
    assert resultado.metricas["mae"] < resultado.metricas["mae_chute"]
    assert prever_dias(resultado.pipeline, base.iloc[0].to_dict()) >= 0


def test_modelo_com_poucas_caracteristicas(base):
    so_onde = GRUPOS["Onde (bairro e regional) — H1"]
    resultado = treinar_classificador(base, "Regressão Logística", {"C": 1.0, "class_weight": "balanced"}, "t",
                                      (None, None), tamanho_amostra=8_000, caracteristicas=so_onde)
    assert resultado.detalhes["caracteristicas"] == so_onde
    # A previsão aceita um chamado com todas as colunas, mesmo que o modelo use só algumas
    assert 0 <= prever_chamado(resultado.pipeline, base.iloc[0].to_dict()) <= 1
