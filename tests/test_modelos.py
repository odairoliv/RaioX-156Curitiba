"""Treino, avaliação e previsão dos modelos de machine learning."""
import pytest

from src.modelos.catalogo import MODELOS_CLASSIFICACAO, MODELOS_REGRESSAO, parametros_padrao
from src.modelos.classificacao import montar_base_atraso, prever_chamado, treinar_classificador
from src.modelos.previsao import prever_futuro, serie_semanal, treinar_previsao


@pytest.fixture(scope="module")
def serie(bases):
    return serie_semanal(bases["156"])


@pytest.mark.parametrize("nome", list(MODELOS_REGRESSAO))
def test_todos_os_regressores_treinam(serie, nome):
    resultado = treinar_previsao(serie, nome, parametros_padrao(MODELOS_REGRESSAO, nome), "teste")
    assert resultado.metricas["r2"] > 0.5
    assert resultado.metricas["mae"] > 0


def test_previsao_futura_nao_tem_negativos(serie):
    resultado = treinar_previsao(serie, "Regressão Linear", {}, "teste")
    futuro = prever_futuro(resultado.pipeline, serie, 4)
    assert futuro["semana"].nunique() == 4
    assert (futuro["qtd"] >= 0).all()
    assert futuro["semana"].min() > serie["semana"].max()


@pytest.mark.parametrize("nome", list(MODELOS_CLASSIFICACAO))
def test_todos_os_classificadores_treinam(bases, nome):
    base = montar_base_atraso(bases["156"])
    resultado = treinar_classificador(base, nome, parametros_padrao(MODELOS_CLASSIFICACAO, nome), "teste",
                                      (None, None), tamanho_amostra=8_000)
    assert 0 <= resultado.metricas["f1"] <= 1
    probabilidade = prever_chamado(resultado.pipeline, base.iloc[0].to_dict())
    assert 0 <= probabilidade <= 1
