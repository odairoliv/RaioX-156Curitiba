"""Atualização da população e dicionário de indicadores."""
import shutil

import pytest

from src import bairros, estado
from src.dicionario import INDICADORES, comparar_com_bairros, explicar
from src.indicadores import DIMENSAO_POR_CHAVE, calcular_indicadores


@pytest.fixture
def populacao_temporaria(tmp_path, monkeypatch):
    """Usa uma cópia do arquivo de população para o teste não alterar o arquivo real."""
    copia = tmp_path / "populacao_bairros.csv"
    shutil.copy(bairros.ARQUIVO_POPULACAO, copia)
    monkeypatch.setattr(bairros, "ARQUIVO_POPULACAO", copia)
    bairros.tabela_bairros.cache_clear()
    yield copia
    bairros.tabela_bairros.cache_clear()


def test_populacao_tem_os_75_bairros_e_fonte():
    tabela = bairros.tabela_bairros()
    assert tabela["populacao"].notna().sum() == 75
    assert tabela["populacao"].sum() > 1_700_000
    assert "IBGE" in bairros.fonte_populacao()


def test_upload_de_populacao_substitui_o_arquivo(populacao_temporaria):
    estado.iniciar()
    texto = populacao_temporaria.read_text(encoding="utf-8").replace("Batel;9913", "Batel;20000")
    resumo = estado.registrar_upload("pop_nova.csv", texto.encode("utf-8"), adicionar_ao_periodo=True)
    assert resumo["tipo_chave"] == "populacao"
    assert bairros.tabela_bairros().set_index("chave").loc["BATEL", "populacao"] == 20000
    assert "pop_nova.csv" in bairros.fonte_populacao()


def test_upload_de_populacao_incompleta_e_recusado(populacao_temporaria):
    estado.iniciar()
    with pytest.raises(ValueError, match="75 bairros"):
        estado.registrar_upload("pop.csv", b"bairro;populacao\nBatel;10.000\n", adicionar_ao_periodo=True)


def test_todo_indicador_tem_definicao_e_existe(bases):
    ind = calcular_indicadores(bases["156"], bases["equipamentos"], bases["esaude"])
    for indicador in INDICADORES:
        assert indicador.coluna in ind.columns
        assert indicador.dimensao in DIMENSAO_POR_CHAVE
        assert indicador.o_que_mede and indicador.como_calcula and indicador.unidade


def test_explicacao_usa_numeros_reais(bases):
    ind = calcular_indicadores(bases["156"], bases["equipamentos"], bases["esaude"])
    indicador = INDICADORES[1]  # trânsito
    comparacao = comparar_com_bairros(ind, "BATEL", indicador)
    assert comparacao["melhores"] + comparacao["piores"] <= 74
    assert "mediana dos bairros" in explicar(ind, "BATEL", indicador)
