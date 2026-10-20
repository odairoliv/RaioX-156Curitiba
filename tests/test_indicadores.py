"""Indicadores e notas dos bairros."""
import pandas as pd

from src.bairros import chave_bairro, nome_proprio
from src.indicadores import PRESETS_PESOS, calcular_indicadores, calcular_notas
from src.preparacao import juntar_156


def test_nomes_de_bairro_sao_padronizados():
    assert chave_bairro("Água  Verde") == "AGUA VERDE"
    assert chave_bairro("CIDADE INDUSTRIAL") == "CIDADE INDUSTRIAL DE CURITIBA"
    assert nome_proprio("ALAMEDA DOM PEDRO II") == "Alameda Dom Pedro II"


def test_notas_ficam_entre_0_e_10_para_os_75_bairros(bases):
    ind = calcular_indicadores(bases["156"], bases["equipamentos"], bases["esaude"])
    notas = calcular_notas(ind, PRESETS_PESOS["Equilibrado"])
    assert len(notas) == 75
    assert notas["nota_final"].between(0, 10).all()
    assert set(notas["situacao"]) <= {"Boa", "Atenção", "Crítica"}


def test_pesos_mudam_a_nota(bases):
    ind = calcular_indicadores(bases["156"], bases["equipamentos"], bases["esaude"])
    morar = calcular_notas(ind, PRESETS_PESOS["Vou morar no bairro"])["nota_final"]
    visitar = calcular_notas(ind, PRESETS_PESOS["Vou visitar / me hospedar"])["nota_final"]
    assert not morar.equals(visitar)


def test_juntar_substitui_o_periodo_do_arquivo_novo(bases):
    atual = bases["156"]
    setembro = atual[atual["data"] >= "2026-09-01"]
    juntos = juntar_156(atual, setembro)
    # Enviar de novo um período que já existe não duplica linhas
    assert len(juntos) == len(atual)
    assert juntos["data"].is_monotonic_increasing
    assert isinstance(juntos["chave"].dtype, pd.CategoricalDtype)
