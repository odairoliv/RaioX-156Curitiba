"""Leitura dos três formatos, validação das colunas e limpeza."""
import pandas as pd
import pytest

from src.bairros import chave_bairro, nome_proprio
from src.leitura import ErroDeArquivo, ler_156
from src.preparacao import juntar_156, preparar_156


def test_csv_json_e_xml_geram_a_mesma_base(pasta_exemplos):
    json_ = preparar_156(ler_156("a.json", (pasta_exemplos / "156_semana_01a07_out_2026.json").read_bytes()))[0]
    xml = preparar_156(ler_156("a.xml", (pasta_exemplos / "156_semana_01a07_out_2026.xml").read_bytes()))[0]
    csv = preparar_156(ler_156("a.csv", (pasta_exemplos / "156_outubro_2026.csv").read_bytes()))[0]
    semana_csv = csv[csv["data"] <= "2026-10-07"]
    assert len(json_) == len(xml) == len(semana_csv) > 10_000


def test_csv_em_latin1_com_ponto_e_virgula():
    conteudo = ("DataCriacao;DataResposta;Assunto;Bairro;Situacao;Orgao;Tipo;Origem\n"
                "03/01/2026;05/01/2026;Árvore;Água Verde;Concluído;SMMA;Solicitação;Telefone\n")
    limpo, relatorio = preparar_156(ler_156("x.csv", conteudo.encode("latin-1")))
    assert relatorio["linhas_validas"] == 1
    assert limpo.loc[0, "chave"] == "AGUA VERDE" and limpo.loc[0, "dias_resposta"] == 2


def test_arquivo_sem_as_colunas_do_156_da_erro_amigavel():
    with pytest.raises(ErroDeArquivo, match="Faltam as colunas"):
        ler_156("x.csv", b"nome;idade\nAna;30\n")


def test_formato_nao_suportado():
    with pytest.raises(ErroDeArquivo, match="não suportado"):
        ler_156("planilha.xlsx", b"qualquer coisa")


def test_nomes_de_bairro_sao_padronizados():
    assert chave_bairro("Água  Verde") == "AGUA VERDE"
    assert chave_bairro("CIDADE INDUSTRIAL") == "CIDADE INDUSTRIAL DE CURITIBA"
    assert nome_proprio("ALAMEDA DOM PEDRO II") == "Alameda Dom Pedro II"


def test_juntar_substitui_o_periodo_do_arquivo_novo(chamados):
    setembro = chamados[chamados["data"] >= "2026-09-01"]
    juntos = juntar_156(chamados, setembro)
    assert len(juntos) == len(chamados)  # enviar de novo o mesmo período não duplica
    assert juntos["data"].is_monotonic_increasing
    assert isinstance(juntos["chave"].dtype, pd.CategoricalDtype)
