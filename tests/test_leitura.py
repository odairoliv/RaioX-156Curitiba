"""Leitura dos três formatos e validação das colunas."""
import pytest

from src.leitura import ErroDeArquivo, ler_e_identificar
from src.preparacao import preparar_156


def test_csv_json_e_xml_geram_a_mesma_base(pasta_exemplos):
    csv = (pasta_exemplos / "156_outubro_2026.csv").read_bytes()
    json_ = (pasta_exemplos / "156_semana_01a07_out_2026.json").read_bytes()
    xml = (pasta_exemplos / "156_semana_01a07_out_2026.xml").read_bytes()

    tipo_csv, df_csv = ler_e_identificar("a.csv", csv)
    tipo_json, df_json = ler_e_identificar("a.json", json_)
    tipo_xml, df_xml = ler_e_identificar("a.xml", xml)

    assert tipo_csv == tipo_json == tipo_xml == "156"
    limpo_json, _ = preparar_156(df_json)
    limpo_xml, _ = preparar_156(df_xml)
    assert len(limpo_json) == len(limpo_xml) > 10_000
    semana_csv, _ = preparar_156(df_csv)
    semana_csv = semana_csv[semana_csv["data"] <= "2026-10-07"]
    assert len(semana_csv) == len(limpo_json)


def test_csv_em_latin1_com_ponto_e_virgula():
    conteudo = "NM_BAIRRO;DS_TEMA;DS_TP_EQUIPAMENTO;NM_EQUI\nÁgua Verde;SAÚDE;Unidade de Saúde;UMS X\n"
    tipo, df = ler_e_identificar("equip.csv", conteudo.encode("latin-1"))
    assert tipo == "equipamentos"
    assert df.loc[0, "NM_BAIRRO"] == "Água Verde"


def test_arquivo_com_colunas_desconhecidas_da_erro_amigavel():
    with pytest.raises(ErroDeArquivo, match="Não reconheci as colunas"):
        ler_e_identificar("x.csv", b"nome;idade\nAna;30\n")


def test_formato_nao_suportado():
    with pytest.raises(ErroDeArquivo, match="não suportado"):
        ler_e_identificar("planilha.xlsx", b"qualquer coisa")
