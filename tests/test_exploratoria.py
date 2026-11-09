"""Análise exploratória e qualidade dos dados."""
import pandas as pd

from src.exploratoria import (analise_exploratoria, completude, estatisticas_dias, problemas_na_base,
                              taxa_aproveitamento)
from src.leitura import ler_156
from src.preparacao import preparar_156

CABECALHO = "DataCriacao;DataResposta;Assunto;Subdivisao;Bairro;Situacao;Orgao;Tipo;Origem;Regional;Logradouro"


def _base_pequena(linhas: list[str]) -> tuple[pd.DataFrame, dict]:
    conteudo = "\n".join([CABECALHO, *linhas]).encode("utf-8")
    return preparar_156(ler_156("x.csv", conteudo))


def test_limpeza_conta_cada_tipo_de_problema():
    _, relatorio = _base_pequena([
        "03/01/2026;05/01/2026;Árvore;Poda;Centro;Concluído;SMMA;Solicitação;Telefone;;Rua A",
        "sem-data;05/01/2026;Árvore;Poda;Centro;Concluído;SMMA;Solicitação;Telefone;;Rua A",
        "03/01/2026;05/01/2026;Árvore;Poda;Bairro Inventado;Concluído;SMMA;Solicitação;Telefone;;Rua A",
        "10/01/2026;02/01/2026;Árvore;Poda;Centro;Concluído;SMMA;Solicitação;Telefone;;Rua A",
        "10/01/2026;ontem;Árvore;Poda;Centro;Concluído;SMMA;Solicitação;Telefone;;Rua A",
    ])
    assert relatorio["linhas_lidas"] == 5
    assert relatorio["sem_data"] == 1
    assert relatorio["bairro_desconhecido"] == 1
    assert relatorio["resposta_antes_abertura"] == 1
    assert relatorio["data_resposta_invalida"] == 1
    assert relatorio["regional_deduzida"] == 3
    assert relatorio["linhas_validas"] == 3


def test_completude_e_problemas_em_base_pequena():
    chamados, _ = _base_pequena([
        "03/01/2026;05/01/2026;Árvore;Poda;Centro;Concluído;SMMA;Solicitação;Telefone;;Rua A",
        "03/01/2026;05/01/2026;Árvore;Poda;Centro;Concluído;SMMA;Solicitação;Telefone;;Rua A",
        "20/03/2026;;Árvore;;Batel;Aberto;SMMA;Reclamação;Portal;;",
        "25/03/2026;;Árvore;Poda;Batel;Concluído;SMMA;Denúncia;Chat;;Rua B",
    ])
    tabela = completude(chamados).set_index("Coluna")
    assert tabela.loc["Data da resposta", "Vazios"] == 2
    assert tabela.loc["Detalhe do assunto", "Vazios"] == 1
    assert tabela.loc["Bairro", "Preenchido (%)"] == 100
    problemas = problemas_na_base(chamados)
    assert problemas["possiveis_duplicadas"] == 1
    assert problemas["concluido_sem_data_resposta"] == 1
    assert problemas["pendentes_antigos"] == 0  # o "Aberto" de 20/03 tem só 5 dias na data mais recente (25/03)
    assert problemas["recentes"] == 2  # 20/03 e 25/03 estão dentro dos últimos 30 dias
    assert estatisticas_dias(chamados)["mediana"] == 2


def test_analise_completa_na_base_real(chamados):
    eda = analise_exploratoria(chamados)
    assert eda["geral"]["chamados"] == len(chamados)
    assert eda["geral"]["bairros"] == 75
    assert eda["por_mes"].sum() == len(chamados)
    assert len(eda["completude"]) == 11
    assert eda["dias"]["q1"] <= eda["dias"]["mediana"] <= eda["dias"]["q3"]


def test_aproveitamento_so_mostra_100_quando_nada_foi_descartado():
    assert taxa_aproveitamento(585_610, 585_696) == 99.98  # 99,985% não vira "100%"
    assert taxa_aproveitamento(999_999, 1_000_000) < 100
    assert taxa_aproveitamento(500, 500) == 100
    assert taxa_aproveitamento(0, 0) == 0
