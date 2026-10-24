"""Cálculos das hipóteses."""
from src import analise
from src.config import PRAZO_ATRASO_DIAS


def test_base_so_tem_demandas_com_idade_minima(chamados, base):
    assert base["Tipo"].isin(["Solicitação", "Reclamação", "Denúncia"]).all()
    assert base["data"].max() <= chamados["data"].max()
    assert set(base["atrasou"].unique()) <= {0, 1}
    respondidos = base.dropna(subset=["dias_resposta"])
    assert ((respondidos["dias_resposta"] > PRAZO_ATRASO_DIAS) == respondidos["atrasou"].astype(bool)).all()


def test_taxa_de_atraso_bate_com_a_conta_manual(base):
    tabela = analise.taxa_de_atraso(base, "Tipo")
    denuncias = base[base["Tipo"] == "Denúncia"]
    assert abs(tabela.loc["Denúncia", "pct_atraso"] - denuncias["atrasou"].mean() * 100) < 1e-9


def test_comparacao_justa_preserva_a_media(base):
    justa = analise.comparacao_justa(base, "Regional")
    media = base["atrasou"].mean() * 100
    # O ajuste reduz a diferença entre regionais, mas não muda a ordem de grandeza
    assert analise.amplitude(justa, "ajustado") < analise.amplitude(justa, "observado")
    assert abs(justa["ajustado"].mean() - media) < 5


def test_mapa_por_bairro_tem_nomes(base):
    bairros = analise.por_bairro(base)
    assert len(bairros) >= 70 and bairros["bairro"].notna().all()


def test_diferenca_entre_regionais_por_assunto(base):
    tabela = analise.diferenca_entre_regionais_por_assunto(base, quantidade=5)
    assert 1 <= len(tabela) <= 5
    assert (tabela["% atraso (mais lenta)"] >= tabela["% atraso (mais rápida)"]).all()
    assert tabela["Diferença (pontos)"].is_monotonic_decreasing
