"""Cálculos que respondem às hipóteses do trabalho.

Pergunta central: o tempo que a prefeitura leva para responder um chamado do 156
depende da REGIÃO onde a pessoa mora ou do TIPO DE PROBLEMA?

- H1: a região (regional/bairro) muda a chance de atraso.
- H2: o tipo de problema (órgão/assunto) pesa mais que a região.
- H3: o tipo de chamado (solicitação, reclamação, denúncia) muda a chance de atraso.
- H4: o canal usado (telefone, aplicativo, portal, chat) muda a chance de atraso.

Definição: um chamado está ATRASADO quando levou mais de 30 dias para ser
respondido ou continua sem resposta depois de 30 dias.
"""
import numpy as np
import pandas as pd

from src.bairros import tabela_bairros
from src.config import MINIMO_CHAMADOS_GRUPO, PRAZO_ATRASO_DIAS
from src.preparacao import apenas_demandas

DIAS_DA_SEMANA = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"]


def montar_base(chamados: pd.DataFrame, prazo: int = PRAZO_ATRASO_DIAS) -> pd.DataFrame:
    """Tabela usada na análise e nos modelos: uma linha por demanda, com 'atrasou' (1/0).

    Só entram chamados abertos há pelo menos `prazo` dias — os mais recentes ainda
    não tiveram tempo de atrasar e distorceriam a porcentagem.
    """
    demandas = apenas_demandas(chamados)
    data_ref = chamados["data"].max()
    base = demandas[demandas["data"] <= data_ref - pd.Timedelta(days=prazo)].copy()
    respondido = base["dias_resposta"].notna()
    base["atrasou"] = np.where(respondido, base["dias_resposta"] > prazo, base["pendente"]).astype(int)
    base = base[respondido | base["pendente"]]  # sem resposta e sem estar pendente = situação desconhecida

    resultado = pd.DataFrame({
        "data": base["data"], "mes": base["mes"], "dias_resposta": base["dias_resposta"],
        "Assunto": base["Assunto"].astype(str), "Subdivisao": base["Subdivisao"].astype(str),
        "Orgao": base["Orgao"].astype(str), "Tipo": base["Tipo"].astype(str), "Origem": base["Origem"].astype(str),
        "chave": base["chave"].astype(str),
        "Regional": base["Regional"].astype(str).str.replace("Unidade Regional ", "", regex=False),
        "dia_semana": base["data"].dt.dayofweek.map(dict(enumerate(DIAS_DA_SEMANA))),
        "mes_nome": base["data"].dt.month.astype(str),
        "atrasou": base["atrasou"],
    })
    return resultado.reset_index(drop=True)


def taxa_de_atraso(base: pd.DataFrame, coluna: str, minimo: int = MINIMO_CHAMADOS_GRUPO) -> pd.DataFrame:
    """% de chamados atrasados e tempo típico de resposta para cada valor de `coluna`."""
    grupos = base.groupby(coluna)
    tabela = pd.DataFrame({
        "chamados": grupos.size(),
        "atrasados": grupos["atrasou"].sum(),
        "pct_atraso": grupos["atrasou"].mean() * 100,
        "mediana_dias": grupos["dias_resposta"].median(),
    })
    return tabela[tabela["chamados"] >= minimo].sort_values("pct_atraso")


def comparacao_justa(base: pd.DataFrame, coluna: str = "Regional",
                     minimo: int = MINIMO_CHAMADOS_GRUPO) -> pd.DataFrame:
    """Compara grupos "maçã com maçã": retira o efeito do tipo de problema.

    Cada região tem uma mistura diferente de assuntos (o Centro tem muita abordagem
    social, que é resolvida rápido; outras regiões têm mais pedidos de poda, que demoram).
    Para cada região calculamos:
    - observado: % de atraso real;
    - esperado: % que ela teria se cada chamado atrasasse na média da cidade para o seu assunto;
    - ajustado: observado ÷ esperado × média da cidade. É o atraso que a região teria se tivesse
      a mesma mistura de assuntos que a cidade toda.
    Se o ajustado continua diferente entre regiões, a diferença NÃO é explicada pelo tipo de problema.
    """
    atraso_do_assunto = base.groupby("Assunto")["atrasou"].mean()
    esperado = base["Assunto"].map(atraso_do_assunto)
    grupos = base.assign(esperado=esperado).groupby(coluna)
    tabela = pd.DataFrame({
        "chamados": grupos.size(),
        "observado": grupos["atrasou"].mean() * 100,
        "esperado": grupos["esperado"].mean() * 100,
    })
    tabela["ajustado"] = tabela["observado"] / tabela["esperado"] * base["atrasou"].mean() * 100
    return tabela[tabela["chamados"] >= minimo].sort_values("ajustado")


def diferenca_entre_regionais_por_assunto(base: pd.DataFrame, quantidade: int = 10,
                                         minimo: int = 100) -> pd.DataFrame:
    """Para os assuntos mais pedidos: atraso na regional mais rápida e na mais lenta.

    Mostra ONDE o lugar importa: se para o mesmo assunto uma regional atrasa muito mais
    que outra, a diferença não pode ser explicada pelo tipo de problema.
    """
    linhas = []
    for assunto in base["Assunto"].value_counts().head(quantidade).index:
        por_regional = taxa_de_atraso(base[base["Assunto"] == assunto], "Regional", minimo)
        if len(por_regional) < 2:
            continue
        linhas.append({
            "Assunto": assunto, "Chamados": int(por_regional["chamados"].sum()),
            "Regional mais rápida": por_regional.index[0], "% atraso (mais rápida)": por_regional["pct_atraso"].iloc[0],
            "Regional mais lenta": por_regional.index[-1], "% atraso (mais lenta)": por_regional["pct_atraso"].iloc[-1],
        })
    tabela = pd.DataFrame(linhas)
    tabela["Diferença (pontos)"] = tabela["% atraso (mais lenta)"] - tabela["% atraso (mais rápida)"]
    return tabela.sort_values("Diferença (pontos)", ascending=False).reset_index(drop=True)


def por_bairro(base: pd.DataFrame, minimo: int = 50) -> pd.DataFrame:
    """% de atraso por bairro, com o nome do bairro (para o mapa). Índice = chave do bairro."""
    tabela = taxa_de_atraso(base, "chave", minimo)
    return tabela.join(tabela_bairros().set_index("chave")[["bairro", "regional"]])


def evolucao_mensal(base: pd.DataFrame) -> pd.DataFrame:
    """% de atraso em cada mês de abertura do chamado."""
    tabela = taxa_de_atraso(base, "mes", minimo=1)
    return tabela.sort_index()


def amplitude(tabela: pd.DataFrame, coluna: str = "pct_atraso") -> float:
    """Diferença entre o maior e o menor valor (em pontos percentuais)."""
    return float(tabela[coluna].max() - tabela[coluna].min()) if len(tabela) else 0.0
