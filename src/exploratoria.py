"""Análise exploratória e qualidade dos dados da base de chamados (pandas).

Responde, antes de qualquer modelo: o que tem na base, o quanto ela está completa,
como os valores se distribuem e que problemas a limpeza encontrou. É recalculada
toda vez que um arquivo novo é enviado.
"""
import math

import pandas as pd

from src.bairros import tabela_bairros
from src.config import PRAZO_ATRASO_DIAS

PRAZO_MUITO_LONGO = 180  # dias: respostas acima disso são destacadas como casos extremos

# Coluna da base limpa -> (nome amigável, o que significa quando está vazia)
COLUNAS_DESCRITAS = {
    "data": ("Data de abertura", "Não acontece: chamados sem data são descartados na limpeza."),
    "data_resposta": ("Data da resposta", "Chamado ainda sem resposta (aberto ou em andamento)."),
    "Tipo": ("Tipo", "Tipo não informado."),
    "Orgao": ("Órgão", "Órgão não informado."),
    "Assunto": ("Assunto", "Assunto não informado."),
    "Subdivisao": ("Detalhe do assunto", "Detalhe não informado: o modelo perde essa informação."),
    "Situacao": ("Situação", "Situação não informada."),
    "Origem": ("Canal", "Canal não informado."),
    "chave": ("Bairro", "Não acontece: chamados fora dos 75 bairros são descartados na limpeza."),
    "Regional": ("Regional", "Não acontece: quando falta, é preenchida pelo bairro."),
    "Logradouro": ("Rua", "Rua não informada (não é usada nas análises)."),
}


def taxa_aproveitamento(validas, lidas) -> float:
    """% de linhas aproveitadas, arredondada PARA BAIXO com 2 casas decimais.

    Arredondar para baixo garante que só aparece 100% quando nada foi descartado
    (ex.: 585.610 de 585.696 = 99,985% -> 99,98%, e não "100,0%").
    """
    if not lidas:
        return 0.0
    return math.floor(validas / lidas * 100 * 100) / 100


def _vazios(coluna: pd.Series) -> pd.Series:
    """True onde o valor está vazio (data ausente ou texto em branco)."""
    if pd.api.types.is_datetime64_any_dtype(coluna):
        return coluna.isna()
    return coluna.isna() | coluna.astype(str).str.strip().eq("")


def completude(chamados: pd.DataFrame) -> pd.DataFrame:
    """Para cada coluna: % preenchido, vazios, valores diferentes e o valor mais comum."""
    linhas = []
    for coluna, (nome, significado) in COLUNAS_DESCRITAS.items():
        valores = chamados[coluna]
        vazios = _vazios(valores)
        preenchidos = valores[~vazios]
        mais_comum = preenchidos.astype(str).mode()
        if coluna == "chave":  # mostra o nome do bairro, não a chave interna
            mais_comum = mais_comum.map(tabela_bairros().set_index("chave")["bairro"])
        linhas.append({
            "Coluna": nome,
            "Preenchido (%)": (1 - vazios.mean()) * 100 if len(valores) else 0.0,
            "Vazios": int(vazios.sum()),
            "Valores diferentes": int(preenchidos.nunique()),
            "Valor mais comum": "—" if pd.api.types.is_datetime64_any_dtype(valores) or mais_comum.empty
                                else str(mais_comum.iloc[0]),
            "Quando está vazio significa": significado,
        })
    return pd.DataFrame(linhas)


def estatisticas_dias(chamados: pd.DataFrame) -> dict:
    """Estatísticas descritivas dos dias até a resposta (só chamados respondidos)."""
    dias = chamados["dias_resposta"].dropna().astype(float)
    if dias.empty:
        return {"quantidade": 0}
    q1, q3 = dias.quantile(0.25), dias.quantile(0.75)
    limite_extremo = q3 + 1.5 * (q3 - q1)  # regra clássica do boxplot para valores atípicos
    return {
        "quantidade": int(len(dias)), "media": dias.mean(), "desvio": dias.std(), "minimo": dias.min(),
        "q1": q1, "mediana": dias.median(), "q3": q3, "maximo": dias.max(),
        "limite_extremo": limite_extremo, "extremos": int((dias > limite_extremo).sum()),
        "muito_longos": int((dias > PRAZO_MUITO_LONGO).sum()),
    }


def problemas_na_base(chamados: pd.DataFrame) -> dict:
    """Situações que merecem atenção e que dá para medir olhando a base já limpa."""
    data_ref = chamados["data"].max()
    corte = data_ref - pd.Timedelta(days=PRAZO_ATRASO_DIAS)
    return {
        # Sem número de protocolo não dá para ter certeza de que são repetidos: só contamos
        "possiveis_duplicadas": int(chamados.duplicated().sum()),
        "concluido_sem_data_resposta": int((~chamados["pendente"] & chamados["data_resposta"].isna()).sum()),
        "pendentes_antigos": int((chamados["pendente"] & (chamados["data"] <= corte)).sum()),
        "recentes": int((chamados["data"] > corte).sum()),
        "prazos_muito_longos": int((chamados["dias_resposta"] > PRAZO_MUITO_LONGO).sum()),
    }


def analise_exploratoria(chamados: pd.DataFrame) -> dict:
    """Junta tudo o que a tela de análise exploratória mostra."""
    nomes_bairros = tabela_bairros().set_index("chave")["bairro"]
    inicio, fim = chamados["data"].min(), chamados["data"].max()
    return {
        "geral": {
            "chamados": len(chamados), "inicio": inicio, "fim": fim,
            "dias_cobertos": (fim - inicio).days + 1 if len(chamados) else 0,
            "bairros": int(chamados["chave"].nunique()), "assuntos": int(chamados["Assunto"].nunique()),
            "orgaos": int(chamados["Orgao"].nunique()),
            "respondidos_pct": chamados["data_resposta"].notna().mean() * 100 if len(chamados) else 0.0,
        },
        "completude": completude(chamados),
        "dias": estatisticas_dias(chamados),
        "dias_valores": chamados["dias_resposta"].dropna().astype(float),
        "por_mes": chamados.groupby("mes").size(),
        "por_tipo": chamados["Tipo"].value_counts(),
        "por_situacao": chamados["Situacao"].value_counts(),
        "por_canal": chamados["Origem"].value_counts(),
        "top_assuntos": chamados["Assunto"].value_counts().head(10),
        "top_bairros": chamados["chave"].astype(str).map(nomes_bairros).value_counts().head(10),
        "problemas": problemas_na_base(chamados),
    }
