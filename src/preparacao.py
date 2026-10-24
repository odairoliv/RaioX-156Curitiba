"""Limpeza e preparação da base da Central 156 (pandas).

Recebe o DataFrame "cru" vindo de src.leitura e devolve uma versão limpa, com
bairros padronizados, datas convertidas e colunas calculadas (dias até a
resposta, mês...). Também devolve um pequeno relatório de qualidade: quantas
linhas foram descartadas e por quê.
"""
import pandas as pd

from src.bairros import chave_bairro, nome_proprio, normalizar_texto, tabela_bairros

SITUACOES_PENDENTES = {"ABERTO", "EM ANDAMENTO", "EM AVALIACAO", "FALTA DE DADOS"}
# Só estes tipos pedem uma ação da prefeitura (elogios, sugestões e informações ficam de fora)
TIPOS_DEMANDA = {"SOLICITACAO", "RECLAMACAO", "DENUNCIA"}
COLUNAS_TEXTO = ["chave", "Tipo", "Orgao", "Assunto", "Subdivisao", "Situacao", "Logradouro", "Regional", "Origem"]


def _mapear(serie: pd.Series, funcao) -> pd.Series:
    """Aplica uma função só nos valores distintos (muito mais rápido em bases grandes)."""
    return serie.map({valor: funcao(valor) for valor in serie.unique()})


def _converter_datas(coluna: pd.Series) -> pd.Series:
    """Converte '03/01/2026' (padrão da prefeitura) e, se não der, tenta outros formatos."""
    datas = pd.to_datetime(coluna, format="%d/%m/%Y", errors="coerce")
    if datas.isna().mean() > 0.5:
        datas = pd.to_datetime(coluna, dayfirst=True, errors="coerce", format="mixed")
    return datas.dt.normalize()


def preparar_156(bruto: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Limpa a base de chamados. Devolve (base limpa, relatório de qualidade)."""
    relatorio = {"linhas_lidas": len(bruto)}
    df = bruto.copy()

    # 1. Datas
    df["data"] = _converter_datas(df["DataCriacao"])
    df["data_resposta"] = _converter_datas(df["DataResposta"])
    relatorio["sem_data"] = int(df["data"].isna().sum())
    df = df.dropna(subset=["data"])

    # 2. Bairro: só os 75 bairros oficiais (nomes padronizados para casar com o mapa)
    df["chave"] = _mapear(df["Bairro"], chave_bairro)
    validos = df["chave"].isin(set(tabela_bairros()["chave"]))
    relatorio["bairro_desconhecido"] = int((~validos).sum())
    df = df[validos]

    # 3. Colunas calculadas
    df["dias_resposta"] = (df["data_resposta"] - df["data"]).dt.days
    df.loc[df["dias_resposta"] < 0, "dias_resposta"] = pd.NA  # resposta antes da abertura = erro de cadastro
    df["pendente"] = _mapear(df["Situacao"], normalizar_texto).isin(SITUACOES_PENDENTES)
    df["mes"] = df["data"].dt.to_period("M").dt.start_time
    df["Logradouro"] = _mapear(df["Logradouro"], lambda s: nome_proprio(normalizar_texto(s)))

    df = df[["data", "data_resposta", "mes", "dias_resposta", "pendente", *COLUNAS_TEXTO]].reset_index(drop=True)
    # O tipo "category" economiza muita memória (são ~600 mil linhas)
    for coluna in COLUNAS_TEXTO:
        df[coluna] = df[coluna].astype("category")

    relatorio["linhas_validas"] = len(df)
    relatorio["periodo"] = (df["data"].min(), df["data"].max()) if len(df) else (None, None)
    return df, relatorio


def apenas_demandas(chamados: pd.DataFrame) -> pd.DataFrame:
    """Filtra solicitações, reclamações e denúncias."""
    tipos = chamados["Tipo"].map(normalizar_texto)  # em "category" roda só nos valores distintos
    return chamados[tipos.isin(TIPOS_DEMANDA).to_numpy()]


def juntar_156(atual: pd.DataFrame, novo: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta novos chamados à base atual.

    O período coberto pelo arquivo novo SUBSTITUI o mesmo período da base atual.
    Assim, enviar o mesmo mês duas vezes não duplica os chamados.
    """
    inicio, fim = novo["data"].min(), novo["data"].max()
    fora_do_periodo = atual[(atual["data"] < inicio) | (atual["data"] > fim)]
    juntos = pd.concat([fora_do_periodo, novo], ignore_index=True)
    for coluna in COLUNAS_TEXTO:
        juntos[coluna] = juntos[coluna].astype("category")
    return juntos.sort_values("data").reset_index(drop=True)
