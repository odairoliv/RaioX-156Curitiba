"""Limpeza e preparação de cada base (pandas).

Recebe o DataFrame "cru" vindo de src.leitura e devolve uma versão limpa, com
bairros padronizados, datas convertidas e colunas calculadas. Também devolve um
pequeno relatório de qualidade (quantas linhas foram descartadas e por quê).
"""
import pandas as pd

from src.bairros import chave_bairro, nome_proprio, normalizar_texto, tabela_bairros
from src.categorias import categoria_do_assunto

SITUACOES_PENDENTES = {"ABERTO", "EM ANDAMENTO", "EM AVALIACAO", "FALTA DE DADOS"}

# Temas dos equipamentos urbanos que contam como "infraestrutura disponível"
TEMAS_INFRAESTRUTURA = {
    "EDUCACAO": "Educação",
    "SAUDE": "Saúde",
    "SEGURANCA": "Segurança",
    "TRANSPORTE": "Transporte",
    "ESPORTE": "Esporte e lazer",
    "CULTURA": "Cultura",
    "ASSISTENCIA SOCIAL": "Assistência social",
    "MEIO AMBIENTE": "Meio ambiente",
    "ABASTECIMENTO ALIMENTAR": "Abastecimento alimentar",
}

# Para cada item de moradia: coluna do E-Saúde e respostas consideradas adequadas
ITENS_SANEAMENTO = {
    "Água da rede pública": ("Abastecimento", {"REDE PUBLICA"}),
    "Lixo coletado": ("Destino Lixo", {"COLETADO"}),
    "Esgoto adequado": ("Fezes/Urina", {"SISTEMA DE ESGOTO", "FOSSA SEPTICA"}),
    "Moradia adequada": ("Tipo de Habitação", {
        "TIJOLO/ALVENARIA COM REVESTIMENTO", "TIJOLO ALVENARIA SEM REVESTIMENTO",
        "MADEIRA APARELHADA",
    }),
}


def _mapear(serie: pd.Series, funcao) -> pd.Series:
    """Aplica uma função só nos valores distintos (muito mais rápido em bases grandes)."""
    return serie.map({valor: funcao(valor) for valor in serie.unique()})


def _converter_datas(coluna: pd.Series) -> pd.Series:
    """Converte '03/01/2026' (padrão da prefeitura) e, se não der, tenta outros formatos."""
    datas = pd.to_datetime(coluna, format="%d/%m/%Y", errors="coerce")
    if datas.isna().mean() > 0.5:
        datas = pd.to_datetime(coluna, dayfirst=True, errors="coerce", format="mixed")
    return datas.dt.normalize()


def _filtrar_bairros_validos(df: pd.DataFrame, coluna_bairro: str, relatorio: dict) -> pd.DataFrame:
    """Cria a coluna 'chave' e descarta linhas com bairro fora dos 75 oficiais."""
    df = df.copy()
    df["chave"] = _mapear(df[coluna_bairro], chave_bairro)
    validos = df["chave"].isin(set(tabela_bairros()["chave"]))
    relatorio["bairro_desconhecido"] = int((~validos).sum())
    return df[validos]


def preparar_156(bruto: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Prepara a base de chamados da Central 156."""
    relatorio = {"linhas_lidas": len(bruto)}
    df = bruto.copy()
    for coluna in ["Tipo", "Orgao", "Subdivisao", "Logradouro", "Regional", "DataResposta", "Origem"]:
        if coluna not in df.columns:
            df[coluna] = ""

    df["data"] = _converter_datas(df["DataCriacao"])
    df["data_resposta"] = _converter_datas(df["DataResposta"])
    relatorio["sem_data"] = int(df["data"].isna().sum())
    df = df.dropna(subset=["data"])
    df = _filtrar_bairros_validos(df, "Bairro", relatorio)

    df["categoria"] = _mapear(df["Assunto"], categoria_do_assunto)
    df["dias_resposta"] = (df["data_resposta"] - df["data"]).dt.days
    df.loc[df["dias_resposta"] < 0, "dias_resposta"] = pd.NA
    df["pendente"] = _mapear(df["Situacao"], normalizar_texto).isin(SITUACOES_PENDENTES)
    df["semana"] = df["data"].dt.to_period("W-SUN").dt.start_time
    df["mes"] = df["data"].dt.to_period("M").dt.start_time
    df["Logradouro"] = _mapear(df["Logradouro"], lambda s: nome_proprio(normalizar_texto(s)))

    colunas = ["data", "data_resposta", "semana", "mes", "chave", "categoria", "Tipo", "Orgao",
               "Assunto", "Subdivisao", "Situacao", "Logradouro", "Regional", "Origem",
               "dias_resposta", "pendente"]
    df = df[colunas].reset_index(drop=True)
    # "category" economiza muita memória (620 mil linhas)
    for coluna in ["chave", "categoria", "Tipo", "Orgao", "Assunto", "Subdivisao", "Situacao",
                   "Logradouro", "Regional", "Origem"]:
        df[coluna] = df[coluna].astype("category")

    relatorio["linhas_validas"] = len(df)
    relatorio["periodo"] = (df["data"].min(), df["data"].max()) if len(df) else (None, None)
    return df, relatorio


def preparar_equipamentos(bruto: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Prepara a base de equipamentos urbanos (mantém só os temas de infraestrutura)."""
    relatorio = {"linhas_lidas": len(bruto)}
    df = _filtrar_bairros_validos(bruto, "NM_BAIRRO", relatorio)
    df["tema"] = df["DS_TEMA"].map(normalizar_texto).map(TEMAS_INFRAESTRUTURA)
    relatorio["fora_dos_temas"] = int(df["tema"].isna().sum())
    df = df.dropna(subset=["tema"])
    df = df.rename(columns={"NM_EQUI": "nome", "DS_TP_EQUIPAMENTO": "tipo_equipamento"})
    if "nome" not in df.columns:
        df["nome"] = ""
    df = df[["chave", "tema", "tipo_equipamento", "nome"]].reset_index(drop=True)
    relatorio["linhas_validas"] = len(df)
    return df, relatorio


def preparar_esaude(bruto: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Prepara a base do E-Saúde: uma linha por morador, com 1/0 para cada item de saneamento."""
    relatorio = {"linhas_lidas": len(bruto)}
    df = bruto.copy()
    if "Municício" in df.columns:
        df = df[df["Municício"].map(normalizar_texto).isin({"CURITIBA", ""})]
    # Um mesmo morador pode ter vários atendimentos: contamos cada pessoa uma vez só
    if "cod_usuario" in df.columns:
        df = df.drop_duplicates(subset="cod_usuario", keep="last")
    relatorio["moradores_unicos"] = len(df)
    df = _filtrar_bairros_validos(df, "Bairro", relatorio)

    for item, (coluna, adequados) in ITENS_SANEAMENTO.items():
        if coluna not in df.columns:
            df[item] = pd.NA
            continue
        resposta = df[coluna].map(normalizar_texto)
        # Resposta vazia = não informado (fica de fora da porcentagem)
        df[item] = resposta.isin(adequados).astype("Float64").where(resposta != "")
    df = df[["chave", *ITENS_SANEAMENTO]].reset_index(drop=True)
    relatorio["linhas_validas"] = len(df)
    return df, relatorio


PREPARADORES = {
    "156": preparar_156,
    "equipamentos": preparar_equipamentos,
    "esaude": preparar_esaude,
}


def juntar_156(atual: pd.DataFrame, novo: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta novos chamados à base atual.

    O período coberto pelo arquivo novo SUBSTITUI o mesmo período da base atual.
    Assim, enviar o mesmo mês duas vezes não duplica os chamados.
    """
    inicio, fim = novo["data"].min(), novo["data"].max()
    fora_do_periodo = atual[(atual["data"] < inicio) | (atual["data"] > fim)]
    juntos = pd.concat([fora_do_periodo, novo], ignore_index=True)
    for coluna in novo.select_dtypes("category").columns:
        juntos[coluna] = juntos[coluna].astype("category")
    return juntos.sort_values("data").reset_index(drop=True)
