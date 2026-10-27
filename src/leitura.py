"""Leitura dos arquivos da Central 156 enviados pelo usuário (.csv, .json ou .xml, opcionalmente .gz).

Os arquivos do portal de dados abertos de Curitiba variam: alguns vêm em UTF-8,
outros em Latin-1, quase sempre separados por ";". Esta etapa descobre isso
sozinha, confere se as colunas necessárias existem e devolve um DataFrame.
"""
import gzip
import io
import json
import re
from pathlib import Path

import pandas as pd

from src.bairros import normalizar_texto

# Colunas sem as quais a análise não funciona
COLUNAS_OBRIGATORIAS = ["DataCriacao", "DataResposta", "Assunto", "Bairro", "Situacao", "Orgao", "Tipo", "Origem"]
# Colunas usadas se existirem. Se faltarem (ou vierem vazias) o arquivo ainda pode ser usado,
# mas o usuário é avisado do efeito e decide se continua.
COLUNAS_OPCIONAIS = ["Subdivisao", "Regional", "Logradouro"]
EFEITO_SEM_COLUNA = {
    "Subdivisao": "Sem o **detalhe do assunto** (ex.: \"lâmpada apagada\" dentro de \"Iluminação Pública\"), os "
                  "modelos perdem a informação que mais ajuda a prever o atraso: as previsões ficam menos precisas e "
                  "o campo *Detalhe* do simulador fica sem efeito para esses chamados.",
    "Regional": "A **regional** será preenchida automaticamente a partir do bairro, usando a divisão oficial da "
                "prefeitura. Nada se perde.",
    "Logradouro": "A **rua** não é usada nas análises nem nos modelos. Nada se perde.",
}

FORMATOS_ACEITOS = ["csv", "json", "xml", "gz"]


class ErroDeArquivo(Exception):
    """Erro com mensagem amigável para mostrar ao usuário."""


def _simplificar(nome: str) -> str:
    """'Data Criacao', 'data_criacao' e 'DataCriação' viram 'datacriacao' — para casar nomes de colunas."""
    return re.sub(r"[^a-z0-9]", "", normalizar_texto(nome).lower())


def _decodificar(conteudo: bytes) -> str:
    """Tenta UTF-8 (com ou sem BOM) e, se falhar, Latin-1."""
    try:
        return conteudo.decode("utf-8-sig")
    except UnicodeDecodeError:
        return conteudo.decode("latin-1")


def _ler_csv(texto: str) -> pd.DataFrame:
    primeira_linha = texto.split("\n", 1)[0]
    separador = ";" if primeira_linha.count(";") >= primeira_linha.count(",") else ","
    df = pd.read_csv(io.StringIO(texto), sep=separador, dtype=str, keep_default_na=False)
    # Remove colunas vazias criadas pelo ";" no fim de cada linha
    return df.loc[:, ~df.columns.str.startswith("Unnamed")]


def _ler_json(texto: str) -> pd.DataFrame:
    dados = json.loads(texto)
    if isinstance(dados, dict):
        # Aceita {"registros": [...]} ou qualquer outra chave que contenha a lista
        listas = [v for v in dados.values() if isinstance(v, list)]
        if not listas:
            raise ErroDeArquivo("O JSON precisa ser uma lista de registros (ou um objeto com uma lista).")
        dados = listas[0]
    return pd.DataFrame(dados).astype(str).replace({"None": "", "nan": ""})


def _ler_xml(texto: str) -> pd.DataFrame:
    return pd.read_xml(io.StringIO(texto), dtype=str).fillna("")


def ler_arquivo(nome_arquivo: str, conteudo: bytes) -> pd.DataFrame:
    """Lê os bytes de um arquivo e devolve um DataFrame (todas as colunas como texto)."""
    extensao = Path(nome_arquivo).suffix.lower().lstrip(".")
    if extensao == "gz":
        conteudo = gzip.decompress(conteudo)
        extensao = Path(Path(nome_arquivo).stem).suffix.lower().lstrip(".")

    leitores = {"csv": _ler_csv, "json": _ler_json, "xml": _ler_xml}
    if extensao not in leitores:
        raise ErroDeArquivo(f"Formato '.{extensao}' não suportado. Envie .csv, .json ou .xml.")
    try:
        df = leitores[extensao](_decodificar(conteudo))
    except ErroDeArquivo:
        raise
    except Exception as erro:  # erros de leitura do pandas/lxml
        raise ErroDeArquivo(f"Não foi possível ler o arquivo como {extensao.upper()}: {erro}") from erro

    if df.empty:
        raise ErroDeArquivo("O arquivo não tem nenhuma linha de dados.")
    return df


def ler_156(nome_arquivo: str, conteudo: bytes) -> pd.DataFrame:
    """Lê um arquivo da Central 156, confere as colunas e padroniza os nomes delas."""
    df = ler_arquivo(nome_arquivo, conteudo)
    # Recupera o nome original mesmo se veio com outra grafia (ex.: 'data_criacao' no XML)
    originais = {_simplificar(c): c for c in COLUNAS_OBRIGATORIAS + COLUNAS_OPCIONAIS}
    df = df.rename(columns={c: originais.get(_simplificar(c), c) for c in df.columns})

    faltando = [c for c in COLUNAS_OBRIGATORIAS if c not in df.columns]
    if faltando:
        raise ErroDeArquivo(
            f"O arquivo não parece ser da Central 156. Faltam as colunas: {', '.join(faltando)}. "
            f"Colunas encontradas: {', '.join(map(str, df.columns[:12]))}."
        )
    # Opcional "faltando" = não existe ou está totalmente vazia
    faltando = [c for c in COLUNAS_OPCIONAIS
                if c not in df.columns or df[c].astype(str).str.strip().eq("").all()]
    for coluna in COLUNAS_OPCIONAIS:
        if coluna not in df.columns:
            df[coluna] = ""
    df.attrs["opcionais_faltando"] = faltando
    return df
