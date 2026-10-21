"""Leitura dos arquivos enviados pelo usuário (.csv, .json ou .xml, opcionalmente .gz).

Os arquivos do portal de dados abertos de Curitiba variam: alguns vêm em UTF-8,
outros em Latin-1, quase sempre separados por ";". Esta etapa descobre isso
sozinha e entrega um DataFrame com os nomes de coluna padronizados.
"""
import gzip
import io
import json
import re
from pathlib import Path

import pandas as pd

from src.bairros import normalizar_texto

# Colunas mínimas que cada tipo de base precisa ter
COLUNAS_OBRIGATORIAS = {
    "156": ["DataCriacao", "Assunto", "Bairro", "Situacao"],
    "equipamentos": ["NM_BAIRRO", "DS_TEMA", "DS_TP_EQUIPAMENTO"],
    "esaude": ["Bairro", "Abastecimento", "Destino Lixo", "Fezes/Urina"],
    "populacao": ["bairro", "populacao"],
}

# Colunas opcionais conhecidas (usadas para recuperar o nome original, ex.: no XML)
COLUNAS_CONHECIDAS = {
    "156": ["Tipo", "Orgao", "DataCriacao", "Assunto", "Subdivisao", "Situacao",
            "Logradouro", "Bairro", "Regional", "DataResposta", "Origem"],
    "equipamentos": ["NM_EQUI", "DS_TEMA", "DS_TP_EQUIPAMENTO", "DS_SUBTIPO_EQUIPAMENTO",
                     "NM_BAIRRO", "NM_REGIONAL", "NM_RUA", "NUMERO_EQUI"],
    "esaude": ["Data do Atendimento", "Bairro", "Municício", "Abastecimento", "Destino Lixo",
               "Fezes/Urina", "Tipo de Habitação", "Energia Elétrica", "cod_usuario"],
    "populacao": ["bairro", "populacao"],
}

NOMES_TIPOS = {
    "156": "Central 156 — chamados dos cidadãos",
    "equipamentos": "Equipamentos urbanos (escolas, saúde, segurança...)",
    "esaude": "E-Saúde — condições de moradia",
    "populacao": "População por bairro",
}

FORMATOS_ACEITOS = ["csv", "json", "xml", "gz"]


class ErroDeArquivo(Exception):
    """Erro com mensagem amigável para mostrar ao usuário."""


def _simplificar(nome: str) -> str:
    """'Fezes/Urina' e 'Fezes_Urina' viram 'fezesurina' — usado para casar nomes de colunas."""
    return re.sub(r"[^a-z0-9]", "", normalizar_texto(nome).lower())


def _decodificar(conteudo: bytes) -> str:
    """Tenta UTF-8 (com ou sem BOM) e, se falhar, Latin-1."""
    try:
        return conteudo.decode("utf-8-sig")
    except UnicodeDecodeError:
        return conteudo.decode("latin-1")


def _ler_csv(texto: str) -> pd.DataFrame:
    # Linhas começando com "#" são comentários (ex.: "# Fonte: IBGE..." no arquivo de população)
    if texto.startswith("#") or "\n#" in texto:
        texto = "\n".join(linha for linha in texto.splitlines() if not linha.startswith("#"))
    primeira_linha = texto.split("\n", 1)[0]
    separador = ";" if primeira_linha.count(";") >= primeira_linha.count(",") else ","
    df = pd.read_csv(io.StringIO(texto), sep=separador, dtype=str, keep_default_na=False)
    # Alguns arquivos têm uma linha "-------;-----" logo depois do cabeçalho
    if len(df) and df.iloc[0].astype(str).str.fullmatch(r"-*").all():
        df = df.iloc[1:]
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
    """Lê os bytes de um arquivo enviado e devolve um DataFrame (todas as colunas como texto)."""
    extensao = Path(nome_arquivo).suffix.lower().lstrip(".")
    if extensao == "gz":
        conteudo = gzip.decompress(conteudo)
        extensao = Path(Path(nome_arquivo).stem).suffix.lower().lstrip(".")

    texto = _decodificar(conteudo)
    leitores = {"csv": _ler_csv, "json": _ler_json, "xml": _ler_xml}
    if extensao not in leitores:
        raise ErroDeArquivo(f"Formato '.{extensao}' não suportado. Envie .csv, .json ou .xml.")
    try:
        df = leitores[extensao](texto)
    except ErroDeArquivo:
        raise
    except Exception as erro:  # erros de parsing do pandas/lxml
        raise ErroDeArquivo(f"Não foi possível ler o arquivo como {extensao.upper()}: {erro}") from erro

    df.columns = [str(c).strip().lstrip("﻿") for c in df.columns]
    if df.empty:
        raise ErroDeArquivo("O arquivo não tem nenhuma linha de dados.")
    return df


def identificar_tipo(df: pd.DataFrame) -> str | None:
    """Descobre de qual base é o arquivo olhando as colunas. Devolve None se não reconhecer."""
    colunas = {_simplificar(c) for c in df.columns}
    for tipo, obrigatorias in COLUNAS_OBRIGATORIAS.items():
        if all(_simplificar(c) in colunas for c in obrigatorias):
            return tipo
    return None


def padronizar_colunas(df: pd.DataFrame, tipo: str) -> pd.DataFrame:
    """Renomeia colunas para o nome original (ex.: 'Fezes_Urina' do XML -> 'Fezes/Urina')."""
    originais = {_simplificar(c): c for c in COLUNAS_CONHECIDAS[tipo] + COLUNAS_OBRIGATORIAS[tipo]}
    return df.rename(columns={c: originais.get(_simplificar(c), c) for c in df.columns})


def ler_e_identificar(nome_arquivo: str, conteudo: bytes) -> tuple[str, pd.DataFrame]:
    """Atalho: lê o arquivo, descobre o tipo e padroniza as colunas."""
    df = ler_arquivo(nome_arquivo, conteudo)
    tipo = identificar_tipo(df)
    if tipo is None:
        esperado = "; ".join(f"{NOMES_TIPOS[t]}: {', '.join(c)}" for t, c in COLUNAS_OBRIGATORIAS.items())
        raise ErroDeArquivo(
            "Não reconheci as colunas deste arquivo. Colunas encontradas: "
            f"{', '.join(df.columns[:12])}. Colunas esperadas — {esperado}."
        )
    return tipo, padronizar_colunas(df, tipo)
