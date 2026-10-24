"""Lista oficial dos 75 bairros (vinda do mapa do IPPUC) e padronização de nomes.

A base do 156 e o mapa escrevem o nome do bairro de jeitos diferentes ("ÁGUA VERDE",
"Agua Verde", "CIDADE INDUSTRIAL"...). Aqui transformamos tudo numa "chave" sem acentos
e em maiúsculas, para conseguir ligar os chamados ao mapa.
"""
import json
import unicodedata
from functools import lru_cache

import pandas as pd

from src.config import ARQUIVO_GEOJSON

# Nomes usados nas bases da prefeitura que diferem do mapa
APELIDOS = {
    "CIDADE INDUSTRIAL": "CIDADE INDUSTRIAL DE CURITIBA",
    "CIC": "CIDADE INDUSTRIAL DE CURITIBA",
}

_PALAVRAS_MINUSCULAS = {"de", "da", "do", "das", "dos", "e"}


def normalizar_texto(texto) -> str:
    """Remove acentos, espaços extras e coloca em maiúsculas: 'Água  Verde' -> 'AGUA VERDE'."""
    if texto is None or (isinstance(texto, float) and pd.isna(texto)):
        return ""
    sem_acento = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode()
    return " ".join(sem_acento.upper().split())


def chave_bairro(nome) -> str:
    """Converte qualquer grafia de bairro na chave usada pelo projeto."""
    chave = normalizar_texto(nome)
    return APELIDOS.get(chave, chave)


_SIGLAS = {"CIC", "BR", "PR", "II", "III", "IV", "VI", "VII", "VIII", "IX", "XI", "XV"}


def nome_proprio(texto: str) -> str:
    """'CIDADE INDUSTRIAL DE CURITIBA' -> 'Cidade Industrial de Curitiba'; 'DOM PEDRO II' -> 'Dom Pedro II'."""
    palavras = str(texto).split()
    return " ".join(
        p.upper() if p.upper() in _SIGLAS
        else p.lower() if (i > 0 and p.lower() in _PALAVRAS_MINUSCULAS)
        else p.capitalize()
        for i, p in enumerate(palavras)
    )


@lru_cache(maxsize=1)
def carregar_geojson() -> dict:
    """Lê o GeoJSON dos bairros e acrescenta a propriedade 'chave' em cada um."""
    with open(ARQUIVO_GEOJSON, encoding="utf-8") as arquivo:
        geo = json.load(arquivo)
    for bairro in geo["features"]:
        bairro["properties"]["chave"] = chave_bairro(bairro["properties"]["nome"])
    return geo


@lru_cache(maxsize=1)
def tabela_bairros() -> pd.DataFrame:
    """Tabela com chave, nome e regional de cada um dos 75 bairros."""
    linhas = []
    for bairro in carregar_geojson()["features"]:
        p = bairro["properties"]
        linhas.append({
            "chave": p["chave"],
            "bairro": nome_proprio(p["nome"]),
            "regional": nome_proprio(p["nome_regional"].replace("REGIONAL ", "")),
        })
    return pd.DataFrame(linhas).sort_values("bairro").reset_index(drop=True)

