"""Estado da sessão do Streamlit: a base de chamados carregada e os modelos treinados.

Toda página usa as funções daqui para pegar os dados. Quando o usuário envia um
arquivo novo, todas as páginas passam a usar a base nova automaticamente.
"""
import json
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from src.analise import montar_base
from src.config import BASE_PADRAO, PASTA_CACHE
from src.leitura import ler_156
from src.modelos.registro import ResultadoTreino
from src.preparacao import juntar_156, preparar_156

ss = st.session_state


@st.cache_resource(show_spinner="Carregando a base da Central 156...")
def _carregar_base_padrao() -> tuple[pd.DataFrame, dict]:
    """Lê a base que acompanha o projeto (fica em cache: só é lida uma vez).

    A versão já limpa é guardada em data/cache/ (Parquet). Nas próximas aberturas do
    app ela é lida em menos de 1 segundo, em vez de limpar tudo de novo.
    """
    cache = PASTA_CACHE / "156.parquet"
    cache_relatorio = cache.with_suffix(".json")
    # O cache vale enquanto for mais novo que o arquivo de dados E que o código de limpeza
    pasta_src = Path(__file__).parent
    dependencias = [BASE_PADRAO] + [pasta_src / m for m in ("preparacao.py", "bairros.py", "leitura.py")]
    if cache.exists() and cache_relatorio.exists() and cache.stat().st_mtime > max(p.stat().st_mtime for p in dependencias):
        relatorio = json.loads(cache_relatorio.read_text(encoding="utf-8"))
        relatorio["periodo"] = tuple(pd.Timestamp(d) for d in relatorio["periodo"])
        return pd.read_parquet(cache), relatorio

    limpo, relatorio = preparar_156(ler_156(BASE_PADRAO.name, BASE_PADRAO.read_bytes()))
    PASTA_CACHE.mkdir(exist_ok=True)
    limpo.to_parquet(cache, index=False)
    cache_relatorio.write_text(json.dumps(relatorio, default=str), encoding="utf-8")
    return limpo, relatorio


def iniciar() -> None:
    """Prepara o estado na primeira vez que a sessão abre."""
    if "chamados" in ss:
        return
    ss.chamados, ss.relatorio = _carregar_base_padrao()
    ss.origem = f"Base padrão do projeto ({BASE_PADRAO.name})"
    ss.versao_dados = _nova_versao()
    ss.modelos = {}             # "classificacao" / "regressao" -> ResultadoTreino
    ss.historico_uploads = []


def _nova_versao() -> str:
    """Identificador único dos dados atuais (muda a cada upload)."""
    return uuid.uuid4().hex[:8]


def chamados() -> pd.DataFrame:
    return ss.chamados


def periodo_chamados() -> tuple[pd.Timestamp, pd.Timestamp]:
    return ss.chamados["data"].min(), ss.chamados["data"].max()


def registrar_upload(nome_arquivo: str, conteudo: bytes, adicionar_ao_periodo: bool) -> dict:
    """Lê, valida e prepara um arquivo enviado. Devolve um resumo para mostrar na tela."""
    limpo, relatorio = preparar_156(ler_156(nome_arquivo, conteudo))
    if limpo.empty:
        raise ValueError("Nenhuma linha válida depois da limpeza (verifique as datas e os nomes de bairro).")
    if adicionar_ao_periodo:
        ss.chamados = juntar_156(ss.chamados, limpo)
        ss.origem += f" + {nome_arquivo}"
    else:
        ss.chamados = limpo
        ss.origem = f"Arquivo enviado: {nome_arquivo}"
    ss.relatorio = relatorio
    ss.versao_dados = _nova_versao()
    resumo = {"arquivo": nome_arquivo, "enviado_em": datetime.now().strftime("%d/%m/%Y %H:%M"), **relatorio}
    ss.historico_uploads.append(resumo)
    return resumo


def restaurar_padrao() -> None:
    modelos, historico = ss.get("modelos", {}), ss.get("historico_uploads", [])
    ss.pop("chamados", None)
    iniciar()
    ss.modelos, ss.historico_uploads = modelos, historico


@st.cache_data(show_spinner="Preparando a base de análise...", max_entries=3)
def _base(versao: str, _chamados) -> pd.DataFrame:
    # "_chamados" não entra na chave do cache: a chave é a versão dos dados
    return montar_base(_chamados)


def base() -> pd.DataFrame:
    """Base de análise (demandas com 30+ dias, com a coluna 'atrasou'). Recalculada a cada upload."""
    return _base(ss.versao_dados, ss.chamados)


# ---------- Modelos ----------

def guardar_modelo(resultado: ResultadoTreino) -> None:
    ss.modelos[resultado.tipo] = resultado


def modelo(tipo: str) -> ResultadoTreino | None:
    return ss.modelos.get(tipo)


def modelo_desatualizado(tipo: str) -> bool:
    """True se o modelo foi treinado com dados diferentes dos atuais."""
    atual = modelo(tipo)
    return atual is not None and atual.versao_dados != ss.versao_dados
