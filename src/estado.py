"""Estado da sessão do Streamlit: quais bases estão carregadas e quais modelos foram treinados.

Toda página chama `iniciar()` (feito em app.py) e depois usa as funções daqui para
pegar os dados. Assim, quando o usuário envia um arquivo novo, todas as páginas
passam a usar a base nova automaticamente.
"""
import json
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from src.bairros import salvar_populacao, tabela_bairros
from src.config import BASES_PADRAO, PASTA_CACHE
from src.indicadores import calcular_indicadores, calcular_notas
from src.leitura import NOMES_TIPOS, ler_e_identificar
from src.modelos.classificacao import montar_base_atraso
from src.modelos.previsao import serie_semanal
from src.modelos.registro import ResultadoTreino
from src.preparacao import PREPARADORES, juntar_156

ss = st.session_state


@st.cache_resource(show_spinner="Carregando as bases de dados padrão...")
def _carregar_base_padrao(tipo: str) -> tuple[pd.DataFrame, dict]:
    """Lê uma base que acompanha o projeto (fica em cache: só é lida uma vez).

    A versão já limpa é guardada em data/cache/ (formato Parquet). Nas próximas
    aberturas do app ela é lida em menos de 1 segundo, em vez de limpar tudo de novo.
    """
    caminho = Path(BASES_PADRAO[tipo])
    cache = PASTA_CACHE / f"{tipo}.parquet"
    cache_relatorio = cache.with_suffix(".json")
    # O cache vale enquanto for mais novo que o arquivo de dados E que o código de limpeza
    pasta_src = Path(__file__).parent
    dependencias = [caminho] + [pasta_src / m for m in ("preparacao.py", "categorias.py", "bairros.py", "leitura.py")]
    mais_recente = max(p.stat().st_mtime for p in dependencias)
    if cache.exists() and cache_relatorio.exists() and cache.stat().st_mtime > mais_recente:
        relatorio = json.loads(cache_relatorio.read_text(encoding="utf-8"))
        if "periodo" in relatorio:
            relatorio["periodo"] = tuple(pd.Timestamp(d) for d in relatorio["periodo"])
        return pd.read_parquet(cache), relatorio

    _, bruto = ler_e_identificar(caminho.name, caminho.read_bytes())
    limpo, relatorio = PREPARADORES[tipo](bruto)
    PASTA_CACHE.mkdir(exist_ok=True)
    limpo.to_parquet(cache, index=False)
    cache_relatorio.write_text(json.dumps(relatorio, default=str), encoding="utf-8")
    return limpo, relatorio


def iniciar() -> None:
    """Prepara o estado na primeira vez que a sessão abre."""
    if "bases" in ss:
        return
    ss.bases, ss.relatorios, ss.origens = {}, {}, {}
    for tipo in BASES_PADRAO:
        ss.bases[tipo], ss.relatorios[tipo] = _carregar_base_padrao(tipo)
        ss.origens[tipo] = f"Base padrão do projeto ({Path(BASES_PADRAO[tipo]).name})"
    ss.versao_dados = _nova_versao()
    ss.modelos = {}             # tipo -> ResultadoTreino
    ss.historico_uploads = []   # lista de dicionários mostrada na página de upload


def _nova_versao() -> str:
    """Identificador único dos dados atuais (muda a cada upload)."""
    return uuid.uuid4().hex[:8]


def chamados() -> pd.DataFrame:
    return ss.bases["156"]


def equipamentos() -> pd.DataFrame | None:
    return ss.bases.get("equipamentos")


def esaude() -> pd.DataFrame | None:
    return ss.bases.get("esaude")


def registrar_upload(nome_arquivo: str, conteudo: bytes, adicionar_ao_periodo: bool) -> dict:
    """Lê, valida e prepara um arquivo enviado. Devolve um resumo para mostrar na tela."""
    tipo, bruto = ler_e_identificar(nome_arquivo, conteudo)
    limpo, relatorio = PREPARADORES[tipo](bruto)
    if limpo.empty:
        raise ValueError("Nenhuma linha válida depois da limpeza (verifique datas e nomes de bairro).")

    if tipo == "populacao":
        # A população vale para o app inteiro: substitui o arquivo data/populacao_bairros.csv
        faltando = set(tabela_bairros()["chave"]) - set(limpo["chave"])
        if faltando:
            raise ValueError(f"O arquivo precisa ter os 75 bairros. Faltam {len(faltando)}: "
                             f"{', '.join(sorted(faltando)[:10])}{'...' if len(faltando) > 10 else ''}")
        salvar_populacao(dict(zip(limpo["chave"], limpo["populacao"])),
                         f"Arquivo enviado pelo app ({nome_arquivo}) em {datetime.now():%d/%m/%Y %H:%M}")
    elif tipo == "156" and adicionar_ao_periodo:
        ss.bases[tipo] = juntar_156(ss.bases[tipo], limpo)
        ss.origens[tipo] += f" + {nome_arquivo}"
    else:
        ss.bases[tipo] = limpo
        ss.origens[tipo] = f"Arquivo enviado: {nome_arquivo}"
    if tipo != "populacao":
        ss.relatorios[tipo] = relatorio
    ss.versao_dados = _nova_versao()

    resumo = {"arquivo": nome_arquivo, "tipo": NOMES_TIPOS[tipo], "tipo_chave": tipo,
              "enviado_em": datetime.now().strftime("%d/%m/%Y %H:%M"), **relatorio}
    ss.historico_uploads.append(resumo)
    return resumo


def restaurar_padrao() -> None:
    for chave in ["bases", "relatorios", "origens"]:
        ss.pop(chave, None)
    modelos, historico = ss.get("modelos", {}), ss.get("historico_uploads", [])
    iniciar()
    ss.modelos, ss.historico_uploads = modelos, historico
    ss.versao_dados = _nova_versao()


# ---------- Cálculos pesados com cache (recalculados só quando os dados mudam) ----------
# Parâmetros com "_" na frente não entram na chave do cache; a chave é a versão dos dados.

@st.cache_data(show_spinner="Calculando indicadores dos bairros...", max_entries=5)
def _indicadores(versao: str, _chamados, _equipamentos, _esaude) -> pd.DataFrame:
    return calcular_indicadores(_chamados, _equipamentos, _esaude)


@st.cache_data(show_spinner="Montando a série semanal...", max_entries=5)
def _serie(versao: str, _chamados) -> pd.DataFrame:
    return serie_semanal(_chamados)


@st.cache_data(show_spinner="Preparando a base de atrasos...", max_entries=3)
def _base_atraso(versao: str, _chamados) -> pd.DataFrame:
    return montar_base_atraso(_chamados)


def indicadores() -> pd.DataFrame:
    return _indicadores(ss.versao_dados, chamados(), equipamentos(), esaude())


def notas(pesos: dict) -> pd.DataFrame:
    """Indicadores + notas com os pesos escolhidos, já com nome do bairro."""
    ind = indicadores()
    return ind[["bairro", "regional", "populacao", "pouca_populacao"]].join(calcular_notas(ind, pesos))


def serie() -> pd.DataFrame:
    return _serie(ss.versao_dados, chamados())


def base_atraso() -> pd.DataFrame:
    return _base_atraso(ss.versao_dados, chamados())


# ---------- Modelos ----------

def guardar_modelo(resultado: ResultadoTreino) -> None:
    ss.modelos[resultado.tipo] = resultado


def modelo(tipo: str) -> ResultadoTreino | None:
    return ss.modelos.get(tipo)


def modelo_desatualizado(tipo: str) -> bool:
    """True se o modelo foi treinado com dados diferentes dos atuais."""
    atual = modelo(tipo)
    return atual is not None and atual.versao_dados != ss.versao_dados


def periodo_chamados() -> tuple[pd.Timestamp, pd.Timestamp]:
    return chamados()["data"].min(), chamados()["data"].max()
