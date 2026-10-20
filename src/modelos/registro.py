"""Guarda os modelos treinados (em memória e em disco) e o histórico de treinos."""
from dataclasses import dataclass, field
from datetime import datetime

import joblib
import pandas as pd

from src.config import PASTA_MODELOS

ARQUIVO_HISTORICO = PASTA_MODELOS / "historico_treinos.csv"


@dataclass
class ResultadoTreino:
    tipo: str               # "previsao" ou "classificacao"
    modelo_nome: str
    parametros: dict
    pipeline: object        # Pipeline do scikit-learn já treinado
    metricas: dict
    n_registros: int
    periodo: tuple          # (data inicial, data final) dos dados usados
    versao_dados: str       # para saber se o modelo ficou desatualizado após um upload
    detalhes: dict = field(default_factory=dict)  # tabelas usadas nos gráficos
    treinado_em: datetime = field(default_factory=datetime.now)


def salvar(resultado: ResultadoTreino) -> str:
    """Salva o modelo em modelos_salvos/ e registra uma linha no histórico."""
    PASTA_MODELOS.mkdir(exist_ok=True)
    carimbo = resultado.treinado_em.strftime("%Y%m%d_%H%M%S")
    caminho = PASTA_MODELOS / f"{resultado.tipo}_{carimbo}.joblib"
    joblib.dump(resultado, caminho)

    linha = {
        "treinado_em": resultado.treinado_em.strftime("%d/%m/%Y %H:%M:%S"),
        "tipo": resultado.tipo,
        "modelo": resultado.modelo_nome,
        "parametros": str(resultado.parametros),
        "registros": resultado.n_registros,
        "periodo_dados": f"{resultado.periodo[0]:%d/%m/%Y} a {resultado.periodo[1]:%d/%m/%Y}",
        **{k: round(v, 4) for k, v in resultado.metricas.items() if isinstance(v, (int, float))},
        "arquivo": caminho.name,
    }
    historico = pd.concat([ler_historico(), pd.DataFrame([linha])], ignore_index=True)
    historico.to_csv(ARQUIVO_HISTORICO, sep=";", index=False, encoding="utf-8")
    return caminho.name


def ler_historico(tipo: str | None = None) -> pd.DataFrame:
    if not ARQUIVO_HISTORICO.exists():
        return pd.DataFrame()
    historico = pd.read_csv(ARQUIVO_HISTORICO, sep=";", encoding="utf-8")
    return historico[historico["tipo"] == tipo] if tipo else historico


def carregar_ultimo(tipo: str) -> ResultadoTreino | None:
    """Carrega o modelo salvo mais recente de um tipo (ou None se não houver)."""
    arquivos = sorted(PASTA_MODELOS.glob(f"{tipo}_*.joblib"))
    return joblib.load(arquivos[-1]) if arquivos else None
