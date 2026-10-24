"""Regressão: "quantos dias a prefeitura vai levar para responder este chamado?"

Usa as mesmas características do classificador, mas prevê um número (dias) em vez
de sim/não. Só entram chamados que já foram respondidos (os que estão abertos ainda
não têm o número de dias final).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.modelos.catalogo import MODELOS_REGRESSAO
from src.modelos.comum import CARACTERISTICAS, criar_pipeline, entrada_para_previsao, importancia_por_caracteristica
from src.modelos.registro import ResultadoTreino

TOLERANCIA_DIAS = 7  # usada na métrica "% de previsões com erro de até 7 dias"


def treinar_regressao(base: pd.DataFrame, modelo_nome: str, parametros: dict, versao_dados: str,
                      periodo: tuple, tamanho_amostra: int = 40_000,
                      caracteristicas: list[str] | None = None,
                      proporcao_teste: float = 0.25) -> ResultadoTreino:
    """Treina numa amostra de chamados respondidos (75% treino / 25% teste)."""
    caracteristicas = caracteristicas or list(CARACTERISTICAS)
    respondidos = base.dropna(subset=["dias_resposta"])
    amostra = respondidos.sample(n=min(tamanho_amostra, len(respondidos)), random_state=42)
    X, y = amostra[caracteristicas], amostra["dias_resposta"].astype(float)
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=proporcao_teste, random_state=42)

    # prever_log: o modelo aprende log(1 + dias) e a previsão volta para dias, para os poucos
    # chamados muito demorados (até ~280 dias) não "puxarem" todas as previsões para cima
    pipeline = criar_pipeline(MODELOS_REGRESSAO, modelo_nome, parametros, caracteristicas,
                              prever_log=True).fit(X_treino, y_treino)
    previsto = np.clip(pipeline.predict(X_teste), 0, None)  # dias negativos não existem
    erro = np.abs(y_teste.to_numpy() - previsto)
    # "Chute": prever sempre a mediana de dias do treino, sem olhar o chamado
    chute = np.full(len(y_teste), y_treino.median())

    metricas = {
        "mae": mean_absolute_error(y_teste, previsto),
        "erro_mediano": median_absolute_error(y_teste, previsto),
        "r2": r2_score(y_teste, previsto),
        "mae_chute": mean_absolute_error(y_teste, chute),
        f"pct_erro_ate_{TOLERANCIA_DIAS}_dias": (erro <= TOLERANCIA_DIAS).mean() * 100,
        "mediana_dias_real": float(y_teste.median()),
    }
    return ResultadoTreino(
        tipo="regressao", modelo_nome=modelo_nome, parametros=parametros, pipeline=pipeline,
        metricas=metricas, n_registros=len(amostra), periodo=periodo, versao_dados=versao_dados,
        detalhes={
            "real_vs_previsto": pd.DataFrame({"Real": y_teste.to_numpy(), "Previsto": previsto}),
            # Alguns chamados de teste com a previsão, para mostrar exemplos reais na tela
            "exemplos": amostra.loc[X_teste.index, ["Orgao", "Assunto", "Subdivisao", "chave", "dias_resposta"]]
                        .assign(previsto=previsto).sample(min(500, len(X_teste)), random_state=7),
            "importancia": importancia_por_caracteristica(pipeline, X_teste, y_teste, "neg_mean_absolute_error"),
            "caracteristicas": caracteristicas, "tamanho_amostra": tamanho_amostra,
            "n_treino": len(X_treino), "n_teste": len(X_teste),
        },
    )


def prever_dias(pipeline: Pipeline, chamado: dict) -> float:
    """Número de dias previsto para um chamado novo."""
    return float(max(pipeline.predict(entrada_para_previsao(pipeline, chamado))[0], 0))


def prever_dias_varios(pipeline: Pipeline, chamados: pd.DataFrame) -> np.ndarray:
    """Dias previstos para vários chamados de uma vez (usado na comparação entre bairros)."""
    colunas = list(pipeline.named_steps["preparo"].feature_names_in_)
    return np.clip(pipeline.predict(chamados[colunas].astype(str)), 0, None)
