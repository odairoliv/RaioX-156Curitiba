"""Classificação: "este chamado vai ser respondido com atraso (mais de 30 dias)?"

O modelo aprende, com os chamados antigos, quais combinações de órgão, assunto,
bairro, canal etc. costumam atrasar. As métricas são calculadas em chamados
separados para teste, que o modelo nunca viu.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.modelos.catalogo import MODELOS_CLASSIFICACAO, criar_modelo
from src.modelos.comum import CARACTERISTICAS, criar_preparo, entrada_para_previsao, importancia_por_caracteristica
from src.modelos.registro import ResultadoTreino


def treinar_classificador(base: pd.DataFrame, modelo_nome: str, parametros: dict, versao_dados: str,
                          periodo: tuple, tamanho_amostra: int = 40_000,
                          caracteristicas: list[str] | None = None,
                          proporcao_teste: float = 0.25) -> ResultadoTreino:
    """Treina numa amostra (75% treino / 25% teste) e devolve métricas e detalhes para os gráficos."""
    caracteristicas = caracteristicas or list(CARACTERISTICAS)
    amostra = base.sample(n=min(tamanho_amostra, len(base)), random_state=42)
    X, y = amostra[caracteristicas], amostra["atrasou"]
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=proporcao_teste, random_state=42, stratify=y)

    modelo = criar_modelo(MODELOS_CLASSIFICACAO, modelo_nome, parametros)
    pipeline = Pipeline([("preparo", criar_preparo(caracteristicas)), ("modelo", modelo)]).fit(X_treino, y_treino)
    previsto = pipeline.predict(X_teste)
    maioria = int(y_treino.mode()[0])

    metricas = {
        "acuracia": accuracy_score(y_teste, previsto),
        "precisao": precision_score(y_teste, previsto, zero_division=0),
        "recall": recall_score(y_teste, previsto, zero_division=0),
        "f1": f1_score(y_teste, previsto, zero_division=0),
        # Acurácia de quem "chuta" sempre a resposta mais comum — o modelo precisa superar isso
        "acuracia_chute": accuracy_score(y_teste, np.full(len(y_teste), maioria)),
        "taxa_atraso_%": y.mean() * 100,
    }
    return ResultadoTreino(
        tipo="classificacao", modelo_nome=modelo_nome, parametros=parametros, pipeline=pipeline,
        metricas=metricas, n_registros=len(amostra), periodo=periodo, versao_dados=versao_dados,
        detalhes={
            "matriz_confusao": confusion_matrix(y_teste, previsto, labels=[0, 1]),
            "importancia": importancia_por_caracteristica(pipeline, X_teste, y_teste, "f1"),
            "caracteristicas": caracteristicas, "tamanho_amostra": tamanho_amostra,
            "n_treino": len(X_treino), "n_teste": len(X_teste),
        },
    )


def prever_chamado(pipeline: Pipeline, chamado: dict) -> float:
    """Probabilidade (0 a 1) de um chamado novo atrasar."""
    return float(pipeline.predict_proba(entrada_para_previsao(pipeline, chamado))[0, 1])
