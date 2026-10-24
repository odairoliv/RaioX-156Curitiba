"""Classificação: "este chamado vai ser respondido com atraso (mais de 30 dias)?"

O modelo aprende, com os chamados antigos, quais combinações de órgão, assunto,
bairro, canal etc. costumam atrasar. Para cada chamado ele calcula uma CHANCE de
atraso (0 a 100%). O LIMIAR decide a partir de que chance o chamado é marcado como
"vai atrasar" — por padrão 50%, mas o usuário pode mudar na tela sem re-treinar.
As métricas são calculadas em chamados separados para teste, que o modelo nunca viu.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.modelos.catalogo import MODELOS_CLASSIFICACAO
from src.modelos.comum import CARACTERISTICAS, criar_pipeline, entrada_para_previsao, importancia_por_caracteristica
from src.modelos.registro import ResultadoTreino

LIMIAR_PADRAO = 0.5
COLUNAS_EXEMPLO = ["Orgao", "Assunto", "Subdivisao", "chave", "dias_resposta", "atrasou"]


def metricas_no_limiar(y_real: pd.Series, chance: np.ndarray, limiar: float) -> dict:
    """Métricas quando o chamado é marcado como "vai atrasar" se a chance for >= limiar."""
    previsto = (chance >= limiar).astype(int)
    maioria = int(y_real.mean() >= 0.5)
    return {
        "acuracia": accuracy_score(y_real, previsto),
        "precisao": precision_score(y_real, previsto, zero_division=0),
        "recall": recall_score(y_real, previsto, zero_division=0),
        "f1": f1_score(y_real, previsto, zero_division=0),
        # Acurácia de quem "chuta" sempre a resposta mais comum — o modelo precisa superar isso
        "acuracia_chute": accuracy_score(y_real, np.full(len(y_real), maioria)),
        "matriz": confusion_matrix(y_real, previsto, labels=[0, 1]),
    }


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

    pipeline = criar_pipeline(MODELOS_CLASSIFICACAO, modelo_nome, parametros, caracteristicas).fit(X_treino, y_treino)
    chance = pipeline.predict_proba(X_teste)[:, 1]
    metricas = metricas_no_limiar(y_teste, chance, LIMIAR_PADRAO)
    matriz = metricas.pop("matriz")
    metricas["taxa_atraso_%"] = y.mean() * 100

    # Alguns chamados de teste com a chance calculada, para mostrar exemplos reais na tela
    exemplos = amostra.loc[X_teste.index, COLUNAS_EXEMPLO].assign(chance=chance).sample(
        min(500, len(X_teste)), random_state=7)
    return ResultadoTreino(
        tipo="classificacao", modelo_nome=modelo_nome, parametros=parametros, pipeline=pipeline,
        metricas=metricas, n_registros=len(amostra), periodo=periodo, versao_dados=versao_dados,
        detalhes={
            "matriz_confusao": matriz, "y_teste": y_teste.to_numpy(), "chance_teste": chance, "exemplos": exemplos,
            "importancia": importancia_por_caracteristica(pipeline, X_teste, y_teste, "f1"),
            "caracteristicas": caracteristicas, "tamanho_amostra": tamanho_amostra,
            "n_treino": len(X_treino), "n_teste": len(X_teste),
        },
    )


def prever_chamado(pipeline: Pipeline, chamado: dict) -> float:
    """Chance (0 a 1) de um chamado novo atrasar."""
    return float(pipeline.predict_proba(entrada_para_previsao(pipeline, chamado))[0, 1])


def prever_varios(pipeline: Pipeline, chamados: pd.DataFrame) -> np.ndarray:
    """Chance de atraso para vários chamados de uma vez (usado na comparação entre bairros)."""
    colunas = list(pipeline.named_steps["preparo"].feature_names_in_)
    return pipeline.predict_proba(chamados[colunas].astype(str))[:, 1]
