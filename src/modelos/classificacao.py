"""Classificação: "este chamado vai ser resolvido com atraso?"

Um chamado é considerado ATRASADO quando levou mais de 30 dias para ser respondido
(ou continua sem resposta depois de 30 dias). O modelo aprende, com os chamados
antigos, quais combinações de assunto, órgão, bairro, canal etc. costumam atrasar.
Útil para a prefeitura priorizar e para o cidadão saber o que esperar.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import PRAZO_ATRASO_DIAS
from src.indicadores import apenas_demandas
from src.modelos.catalogo import MODELOS_CLASSIFICACAO, criar_modelo
from src.modelos.registro import ResultadoTreino

DIAS_DA_SEMANA = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"]

# Características usadas pelo modelo (todas categóricas) e o nome amigável de cada uma
CARACTERISTICAS = {
    "Assunto": "Assunto",
    "Subdivisao": "Detalhe do assunto",
    "Orgao": "Órgão responsável",
    "Tipo": "Tipo de chamado",
    "Origem": "Canal de abertura",
    "chave": "Bairro",
    "Regional": "Regional",
    "dia_semana": "Dia da semana",
    "mes_nome": "Mês",
}


def montar_base_atraso(chamados: pd.DataFrame, prazo: int = PRAZO_ATRASO_DIAS) -> pd.DataFrame:
    """Monta a tabela de treino: características + coluna 'atrasou' (1 = sim, 0 = não)."""
    demandas = apenas_demandas(chamados)
    data_ref = chamados["data"].max()
    # Chamados recentes ainda não tiveram tempo de atrasar: ficam de fora
    maduros = demandas[demandas["data"] <= data_ref - pd.Timedelta(days=prazo)].copy()
    respondido = maduros["dias_resposta"].notna()
    maduros["atrasou"] = np.where(respondido, maduros["dias_resposta"] > prazo, maduros["pendente"])
    # Sem data de resposta e sem estar pendente = situação desconhecida
    maduros = maduros[respondido | maduros["pendente"]]

    maduros["dia_semana"] = maduros["data"].dt.dayofweek.map(dict(enumerate(DIAS_DA_SEMANA)))
    maduros["mes_nome"] = maduros["data"].dt.month.astype(str)
    base = maduros[list(CARACTERISTICAS)].astype(str)
    base["atrasou"] = maduros["atrasou"].astype(int)
    return base.reset_index(drop=True)


def _criar_pipeline(modelo_nome: str, parametros: dict) -> Pipeline:
    # Categorias raras (menos de 20 ocorrências) são agrupadas em "outros"
    preparo = ColumnTransformer([
        ("categoricas", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20), list(CARACTERISTICAS)),
    ])
    return Pipeline([("preparo", preparo), ("modelo", criar_modelo(MODELOS_CLASSIFICACAO, modelo_nome, parametros))])


def _importancia_por_caracteristica(pipeline: Pipeline) -> pd.Series | None:
    """Soma a importância de cada coluna 'one-hot' de volta na característica original."""
    modelo = pipeline.named_steps["modelo"]
    if hasattr(modelo, "feature_importances_"):
        valores = modelo.feature_importances_
    elif hasattr(modelo, "coef_"):
        valores = np.abs(modelo.coef_[0])
    else:
        return None  # KNN e Naive Bayes não oferecem importância diretamente
    # Os nomes gerados são do tipo "categoricas__Assunto_Coleta": casamos pelo prefixo
    nomes = [n.split("__", 1)[1] for n in pipeline.named_steps["preparo"].get_feature_names_out()]
    origem = [next(c for c in CARACTERISTICAS if n.startswith(c + "_")) for n in nomes]
    importancia = pd.Series(valores, index=origem).groupby(level=0).sum()
    importancia.index = importancia.index.map(CARACTERISTICAS)
    return (importancia / importancia.sum() * 100).sort_values()


def treinar_classificador(base: pd.DataFrame, modelo_nome: str, parametros: dict, versao_dados: str,
                          periodo: tuple, tamanho_amostra: int = 40_000,
                          proporcao_teste: float = 0.25) -> ResultadoTreino:
    """Treina o classificador numa amostra, avalia em dados separados e devolve as métricas."""
    amostra = base.sample(n=min(tamanho_amostra, len(base)), random_state=42)
    X, y = amostra[list(CARACTERISTICAS)], amostra["atrasou"]
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=proporcao_teste, random_state=42, stratify=y)

    pipeline = _criar_pipeline(modelo_nome, parametros).fit(X_treino, y_treino)
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
            "importancia": _importancia_por_caracteristica(pipeline),
            "n_treino": len(X_treino), "n_teste": len(X_teste),
        },
    )


def prever_chamado(pipeline: Pipeline, chamado: dict) -> float:
    """Probabilidade (0 a 1) de um chamado novo atrasar."""
    entrada = pd.DataFrame([{c: str(chamado.get(c, "")) for c in CARACTERISTICAS}])
    return float(pipeline.predict_proba(entrada)[0, 1])
