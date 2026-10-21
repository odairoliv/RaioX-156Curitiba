"""Previsão de tendência: quantos chamados cada bairro terá nas próximas semanas.

Ideia (para explicar na apresentação):
- Contamos os chamados de cada bairro e categoria, semana a semana.
- O modelo aprende a relação entre "as últimas 4 semanas" e "a semana seguinte".
- Para prever várias semanas à frente, a previsão de uma semana vira entrada da
  próxima (previsão recursiva).
- Avaliamos o modelo escondendo as últimas semanas: ele prevê sem ver a resposta
  e comparamos com o que realmente aconteceu.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.bairros import tabela_bairros
from src.categorias import TODAS_CATEGORIAS
from src.indicadores import apenas_demandas
from src.modelos.catalogo import MODELOS_REGRESSAO, criar_modelo
from src.modelos.registro import ResultadoTreino

N_SEMANAS_ANTERIORES = 4
COLUNAS_LAG = [f"semana_menos_{i}" for i in range(1, N_SEMANAS_ANTERIORES + 1)]
COLUNAS_NUMERICAS = COLUNAS_LAG + ["media_4_semanas", "tendencia", "nivel_historico"]


def serie_semanal(chamados: pd.DataFrame) -> pd.DataFrame:
    """Tabela longa: bairro, categoria, semana e quantidade de demandas (zeros incluídos).

    Semanas incompletas no começo e no fim do período são descartadas para não
    parecer que o número de chamados caiu.
    """
    demandas = apenas_demandas(chamados)
    inicio, fim = chamados["data"].min(), chamados["data"].max()
    todas = pd.date_range(inicio.to_period("W-SUN").start_time, fim, freq="W-MON")
    semanas = [s for s in todas if s >= inicio and s + pd.Timedelta(days=6) <= fim]
    if len(semanas) < N_SEMANAS_ANTERIORES + 3:
        raise ValueError("São necessárias pelo menos 7 semanas completas de dados para treinar a previsão.")

    contagem = demandas.groupby(["chave", "categoria", "semana"], observed=True).size()
    grade = pd.MultiIndex.from_product(
        [tabela_bairros()["chave"], TODAS_CATEGORIAS, semanas], names=["chave", "categoria", "semana"])
    serie = contagem.reindex(grade, fill_value=0).rename("qtd").reset_index()
    serie["chave"] = serie["chave"].astype(str)
    serie["categoria"] = serie["categoria"].astype(str)
    return serie


def _completar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Colunas derivadas das semanas anteriores (mesma fórmula no treino e na previsão)."""
    df["media_4_semanas"] = df[COLUNAS_LAG].mean(axis=1)
    df["tendencia"] = df["semana_menos_1"] - df["semana_menos_4"]
    return df


def montar_features(serie: pd.DataFrame) -> pd.DataFrame:
    """Para cada semana, coloca na mesma linha os valores das 4 semanas anteriores."""
    df = serie.sort_values(["chave", "categoria", "semana"]).copy()
    grupos = df.groupby(["chave", "categoria"])["qtd"]
    for i, coluna in enumerate(COLUNAS_LAG, start=1):
        df[coluna] = grupos.shift(i)
    # Média de todas as semanas anteriores: diz ao modelo o "tamanho" típico daquela série
    df["nivel_historico"] = grupos.transform(lambda s: s.shift(1).expanding().mean())
    df = _completar_features(df)
    return df.dropna(subset=COLUNAS_LAG).reset_index(drop=True)


def _criar_pipeline(modelo_nome: str, parametros: dict) -> Pipeline:
    preparo = ColumnTransformer([
        ("numericas", StandardScaler(), COLUNAS_NUMERICAS),
        ("categoria", OneHotEncoder(handle_unknown="ignore"), ["categoria"]),
    ])
    return Pipeline([("preparo", preparo), ("modelo", criar_modelo(MODELOS_REGRESSAO, modelo_nome, parametros))])


def treinar_previsao(serie: pd.DataFrame, modelo_nome: str, parametros: dict, versao_dados: str,
                     semanas_teste: int = 6) -> ResultadoTreino:
    """Treina, avalia nas últimas semanas e re-treina com tudo para prever o futuro."""
    dados = montar_features(serie)
    semanas = sorted(dados["semana"].unique())
    semanas_teste = min(semanas_teste, len(semanas) - 2)
    corte = semanas[-semanas_teste]
    treino, teste = dados[dados["semana"] < corte], dados[dados["semana"] >= corte]
    X = COLUNAS_NUMERICAS + ["categoria"]

    avaliacao = _criar_pipeline(modelo_nome, parametros).fit(treino[X], treino["qtd"])
    teste = teste.assign(previsto=np.clip(avaliacao.predict(teste[X]), 0, None))

    # Comparação com uma regra simples: "a próxima semana será igual à média das últimas 4"
    metricas = {
        "mae": mean_absolute_error(teste["qtd"], teste["previsto"]),
        "rmse": float(np.sqrt(mean_squared_error(teste["qtd"], teste["previsto"]))),
        "r2": r2_score(teste["qtd"], teste["previsto"]),
        "mae_regra_simples": mean_absolute_error(teste["qtd"], teste["media_4_semanas"]),
    }
    metricas["ganho_sobre_regra_simples_%"] = (1 - metricas["mae"] / metricas["mae_regra_simples"]) * 100

    final = _criar_pipeline(modelo_nome, parametros).fit(dados[X], dados["qtd"])
    cidade = (teste.groupby("semana")[["qtd", "previsto", "media_4_semanas"]].sum()
              .rename(columns={"qtd": "Real", "previsto": "Modelo", "media_4_semanas": "Regra simples"}))
    return ResultadoTreino(
        tipo="previsao", modelo_nome=modelo_nome, parametros=parametros, pipeline=final,
        metricas=metricas, n_registros=len(dados), versao_dados=versao_dados,
        periodo=(serie["semana"].min(), serie["semana"].max() + pd.Timedelta(days=6)),
        detalhes={"teste_cidade": cidade, "semanas_teste": semanas_teste,
                  "media_real_teste": float(teste["qtd"].mean()), "linhas_teste": len(teste),
                  "erro_por_categoria": teste.assign(erro=(teste["qtd"] - teste["previsto"]).abs())
                  .groupby("categoria")["erro"].mean().sort_values(ascending=False)},
    )


def prever_futuro(pipeline: Pipeline, serie: pd.DataFrame, semanas_a_frente: int) -> pd.DataFrame:
    """Prevê as próximas semanas para todos os bairros e categorias ao mesmo tempo."""
    tabela = serie.pivot_table(index=["chave", "categoria"], columns="semana", values="qtd", aggfunc="sum")
    historico = tabela.to_numpy(dtype=float)
    ultima_semana = tabela.columns.max()
    categorias = tabela.index.get_level_values("categoria")
    previsoes = []

    for passo in range(1, semanas_a_frente + 1):
        janela = historico[:, -N_SEMANAS_ANTERIORES:]
        entrada = pd.DataFrame(janela[:, ::-1], columns=COLUNAS_LAG)  # semana_menos_1 = mais recente
        entrada["nivel_historico"] = historico.mean(axis=1)
        entrada["categoria"] = categorias
        entrada = _completar_features(entrada)
        proxima = np.clip(pipeline.predict(entrada[COLUNAS_NUMERICAS + ["categoria"]]), 0, None)
        historico = np.column_stack([historico, proxima])
        previsoes.append(pd.DataFrame({
            "chave": tabela.index.get_level_values("chave"), "categoria": categorias,
            "semana": ultima_semana + pd.Timedelta(weeks=passo), "qtd": proxima,
        }))
    return pd.concat(previsoes, ignore_index=True)


def tendencia_por_bairro(serie: pd.DataFrame, futuro: pd.DataFrame, categorias: list | None = None) -> pd.DataFrame:
    """Compara a média semanal prevista com a média das últimas 4 semanas reais."""
    if categorias:
        serie = serie[serie["categoria"].isin(categorias)]
        futuro = futuro[futuro["categoria"].isin(categorias)]
    ultimas = sorted(serie["semana"].unique())[-N_SEMANAS_ANTERIORES:]
    recente = serie[serie["semana"].isin(ultimas)].groupby("chave")["qtd"].sum() / N_SEMANAS_ANTERIORES
    previsto = futuro.groupby("chave")["qtd"].sum() / futuro["semana"].nunique()
    resultado = pd.DataFrame({"media_recente": recente, "media_prevista": previsto})
    resultado["variacao_%"] = (resultado["media_prevista"] / resultado["media_recente"].replace(0, np.nan) - 1) * 100
    return resultado
