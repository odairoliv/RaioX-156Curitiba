"""Partes comuns aos dois modelos (classificação e regressão).

As características são todas categóricas (texto). Antes de chegar ao algoritmo elas
são transformadas em números de um destes jeitos:
- one-hot: cada valor vira uma coluna 0/1 ("Assunto = Iluminação Pública" vale 1
  quando o chamado é desse assunto) — usado pela maioria dos algoritmos;
- categorias: cada valor vira um código (Iluminação Pública = 17) e o algoritmo sabe
  que é uma categoria, não um número — usado pelo Gradient Boosting.
Nos dois casos, valores raros (menos de 20 chamados no treino) são agrupados em "outros".
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from src.modelos.catalogo import criar_modelo, tipo_de_entrada

MINIMO_POR_VALOR = 20  # valores com menos chamados que isso no treino viram "outros"

# Característica -> nome amigável
CARACTERISTICAS = {
    "Assunto": "Assunto",
    "Subdivisao": "Detalhe do assunto",
    "Orgao": "Órgão responsável",
    "Tipo": "Tipo de chamado",
    "Origem": "Canal de abertura",
    "chave": "Bairro",
    "Regional": "Regional",
    "dia_semana": "Dia da semana",
    "mes_nome": "Mês de abertura",
}

# Grupos de características ligados às hipóteses. Treinar com só um grupo responde:
# "quanto dá para prever o atraso sabendo só ONDE / só O QUÊ?"
GRUPOS = {
    "Onde (bairro e regional) — H1": ["chave", "Regional"],
    "O quê (órgão, assunto e detalhe) — H2": ["Orgao", "Assunto", "Subdivisao"],
    "Como (tipo e canal) — H3 e H4": ["Tipo", "Origem"],
    "Quando (dia da semana e mês)": ["dia_semana", "mes_nome"],
}


def criar_preparo(caracteristicas: list[str], entrada: str = "onehot") -> ColumnTransformer:
    """Transforma as características escolhidas em números (one-hot ou códigos de categoria)."""
    if entrada == "categorias":
        codificador = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                                     encoded_missing_value=np.nan, min_frequency=MINIMO_POR_VALOR, max_categories=250)
    else:
        codificador = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=MINIMO_POR_VALOR)
    return ColumnTransformer([("categoricas", codificador, caracteristicas)])


def criar_pipeline(catalogo: dict, modelo_nome: str, parametros: dict, caracteristicas: list[str],
                   prever_log: bool = False) -> Pipeline:
    """Monta "preparo dos dados -> algoritmo" para o algoritmo e as características escolhidos.

    `prever_log=True` (regressão de dias): o algoritmo aprende log(1 + dias) e a previsão é
    convertida de volta para dias, para os poucos chamados muito demorados não "puxarem" tudo.
    """
    entrada = tipo_de_entrada(catalogo, modelo_nome)
    modelo = criar_modelo(catalogo, modelo_nome, parametros)
    if entrada == "categorias":
        modelo.set_params(categorical_features=[True] * len(caracteristicas))
    if prever_log:
        modelo = TransformedTargetRegressor(modelo, func=np.log1p, inverse_func=np.expm1)
    return Pipeline([("preparo", criar_preparo(caracteristicas, entrada)), ("modelo", modelo)])


def importancia_por_caracteristica(pipeline, X_teste: pd.DataFrame, y_teste: pd.Series, metrica: str,
                                   amostra: int = 1_500) -> pd.Series:
    """Importância por permutação (em %), igual para qualquer algoritmo.

    Para cada característica, embaralhamos os valores dela no teste e medimos quanto
    o modelo PIORA. Se embaralhar o bairro quase não muda nada, o bairro pouco importa.
    O resultado é a piora de cada característica dividida pela soma das pioras.
    """
    pontos = X_teste.sample(min(amostra, len(X_teste)), random_state=42)
    resultado = permutation_importance(pipeline, pontos, y_teste.loc[pontos.index], scoring=metrica,
                                       n_repeats=2, random_state=42)
    piora = pd.Series(np.clip(resultado.importances_mean, 0, None), index=X_teste.columns)
    piora.index = piora.index.map(CARACTERISTICAS)
    total = piora.sum()
    return (piora / total * 100 if total > 0 else piora).sort_values()


def entrada_para_previsao(pipeline, chamado: dict) -> pd.DataFrame:
    """Monta a linha de entrada com as mesmas colunas usadas no treino do pipeline."""
    colunas = list(pipeline.named_steps["preparo"].feature_names_in_)
    return pd.DataFrame([{c: str(chamado.get(c, "")) for c in colunas}])
