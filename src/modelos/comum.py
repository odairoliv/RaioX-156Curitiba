"""Partes comuns aos dois modelos (classificação e regressão).

As características são todas categóricas (texto). O OneHotEncoder transforma cada
valor em uma coluna 0/1: "Assunto = Iluminação Pública" vira uma coluna que vale 1
quando o chamado é desse assunto.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import OneHotEncoder

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


def criar_preparo(caracteristicas: list[str]) -> ColumnTransformer:
    """One-hot das características escolhidas. Valores raros (< 20 casos) viram "outros"."""
    return ColumnTransformer([
        ("categoricas", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20), caracteristicas),
    ])


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
