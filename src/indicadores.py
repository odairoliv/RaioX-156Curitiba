"""Indicadores por bairro e a "nota de saúde" (0 a 10).

Como a nota funciona (versão para explicar na apresentação):
1. Para cada dimensão calculamos um número por bairro. Ex.: chamados de trânsito
   por mês para cada mil habitantes (ou por km², se a população não foi informada).
2. Comparamos os 75 bairros entre si: a nota de um bairro é a porcentagem de bairros
   que estão PIORES que ele, numa escala de 0 a 10. Nota 8 = melhor que ~80% dos bairros.
3. A nota final é a média ponderada das dimensões. Os pesos podem ser ajustados.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import categorias as cat
from src.bairros import normalizar_texto, tabela_bairros, usa_populacao
from src.config import PRAZO_ATRASO_DIAS
from src.preparacao import ITENS_SANEAMENTO

# Só estes tipos indicam um problema/demanda (elogios e sugestões ficam de fora)
TIPOS_DEMANDA = {"SOLICITACAO", "RECLAMACAO", "DENUNCIA"}
MINIMO_MORADORES_SANEAMENTO = 30
# Com menos moradores que isso, poucos chamados a mais ou a menos mudam muito a taxa
POPULACAO_MINIMA_CONFIAVEL = 2000


@dataclass(frozen=True)
class Dimensao:
    chave: str
    nome: str
    descricao: str
    maior_melhor: bool
    fonte: str  # base de onde vem o dado


DIMENSOES = [
    Dimensao("zeladoria", cat.ZELADORIA, "Chamados sobre buracos, iluminação, calçadas e drenagem.", False, "156"),
    Dimensao("transito", cat.TRANSITO, "Chamados sobre trânsito, sinalização e transporte coletivo.", False, "156"),
    Dimensao("limpeza", cat.LIMPEZA, "Chamados sobre limpeza, árvores, poluição, terrenos e animais.", False, "156"),
    Dimensao("seguranca", cat.SEGURANCA, "Chamados sobre situação de rua, Guarda Municipal, invasões e violência.", False, "156"),
    Dimensao("saude", cat.SAUDE, "Chamados sobre dengue, vigilância sanitária e unidades de saúde.", False, "156"),
    Dimensao("atendimento", "Atendimento da prefeitura", "Tempo de resposta e chamados atrasados no bairro.", False, "156"),
    Dimensao("infraestrutura", "Infraestrutura disponível", "Escolas, unidades de saúde, segurança, transporte e lazer.", True, "equipamentos"),
    Dimensao("saneamento", "Saneamento e moradia", "Moradores com água, esgoto, coleta de lixo e moradia adequados.", True, "esaude"),
]
DIMENSAO_POR_CHAVE = {d.chave: d for d in DIMENSOES}
DIMENSOES_DE_DEMANDA = {d.chave: d.nome for d in DIMENSOES if d.nome in cat.CATEGORIAS_DA_NOTA}

PRESETS_PESOS = {
    "Equilibrado": dict.fromkeys(DIMENSAO_POR_CHAVE, 1.0),
    "Vou morar no bairro": {"zeladoria": 2, "transito": 1, "limpeza": 1.5, "seguranca": 2, "saude": 1.5,
                            "atendimento": 1, "infraestrutura": 2, "saneamento": 2},
    "Vou visitar / me hospedar": {"zeladoria": 1, "transito": 2, "limpeza": 1.5, "seguranca": 2.5, "saude": 0.5,
                                  "atendimento": 0.5, "infraestrutura": 1, "saneamento": 0.5},
    "Gestão pública": {"zeladoria": 2, "transito": 2, "limpeza": 1.5, "seguranca": 1.5, "saude": 1.5,
                       "atendimento": 2.5, "infraestrutura": 1, "saneamento": 1.5},
}


def unidade_taxa() -> str:
    return "por mil habitantes" if usa_populacao() else "por km²"


def _denominador() -> pd.Series:
    """Divisor usado nas taxas: população em milhares ou área em km²."""
    bairros = tabela_bairros().set_index("chave")
    return bairros["populacao"] / 1000 if usa_populacao() else bairros["area_km2"]


def apenas_demandas(chamados: pd.DataFrame) -> pd.DataFrame:
    """Filtra solicitações, reclamações e denúncias."""
    # Em colunas "category" o map roda só nos valores distintos (rápido)
    tipos = chamados["Tipo"].map(normalizar_texto)
    return chamados[tipos.isin(TIPOS_DEMANDA).to_numpy()]


def contagem_por_categoria(chamados: pd.DataFrame) -> pd.DataFrame:
    """Tabela bairro x categoria com o número de demandas."""
    contagem = apenas_demandas(chamados).groupby(["chave", "categoria"], observed=True).size().unstack(fill_value=0)
    contagem.index, contagem.columns = contagem.index.astype(str), contagem.columns.astype(str)
    return contagem.reindex(index=tabela_bairros()["chave"], columns=cat.TODAS_CATEGORIAS, fill_value=0)


def meses_no_periodo(chamados: pd.DataFrame) -> float:
    dias = (chamados["data"].max() - chamados["data"].min()).days + 1
    return max(dias / 30.44, 1 / 30.44)


def _taxas_de_demanda(contagens: pd.DataFrame, meses: float) -> pd.DataFrame:
    """Converte contagens em 'chamados por mês por mil habitantes (ou km²)'."""
    denominador = _denominador()
    return pd.DataFrame({
        f"taxa_{chave}": contagens[nome] / meses / denominador
        for chave, nome in DIMENSOES_DE_DEMANDA.items()
    })


def _indicadores_atendimento(chamados: pd.DataFrame, data_ref: pd.Timestamp) -> pd.DataFrame:
    """Mediana de dias até a resposta e % de chamados atrasados (mais de 30 dias sem resposta)."""
    demandas = apenas_demandas(chamados)
    mediana = demandas.dropna(subset=["dias_resposta"]).groupby("chave", observed=True)["dias_resposta"].median()
    # Só faz sentido avaliar atraso de chamados que já tiveram tempo de ser respondidos
    maduros = demandas[demandas["data"] <= data_ref - pd.Timedelta(days=PRAZO_ATRASO_DIAS)]
    atrasou = (maduros["dias_resposta"] > PRAZO_ATRASO_DIAS) | (maduros["pendente"] & maduros["dias_resposta"].isna())
    pct_atrasados = atrasou.groupby(maduros["chave"], observed=True).mean() * 100
    return pd.DataFrame({"mediana_dias_resposta": mediana, "pct_atrasados": pct_atrasados})


def _indicadores_infraestrutura(equipamentos: pd.DataFrame | None) -> pd.DataFrame:
    if equipamentos is None or equipamentos.empty:
        return pd.DataFrame(index=tabela_bairros()["chave"], data={"taxa_infraestrutura": np.nan})
    por_tema = pd.crosstab(equipamentos["chave"], equipamentos["tema"])
    por_tema = por_tema.reindex(tabela_bairros()["chave"], fill_value=0)
    resultado = por_tema.add_prefix("equip_")
    resultado["total_equipamentos"] = por_tema.sum(axis=1)
    resultado["taxa_infraestrutura"] = resultado["total_equipamentos"] / _denominador()
    return resultado


def _indicadores_saneamento(esaude: pd.DataFrame | None) -> pd.DataFrame:
    if esaude is None or esaude.empty:
        return pd.DataFrame(index=tabela_bairros()["chave"], data={"pct_saneamento": np.nan})
    grupos = esaude.groupby("chave")
    resultado = (grupos[list(ITENS_SANEAMENTO)].mean() * 100).add_prefix("pct_")
    resultado["moradores_avaliados"] = grupos.size()
    resultado["pct_saneamento"] = resultado[[f"pct_{i}" for i in ITENS_SANEAMENTO]].mean(axis=1)
    # Com poucos moradores avaliados a porcentagem não é confiável
    resultado.loc[resultado["moradores_avaliados"] < MINIMO_MORADORES_SANEAMENTO, "pct_saneamento"] = np.nan
    return resultado


def calcular_indicadores(chamados: pd.DataFrame, equipamentos: pd.DataFrame | None = None,
                         esaude: pd.DataFrame | None = None) -> pd.DataFrame:
    """Uma linha por bairro com todos os números usados na nota."""
    bairros = tabela_bairros().set_index("chave")
    contagens = contagem_por_categoria(chamados)
    data_ref = chamados["data"].max()

    ind = bairros.join([
        _taxas_de_demanda(contagens, meses_no_periodo(chamados)),
        contagens.add_prefix("qtd_"),
        _indicadores_atendimento(chamados, data_ref),
        _indicadores_infraestrutura(equipamentos),
        _indicadores_saneamento(esaude),
    ])
    ind["total_demandas"] = contagens.sum(axis=1)
    tipos = chamados["Tipo"].map(normalizar_texto)
    ind["elogios"] = chamados[(tipos == "ELOGIO").to_numpy()].groupby("chave", observed=True).size()
    ind["elogios"] = ind["elogios"].fillna(0).astype(int)
    ind["pouca_populacao"] = ind["populacao"] < POPULACAO_MINIMA_CONFIAVEL
    return ind


def _percentil(valores: pd.Series, referencia: pd.Series) -> pd.Series:
    """Fração (0 a 1) dos valores de referência que são menores que cada valor."""
    ref = np.sort(referencia.dropna().to_numpy(dtype=float))
    if len(ref) == 0:
        return pd.Series(np.nan, index=valores.index)
    x = valores.to_numpy(dtype=float)
    menores = np.searchsorted(ref, x, side="left")
    menores_ou_iguais = np.searchsorted(ref, x, side="right")
    pct = (menores + menores_ou_iguais) / 2 / len(ref)
    return pd.Series(np.where(np.isnan(x), np.nan, pct), index=valores.index)


def _nota(valores: pd.Series, maior_melhor: bool, referencia: pd.Series | None = None) -> pd.Series:
    """Converte um indicador em nota 0–10 comparando com os outros bairros."""
    pct = _percentil(valores, valores if referencia is None else referencia)
    return (pct if maior_melhor else 1 - pct) * 10


def classificar(nota: float) -> str:
    if pd.isna(nota):
        return "Sem dados"
    if nota >= 6.5:
        return "Boa"
    if nota >= 4:
        return "Atenção"
    return "Crítica"


def calcular_notas(ind: pd.DataFrame, pesos: dict, referencia: pd.DataFrame | None = None) -> pd.DataFrame:
    """Calcula a nota de cada dimensão e a nota final.

    `referencia` permite comparar valores projetados com a distribuição ATUAL
    (usado na previsão: "como o bairro ficaria comparado aos bairros de hoje").
    """
    ref = ind if referencia is None else referencia
    notas = pd.DataFrame(index=ind.index)
    for chave in DIMENSOES_DE_DEMANDA:
        notas[chave] = _nota(ind[f"taxa_{chave}"], False, ref[f"taxa_{chave}"])
    notas["atendimento"] = pd.concat([
        _nota(ind["mediana_dias_resposta"], False, ref["mediana_dias_resposta"]),
        _nota(ind["pct_atrasados"], False, ref["pct_atrasados"]),
    ], axis=1).mean(axis=1)
    notas["infraestrutura"] = _nota(ind["taxa_infraestrutura"], True, ref["taxa_infraestrutura"])
    notas["saneamento"] = _nota(ind["pct_saneamento"], True, ref["pct_saneamento"])

    # Média ponderada ignorando dimensões sem dados (o peso delas é redistribuído)
    pesos_serie = pd.Series(pesos).reindex(notas.columns).fillna(0)
    peso_valido = notas.notna().mul(pesos_serie, axis=1)
    notas["nota_final"] = notas.fillna(0).mul(pesos_serie, axis=1).sum(axis=1) / peso_valido.sum(axis=1)
    notas["situacao"] = notas["nota_final"].map(classificar)
    notas["posicao"] = notas["nota_final"].rank(ascending=False, method="min").astype("Int64")
    return notas


def notas_projetadas(ind_atual: pd.DataFrame, contagens_previstas: pd.DataFrame, meses: float,
                     pesos: dict) -> pd.DataFrame:
    """Recalcula as notas trocando as taxas de demanda pelas previstas pelo modelo."""
    projetado = ind_atual.copy()
    taxas = _taxas_de_demanda(contagens_previstas.reindex(columns=cat.TODAS_CATEGORIAS, fill_value=0), meses)
    projetado[taxas.columns] = taxas
    return calcular_notas(projetado, pesos, referencia=ind_atual)


def pontos_fortes_e_fracos(notas_bairro: pd.Series, limite_forte: float = 7, limite_fraco: float = 4) -> tuple[list, list]:
    """Lista as dimensões em que o bairro vai bem e mal (para os textos explicativos)."""
    fortes, fracos = [], []
    for d in DIMENSOES:
        valor = notas_bairro.get(d.chave)
        if pd.isna(valor):
            continue
        if valor >= limite_forte:
            fortes.append((d.nome, valor))
        elif valor < limite_fraco:
            fracos.append((d.nome, valor))
    return sorted(fortes, key=lambda x: -x[1]), sorted(fracos, key=lambda x: x[1])
