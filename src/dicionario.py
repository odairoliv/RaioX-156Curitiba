"""Dicionário de dados: o que significa cada base, termo e indicador.

Este módulo é a "fonte única" das explicações mostradas no app. Se um indicador
mudar em src/indicadores.py, a descrição dele deve ser atualizada aqui.
"""
from dataclasses import dataclass

import pandas as pd

from src import categorias as cat
from src.config import PRAZO_ATRASO_DIAS
from src.indicadores import DIMENSAO_POR_CHAVE, MINIMO_MORADORES_SANEAMENTO, unidade_taxa
from src.preparacao import ITENS_SANEAMENTO, TEMAS_INFRAESTRUTURA


@dataclass(frozen=True)
class Indicador:
    coluna: str          # nome da coluna em calcular_indicadores()
    nome: str
    dimensao: str        # chave da dimensão da nota
    o_que_mede: str
    como_calcula: str
    unidade: str
    fonte: str
    maior_melhor: bool
    casas: int = 1


EXEMPLOS_CATEGORIA = {
    cat.ZELADORIA: "buracos e pavimentação, iluminação pública, calçadas, drenagem e cabos danificados",
    cat.TRANSITO: "estacionamento irregular, semáforos, sinalização, lombadas, ônibus e pontos de ônibus",
    cat.LIMPEZA: "limpeza e varrição, poda e queda de árvores, poluição, terrenos baldios e animais",
    cat.SEGURANCA: "pessoas em situação de rua, pedidos à Guarda Municipal, invasões e violência contra crianças",
    cat.SAUDE: "dengue, vigilância sanitária e de zoonoses, unidades de saúde (UMS), UPAs e SAMU",
}


def _indicadores_de_demanda() -> list[Indicador]:
    lista = []
    for chave, categoria in [("zeladoria", cat.ZELADORIA), ("transito", cat.TRANSITO), ("limpeza", cat.LIMPEZA),
                             ("seguranca", cat.SEGURANCA), ("saude", cat.SAUDE)]:
        lista.append(Indicador(
            coluna=f"taxa_{chave}", nome=f"Chamados de {categoria.lower()}", dimensao=chave,
            o_que_mede=f"Quantos pedidos, reclamações e denúncias sobre {EXEMPLOS_CATEGORIA[categoria]} "
                       "o bairro registra na Central 156, em média, por mês, para cada mil moradores.",
            como_calcula="(nº de solicitações + reclamações + denúncias da categoria no período ÷ nº de meses "
                         "do período) ÷ (população do bairro ÷ 1.000)",
            unidade=f"chamados por mês {unidade_taxa()}", fonte="Central 156 + IBGE Censo 2022",
            maior_melhor=False, casas=2))
    return lista


INDICADORES = _indicadores_de_demanda() + [
    Indicador(
        coluna="mediana_dias_resposta", nome="Tempo típico de resposta", dimensao="atendimento",
        o_que_mede="Quantos dias a prefeitura costuma levar para responder um chamado do bairro. "
                   "É a mediana: metade dos chamados é respondida em até esse número de dias.",
        como_calcula="mediana de (data da resposta − data de abertura), considerando só chamados que já "
                     "têm data de resposta", unidade="dias", fonte="Central 156", maior_melhor=False, casas=0),
    Indicador(
        coluna="pct_atrasados", nome="Chamados atrasados", dimensao="atendimento",
        o_que_mede=f"De cada 100 chamados do bairro, quantos levaram mais de {PRAZO_ATRASO_DIAS} dias para "
                   f"ser respondidos (ou continuam sem resposta depois de {PRAZO_ATRASO_DIAS} dias).",
        como_calcula=f"chamados com mais de {PRAZO_ATRASO_DIAS} dias até a resposta, ou ainda abertos após "
                     f"{PRAZO_ATRASO_DIAS} dias ÷ chamados abertos há pelo menos {PRAZO_ATRASO_DIAS} dias × 100",
        unidade="%", fonte="Central 156", maior_melhor=False),
    Indicador(
        coluna="taxa_infraestrutura", nome="Equipamentos públicos", dimensao="infraestrutura",
        o_que_mede="Quantos equipamentos públicos o bairro tem para cada mil moradores: "
                   + ", ".join(t.lower() for t in TEMAS_INFRAESTRUTURA.values()) + ".",
        como_calcula="nº de equipamentos ativos desses temas no bairro ÷ (população ÷ 1.000)",
        unidade=f"equipamentos {unidade_taxa()}", fonte="Equipamentos urbanos + IBGE Censo 2022",
        maior_melhor=True, casas=2),
    Indicador(
        coluna="pct_saneamento", nome="Saneamento e moradia adequados", dimensao="saneamento",
        o_que_mede="Média, entre 4 itens, da porcentagem de moradores do bairro com: "
                   + "; ".join(i.lower() for i in ITENS_SANEAMENTO) + ".",
        como_calcula="para cada item: moradores com resposta adequada ÷ moradores que responderam × 100; "
                     "depois a média dos 4 itens. Cada morador conta uma vez. Bairros com menos de "
                     f"{MINIMO_MORADORES_SANEAMENTO} moradores avaliados ficam sem nota nesta dimensão.",
        unidade="%", fonte="E-Saúde (cadastro das unidades de saúde)", maior_melhor=True),
]
INDICADOR_POR_COLUNA = {i.coluna: i for i in INDICADORES}


def indicadores_da_dimensao(chave: str) -> list[Indicador]:
    return [i for i in INDICADORES if i.dimensao == chave]


def _numero(valor, casas: int) -> str:
    if valor is None or pd.isna(valor):
        return "—"
    return f"{valor:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def formatar_valor(indicador: Indicador, valor) -> str:
    texto = _numero(valor, indicador.casas)
    return f"{texto}%" if indicador.unidade == "%" else texto


def comparar_com_bairros(ind: pd.DataFrame, chave: str, indicador: Indicador) -> dict:
    """Valor do bairro, mediana da cidade e quantos bairros estão melhores ou piores."""
    valores = ind[indicador.coluna].dropna()
    valor = ind.loc[chave, indicador.coluna]
    if pd.isna(valor):
        return {"valor": valor, "mediana": valores.median(), "melhores": None, "piores": None, "total": len(valores)}
    outros = valores.drop(chave, errors="ignore")
    if indicador.maior_melhor:
        melhores, piores = (outros > valor).sum(), (outros < valor).sum()
    else:
        melhores, piores = (outros < valor).sum(), (outros > valor).sum()
    return {"valor": valor, "mediana": valores.median(), "melhores": int(melhores), "piores": int(piores),
            "total": len(valores)}


def explicar(ind: pd.DataFrame, chave: str, indicador: Indicador) -> str:
    """Frase com números reais: 'Batel registra X... A mediana dos bairros é Y. Está melhor que N...'."""
    c = comparar_com_bairros(ind, chave, indicador)
    if c["melhores"] is None:
        return "Sem dados suficientes para este bairro."
    unidade = "" if indicador.unidade == "%" else f" {indicador.unidade}"
    return (f"{formatar_valor(indicador, c['valor'])}{unidade} "
            f"(mediana dos bairros: {formatar_valor(indicador, c['mediana'])}). "
            f"Melhor que {c['piores']} e pior que {c['melhores']} dos outros {c['total'] - 1} bairros.")


def tabela_indicadores_do_bairro(ind: pd.DataFrame, notas_bairro: pd.Series, chave: str) -> pd.DataFrame:
    """Uma linha por indicador, com valor real, mediana, nota da dimensão e explicação."""
    linhas = []
    for indicador in INDICADORES:
        c = comparar_com_bairros(ind, chave, indicador)
        linhas.append({
            "Dimensão": DIMENSAO_POR_CHAVE[indicador.dimensao].nome,
            "Indicador": indicador.nome,
            "Valor no bairro": formatar_valor(indicador, c["valor"]),
            "Mediana dos bairros": formatar_valor(indicador, c["mediana"]),
            "Unidade": indicador.unidade,
            "Nota da dimensão": notas_bairro.get(indicador.dimensao),
            "Posição": "—" if c["melhores"] is None else f"{c['melhores'] + 1}º de {c['total']}",
        })
    return pd.DataFrame(linhas)


# ---------- Bases e termos ----------

BASES = {
    "156": {
        "nome": "Central 156 — chamados dos cidadãos",
        "registro": "Cada linha é um chamado aberto por um cidadão (telefone 156, aplicativo, portal ou chat).",
        "colunas": {
            "Tipo": "Solicitação, Reclamação, Denúncia, Elogio, Sugestão, Informação ou Recadastro.",
            "Orgao": "Secretaria ou órgão da prefeitura responsável por atender.",
            "DataCriacao / DataResposta": "Data de abertura e data em que a prefeitura respondeu.",
            "Assunto / Subdivisao": "Tema do chamado (ex.: Iluminação Pública) e o detalhe (ex.: lâmpada apagada).",
            "Situacao": "Aberto, Em andamento, Em avaliação, Falta de dados, Respondido ou Concluído.",
            "Logradouro / Bairro / Regional": "Onde está o problema relatado.",
            "Origem": "Canal usado: Telefone, App 156, Portal, Chat, App Curitiba.",
        },
    },
    "equipamentos": {
        "nome": "Equipamentos urbanos ativos",
        "registro": "Cada linha é um equipamento público ou de interesse público em funcionamento "
                    "(escola, unidade de saúde, módulo da Guarda, ponto de ônibus, praça esportiva...).",
        "colunas": {
            "DS_TEMA": "Tema (Educação, Saúde, Segurança, Transporte...). Só os temas de serviço à população entram na nota.",
            "DS_TP_EQUIPAMENTO": "Tipo (Escola, Unidade de Saúde, Hospital, Ponto de Parada...).",
            "NM_EQUI / NM_BAIRRO": "Nome do equipamento e bairro onde fica.",
        },
    },
    "esaude": {
        "nome": "E-Saúde — atendimentos de profissionais de nível superior",
        "registro": "Cada linha é um atendimento na rede municipal de saúde, com o cadastro do morador "
                    "atendido. Usamos só o cadastro de moradia, contando cada morador uma vez.",
        "colunas": {
            "Abastecimento": "Origem da água (rede pública, poço, cisterna, carro-pipa...).",
            "Destino Lixo": "Coletado, queimado/enterrado, céu aberto...",
            "Fezes/Urina": "Destino do esgoto (rede, fossa séptica, fossa rudimentar, céu aberto, rio...).",
            "Tipo de Habitação": "Material da moradia (alvenaria, madeira, taipa, material aproveitado...).",
            "Bairro / cod_usuario": "Bairro de residência e código anônimo do morador.",
        },
    },
}

GLOSSARIO = {
    "Chamado": "Qualquer registro na Central 156.",
    "Demanda": "Chamado do tipo Solicitação, Reclamação ou Denúncia — os que indicam algo a resolver. "
               "Elogios, sugestões, informações e recadastros não contam como demanda.",
    "Categoria": "Agrupamento de assuntos parecidos do 156 (ex.: Iluminação Pública e Pavimentação → Zeladoria urbana).",
    "Taxa por mil habitantes": "Divide a quantidade pela população em milhares. Permite comparar um bairro "
                               "de 5 mil moradores com um de 170 mil.",
    "Mediana": "O valor do meio quando os números são ordenados. Diferente da média, não é distorcida por "
               "poucos casos extremos.",
    "Chamado atrasado": f"Levou mais de {PRAZO_ATRASO_DIAS} dias para ser respondido ou continua sem resposta "
                        f"depois de {PRAZO_ATRASO_DIAS} dias.",
    "Chamado parado": f"Está Aberto, Em andamento, Em avaliação ou com Falta de dados há mais de {PRAZO_ATRASO_DIAS} dias.",
    "Nota da dimensão": "De 0 a 10. Indica a posição do bairro entre os 75: nota 8 = melhor que cerca de 80% dos bairros.",
    "Nota final": "Média ponderada das notas das dimensões, com os pesos do perfil escolhido.",
    "Regional": "Uma das 10 administrações regionais da prefeitura que agrupam os bairros.",
}
