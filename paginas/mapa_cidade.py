"""Visão geral da cidade: mapa com a nota dos bairros, rankings e gráficos gerais."""
import pandas as pd
import streamlit as st

from src import estado
from src import graficos as g
from src.categorias import CATEGORIAS_DA_NOTA
from src.componentes import num, seletor_pesos, situacao_com_icone, status_dados_na_barra_lateral
from src.indicadores import DIMENSOES, apenas_demandas, unidade_taxa

pesos = seletor_pesos()
status_dados_na_barra_lateral()

st.title("🗺️ Mapa da cidade")
st.caption(f"Nota de 0 a 10 comparando os 75 bairros entre si. Perfil: **{st.session_state.preset_pesos}** "
           "(mude na barra lateral).")

notas = estado.notas(pesos)
ind = estado.indicadores()

# Bairros com pouquíssimos moradores aparecem no mapa, mas ficam fora dos rankings
confiaveis = notas[~notas["pouca_populacao"]]
contagem_situacao = notas["situacao"].value_counts()
melhor = confiaveis.loc[confiaveis["nota_final"].idxmax()]
pior = confiaveis.loc[confiaveis["nota_final"].idxmin()]
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("🟢 Boa", contagem_situacao.get("Boa", 0), help="Nota final de 6,5 ou mais.")
c2.metric("🟡 Atenção", contagem_situacao.get("Atenção", 0), help="Nota entre 4 e 6,5.")
c3.metric("🔴 Crítica", contagem_situacao.get("Crítica", 0), help="Nota abaixo de 4.")
c4.metric(f"Melhor: {melhor['bairro']}", num(melhor["nota_final"], 1))
c5.metric(f"Pior: {pior['bairro']}", num(pior["nota_final"], 1))

opcoes = {"Nota final": "nota_final"} | {f"Nota — {d.nome}": d.chave for d in DIMENSOES}
opcoes[f"Chamados por mês ({unidade_taxa()})"] = "taxa_total"
escolha = st.selectbox("Colorir o mapa por", list(opcoes))
coluna = opcoes[escolha]

dados_mapa = notas.copy()
dados_mapa["taxa_total"] = ind[[f"taxa_{d.chave}" for d in DIMENSOES[:5]]].sum(axis=1)
if coluna == "taxa_total":
    figura = g.mapa_bairros(dados_mapa, coluna, "Chamados/mês",
                            escala=["#f0efec", "#86b6ef", "#2a78d6", "#104281"])
else:
    figura = g.mapa_bairros(dados_mapa, coluna, "Nota", faixa=(0, 10))

mapa, ranking = st.columns([3, 2])
with mapa:
    st.plotly_chart(figura, width="stretch")
with ranking:
    aba_melhores, aba_piores = st.tabs(["Melhores", "Piores"])
    ranking_dados = dados_mapa[~dados_mapa["pouca_populacao"]]
    menos_e_melhor = coluna == "taxa_total"  # para a taxa de chamados, "melhor" é ter menos
    with aba_melhores:
        st.plotly_chart(g.barras_ranking(ranking_dados, coluna, "10 melhores", maiores=not menos_e_melhor,
                                         cor=g.AZUL), width="stretch")
    with aba_piores:
        st.plotly_chart(g.barras_ranking(ranking_dados, coluna, "10 piores", maiores=menos_e_melhor,
                                         cor="#d03b3b"), width="stretch")
    pequenos = notas.loc[notas["pouca_populacao"], "bairro"]
    if len(pequenos):
        st.caption(f"Fora do ranking por terem menos de 2 mil moradores: {', '.join(pequenos)}.")

aba_evolucao, aba_regional, aba_tipos, aba_tabela = st.tabs(
    ["📈 Evolução mensal", "🏘️ Por regional", "🧾 Tipos e canais", "📋 Tabela completa"])

demandas = apenas_demandas(estado.chamados())
with aba_evolucao:
    mensal = demandas.groupby(["mes", "categoria"], observed=True).size().rename("qtd").reset_index()
    fim = demandas["data"].max()
    if fim < fim + pd.offsets.MonthEnd(0):  # o último mês ainda não terminou
        mensal = mensal[mensal["mes"] < fim.to_period("M").start_time]
        st.caption(f"O mês de {fim:%m/%Y} está incompleto e foi omitido para não distorcer a comparação.")
    st.plotly_chart(g.linhas_por_categoria(mensal, "mes", "Chamados por mês e categoria"), width="stretch")

with aba_regional:
    por_regional = ind.groupby("regional")[[f"qtd_{c}" for c in CATEGORIAS_DA_NOTA]].sum()
    por_regional.columns = CATEGORIAS_DA_NOTA
    pop = ind.groupby("regional")["populacao"].sum() / 1000
    meses = (demandas["data"].max() - demandas["data"].min()).days / 30.44
    taxa = por_regional.div(pop, axis=0) / meses
    taxa = taxa.loc[taxa.sum(axis=1).sort_values(ascending=False).index]
    st.plotly_chart(g.barras_empilhadas(taxa, "Chamados por mês para cada mil habitantes, por regional",
                                        "Chamados / mês / mil hab."), width="stretch")

with aba_tipos:
    esquerda, direita = st.columns(2)
    todos = estado.chamados()
    esquerda.plotly_chart(g.pizza(todos["Tipo"].value_counts(), "Tipo de chamado"), width="stretch")
    direita.plotly_chart(g.pizza(todos["Origem"].value_counts(), "Canal usado"), width="stretch")

with aba_tabela:
    tabela = notas.sort_values("posicao")[["posicao", "bairro", "regional", "populacao", "nota_final", "situacao",
                                           *[d.chave for d in DIMENSOES]]].copy()
    tabela["situacao"] = tabela["situacao"].map(situacao_com_icone)
    nomes = {"posicao": "Posição", "bairro": "Bairro", "regional": "Regional", "populacao": "População",
             "nota_final": "Nota final", "situacao": "Situação"} | {d.chave: d.nome for d in DIMENSOES}
    tabela = tabela.rename(columns=nomes)
    st.dataframe(tabela, hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.1f")
                                for c in ["Nota final", *[d.nome for d in DIMENSOES]]})
    st.download_button("⬇️ Baixar tabela (CSV)", tabela.to_csv(sep=";", index=False).encode("utf-8-sig"),
                       "notas_bairros_curitiba.csv", "text/csv")
