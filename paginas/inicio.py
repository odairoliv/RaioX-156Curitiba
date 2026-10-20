"""Página inicial: o que é o sistema, para quem é e um resumo dos dados atuais."""
import streamlit as st

from src import estado
from src.componentes import data_br, num, pct, status_dados_na_barra_lateral
from src.indicadores import apenas_demandas

status_dados_na_barra_lateral()

st.title("🏙️ Raio-X dos Bairros de Curitiba")
st.markdown(
    "Um retrato da **saúde de cada bairro** feito com dados abertos da Prefeitura de Curitiba: "
    "o que a população mais reclama, como a prefeitura responde, que estrutura existe no bairro e "
    "**para onde as coisas estão indo** — com previsões feitas por machine learning."
)

chamados = estado.chamados()
demandas = apenas_demandas(chamados)
inicio, fim = estado.periodo_chamados()
respondidos = demandas["dias_resposta"].dropna()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Chamados analisados", num(len(chamados)), help="Registros da Central 156 na base atual.")
col2.metric("Meses de dados", num((fim - inicio).days / 30.44, 1), help=f"De {data_br(inicio)} a {data_br(fim)}.")
col3.metric("Bairros", "75")
col4.metric("Respondidos em até 30 dias", pct((respondidos <= 30).mean() * 100),
            help="Entre os chamados que já têm data de resposta.")

st.divider()
esquerda, direita = st.columns(2)
with esquerda, st.container(border=True):
    st.subheader("👪 Para o cidadão")
    st.markdown(
        "- Vai **se mudar** ou **se hospedar** em Curitiba? Veja a nota de cada bairro.\n"
        "- Compare bairros lado a lado e descubra os pontos fortes e fracos de cada um.\n"
        "- Saiba quanto tempo a prefeitura costuma levar para resolver um pedido."
    )
    st.page_link("paginas/raio_x_bairro.py", label="Abrir o Raio-X do bairro", icon="🔎")
with direita, st.container(border=True):
    st.subheader("🏛️ Para a gestão pública")
    st.markdown(
        "- Bairros e ruas que concentram problemas, e onde as demandas estão **crescendo**.\n"
        "- Órgãos com mais atraso e chamados parados há mais de 30 dias.\n"
        "- Previsão das próximas semanas para planejar equipes."
    )
    st.page_link("paginas/pontos_atencao.py", label="Ver os pontos de atenção", icon="🚨")

st.subheader("Como usar")
st.markdown(
    "1. **Enviar dados** — o sistema já vem com a base da Central 156 (jan–set/2026). Você pode enviar "
    "arquivos novos (.csv, .json ou .xml) com as mesmas colunas, como o mês seguinte ou outro período.\n"
    "2. **Mapa da cidade** e **Raio-X do bairro** — gráficos e notas são recalculados automaticamente.\n"
    "3. **Previsão de tendência** e **Previsão de atraso** — escolha o algoritmo, ajuste os parâmetros e "
    "treine. Sempre que chegarem dados novos, use **Re-treinar**."
)
st.page_link("paginas/enviar_dados.py", label="Enviar um arquivo", icon="📂")
st.caption("Fontes: Portal de Dados Abertos de Curitiba (Central 156, Equipamentos Urbanos, E-Saúde), "
           "IPPUC (limites dos bairros) e IBGE — Censo 2022 (população).")
