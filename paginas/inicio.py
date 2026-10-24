"""Página inicial: a pergunta do trabalho, as hipóteses e um resumo dos dados atuais."""
import streamlit as st

from src import estado
from src.componentes import data_br, num, pct, status_dados_na_barra_lateral
from src.config import PRAZO_ATRASO_DIAS

status_dados_na_barra_lateral()

st.title("⏱️ Seu chamado vai demorar?")
st.markdown(
    "Quando um morador de Curitiba pede algo à prefeitura pela **Central 156** — trocar uma lâmpada, tapar um buraco, "
    "podar uma árvore — quanto tempo leva para ser atendido? E **isso depende de onde a pessoa mora?**\n\n"
    "Este sistema analisa os chamados do 156 (dados abertos da prefeitura) para responder a essa pergunta e usa "
    "**machine learning** para prever se um chamado novo vai atrasar e quantos dias deve levar."
)

base = estado.base()
inicio, fim = estado.periodo_chamados()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Chamados na base", num(len(estado.chamados())), help=f"De {data_br(inicio)} a {data_br(fim)}.")
c2.metric("Chamados analisados", num(len(base)),
          help=f"Solicitações, reclamações e denúncias abertas há pelo menos {PRAZO_ATRASO_DIAS} dias.")
c3.metric(f"Atrasaram (+{PRAZO_ATRASO_DIAS} dias)", pct(base["atrasou"].mean() * 100))
c4.metric("Tempo típico de resposta", f"{num(base['dias_resposta'].median())} dias",
          help="Mediana: metade dos chamados é respondida em até este número de dias.")

st.subheader("As hipóteses")
st.markdown(
    f"Um chamado é considerado **atrasado** quando levou **mais de {PRAZO_ATRASO_DIAS} dias** para ser respondido "
    "(ou continua sem resposta depois disso).\n\n"
    "| | Hipótese | Como testamos |\n|---|---|---|\n"
    "| **H1** | A **região** onde a pessoa mora muda a chance de atraso | % de atraso por regional e por bairro (mapa) |\n"
    "| **H2** | O **tipo de problema** pesa mais que a região | % de atraso por órgão; comparação \"justa\" entre regiões; "
    "modelo treinado só com região × só com tipo de problema |\n"
    "| **H3** | **Denúncias** demoram mais que solicitações e reclamações | % de atraso por tipo de chamado |\n"
    "| **H4** | O **canal** usado (telefone, app, portal, chat) muda a velocidade | % de atraso por canal |"
)

st.subheader("Como usar")
st.markdown(
    "1. **Análise das hipóteses** — gráficos e a conclusão de cada hipótese, calculados com os dados atuais.\n"
    "2. **Previsão** — escolha entre *Vai atrasar?* (classificação) e *Quantos dias?* (regressão), o algoritmo e "
    "os parâmetros; treine e simule um chamado.\n"
    "3. **Enviar dados** — envie um novo arquivo do 156 (.csv, .json ou .xml), como o mês seguinte. "
    "As análises são recalculadas e os modelos podem ser re-treinados."
)
col1, col2, col3 = st.columns(3)
col1.page_link("paginas/hipoteses.py", label="Ver as hipóteses", icon="🔎")
col2.page_link("paginas/previsao.py", label="Fazer uma previsão", icon="🧠")
col3.page_link("paginas/enviar_dados.py", label="Enviar dados", icon="📂")
st.caption("Fonte: Portal de Dados Abertos de Curitiba — Central 156. Mapa dos bairros: IPPUC.")
