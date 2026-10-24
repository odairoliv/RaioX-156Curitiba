"""Seu chamado vai demorar? — análise e previsão de atrasos na Central 156 de Curitiba.

Para rodar:  streamlit run app.py
"""
import streamlit as st

from src import estado

st.set_page_config(page_title="Seu chamado vai demorar?", page_icon="⏱️", layout="wide")

estado.iniciar()

# O Streamlit apaga o valor dos controles quando a página que os desenha não está aberta.
# Regravar estes valores mantém as escolhas do usuário ao trocar de página.
for chave in list(st.session_state.keys()):
    if str(chave).startswith(("classificacao_", "regressao_", "sim_", "tipo_modelo")):
        st.session_state[chave] = st.session_state[chave]

paginas = [
    st.Page("paginas/inicio.py", title="Início", icon="🏠", default=True),
    st.Page("paginas/enviar_dados.py", title="Enviar dados", icon="📂"),
    st.Page("paginas/hipoteses.py", title="Análise das hipóteses", icon="🔎"),
    st.Page("paginas/previsao.py", title="Previsão (machine learning)", icon="🧠"),
    st.Page("paginas/sobre.py", title="Sobre os dados", icon="📘"),
]
st.navigation(paginas).run()
