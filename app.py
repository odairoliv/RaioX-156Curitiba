"""Raio-X dos Bairros de Curitiba — ponto de entrada do aplicativo.

Para rodar:  streamlit run app.py
"""
import streamlit as st

from src import estado

st.set_page_config(page_title="Raio-X dos Bairros de Curitiba", page_icon="🏙️", layout="wide")

estado.iniciar()

# O Streamlit apaga o valor dos controles quando a página que os desenha não está aberta.
# Regravar estes valores mantém os pesos e os parâmetros escolhidos ao trocar de página.
for chave in list(st.session_state.keys()):
    if str(chave).startswith(("peso_", "preset_pesos", "prev_", "cls_")):
        st.session_state[chave] = st.session_state[chave]

paginas = {
    "Começo": [
        st.Page("paginas/inicio.py", title="Início", icon="🏠", default=True),
        st.Page("paginas/enviar_dados.py", title="Enviar dados", icon="📂"),
    ],
    "Análise": [
        st.Page("paginas/mapa_cidade.py", title="Mapa da cidade", icon="🗺️"),
        st.Page("paginas/raio_x_bairro.py", title="Raio-X do bairro", icon="🔎"),
        st.Page("paginas/pontos_atencao.py", title="Pontos de atenção", icon="🚨"),
    ],
    "Machine Learning": [
        st.Page("paginas/previsao_tendencia.py", title="Previsão de tendência", icon="📈"),
        st.Page("paginas/previsao_atraso.py", title="Previsão de atraso", icon="⏱️"),
    ],
    "Sobre": [
        st.Page("paginas/metodologia.py", title="Metodologia e dados", icon="📘"),
    ],
}

st.navigation(paginas).run()
