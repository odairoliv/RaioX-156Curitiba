"""Upload de arquivos (.csv, .json, .xml). O tipo de base é reconhecido pelas colunas."""
import pandas as pd
import streamlit as st

from src import estado
from src.componentes import data_br, num, status_dados_na_barra_lateral
from src.config import PASTA_EXEMPLOS
from src.leitura import COLUNAS_OBRIGATORIAS, FORMATOS_ACEITOS, NOMES_TIPOS, ErroDeArquivo
from src.treino import retreinar_modelos_existentes

ss = st.session_state
status_dados_na_barra_lateral()

st.title("📂 Enviar dados")
st.markdown(
    "Envie arquivos exportados do [Portal de Dados Abertos de Curitiba](https://dadosabertos.curitiba.pr.gov.br/) "
    "ou de qualquer fonte com **as mesmas colunas**. O sistema reconhece sozinho qual é a base, "
    "limpa os dados e atualiza todos os gráficos, notas e modelos."
)

with st.expander("Quais arquivos posso enviar?"):
    for tipo, colunas in COLUNAS_OBRIGATORIAS.items():
        st.markdown(f"- **{NOMES_TIPOS[tipo]}** — colunas obrigatórias: `{'`, `'.join(colunas)}`")
    st.markdown(
        "Formatos: **.csv** (separado por `;` ou `,`, em UTF-8 ou Latin-1), **.json** (lista de registros) "
        "e **.xml** (um elemento por registro). Arquivos `.gz` também são aceitos.\n\n"
        f"Na pasta `{PASTA_EXEMPLOS.relative_to(PASTA_EXEMPLOS.parents[1])}` do projeto há exemplos prontos: "
        "outubro/2026 em CSV e a primeira semana de outubro em JSON e XML."
    )

arquivos = st.file_uploader("Arquivos", type=FORMATOS_ACEITOS, accept_multiple_files=True)
modo = st.radio(
    "Para a base do 156, o que fazer com os dados atuais?",
    ["Adicionar ao que já existe", "Substituir toda a base"],
    help="Ao adicionar, o período coberto pelo arquivo novo substitui o mesmo período da base atual "
         "(enviar o mesmo mês duas vezes não duplica nada).",
    horizontal=True,
)
retreinar = st.checkbox("Re-treinar automaticamente os modelos já treinados", value=True,
                        disabled=not ss.modelos,
                        help="Disponível depois que algum modelo foi treinado nas páginas de Machine Learning.")

if st.button("Processar arquivos", type="primary", disabled=not arquivos):
    algum_ok = False
    for arquivo in arquivos:
        with st.spinner(f"Lendo {arquivo.name}..."):
            try:
                resumo = estado.registrar_upload(arquivo.name, arquivo.getvalue(), modo.startswith("Adicionar"))
            except (ErroDeArquivo, ValueError) as erro:
                st.error(f"**{arquivo.name}**: {erro}")
                continue
        algum_ok = True
        descartadas = resumo["linhas_lidas"] - resumo["linhas_validas"]
        mensagem = (f"**{arquivo.name}** reconhecido como *{resumo['tipo']}*: "
                    f"{num(resumo['linhas_validas'])} linhas válidas de {num(resumo['linhas_lidas'])} lidas")
        if descartadas:
            mensagem += f" ({num(descartadas)} descartadas: sem data, bairro desconhecido, duplicadas ou fora do escopo)"
        st.success(mensagem + ".")
    if algum_ok and retreinar and ss.modelos:
        with st.spinner("Re-treinando modelos com os dados novos..."):
            feitos = retreinar_modelos_existentes()
        st.success("Modelos re-treinados: " + ", ".join(feitos))

st.subheader("Bases em uso")
linhas = []
for tipo, df in ss.bases.items():
    rel = ss.relatorios[tipo]
    linha = {"Base": NOMES_TIPOS[tipo], "Origem": ss.origens[tipo], "Linhas válidas": num(len(df))}
    if tipo == "156":
        linha["Período"] = f"{data_br(df['data'].min())} a {data_br(df['data'].max())}"
    linhas.append(linha)
st.dataframe(pd.DataFrame(linhas), hide_index=True, width="stretch")

if st.button("Restaurar bases padrão"):
    estado.restaurar_padrao()
    st.rerun()

if ss.historico_uploads:
    st.subheader("Arquivos enviados nesta sessão")
    historico = pd.DataFrame(ss.historico_uploads)
    colunas = ["enviado_em", "arquivo", "tipo", "linhas_lidas", "linhas_validas", "bairro_desconhecido"]
    st.dataframe(historico[[c for c in colunas if c in historico]].rename(columns={
        "enviado_em": "Enviado em", "arquivo": "Arquivo", "tipo": "Base", "linhas_lidas": "Lidas",
        "linhas_validas": "Válidas", "bairro_desconhecido": "Bairro desconhecido"}), hide_index=True, width="stretch")

with st.expander("Ver as últimas 200 linhas da base do 156 (já limpa)"):
    st.dataframe(estado.chamados().tail(200), width="stretch")
