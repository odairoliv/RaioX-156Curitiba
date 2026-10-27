"""Upload de arquivos da Central 156 (.csv, .json, .xml)."""
import pandas as pd
import streamlit as st

from src import estado
from src.componentes import data_br, num, status_dados_na_barra_lateral
from src.config import PASTA_EXEMPLOS
from src.leitura import COLUNAS_OBRIGATORIAS, COLUNAS_OPCIONAIS, EFEITO_SEM_COLUNA, FORMATOS_ACEITOS, ErroDeArquivo
from src.treino import retreinar_modelos_existentes

ss = st.session_state
status_dados_na_barra_lateral()

st.title("📂 Enviar dados")
st.markdown(
    "Envie arquivos da **Central 156** exportados do [Portal de Dados Abertos de Curitiba]"
    "(https://dadosabertos.curitiba.pr.gov.br/) — por exemplo, o mês seguinte ou outro período. "
    "O sistema confere as colunas, limpa os dados e recalcula as análises. Os modelos podem ser re-treinados "
    "automaticamente com os dados novos."
)

with st.expander("Quais arquivos posso enviar?"):
    st.markdown(
        f"- **Colunas obrigatórias:** `{'`, `'.join(COLUNAS_OBRIGATORIAS)}`\n"
        f"- **Colunas opcionais:** `{'`, `'.join(COLUNAS_OPCIONAIS)}` — se faltarem, o sistema mostra o que muda e "
        "pergunta se você quer continuar. A regional, se faltar, é preenchida pelo bairro.\n"
        "- **Colunas a mais** são ignoradas, e a ordem das colunas não importa.\n"
        "- **Formatos:** .csv (separado por `;` ou `,`, em UTF-8 ou Latin-1), .json (lista de registros) e .xml "
        "(um elemento por chamado). Arquivos `.gz` também são aceitos.\n\n"
        f"Na pasta `{PASTA_EXEMPLOS.relative_to(PASTA_EXEMPLOS.parents[1])}` do projeto há exemplos prontos: "
        "outubro/2026 em CSV e a primeira semana de outubro em JSON e XML."
    )

arquivos = st.file_uploader("Arquivos", type=FORMATOS_ACEITOS, accept_multiple_files=True)
modo = st.radio(
    "O que fazer com os dados atuais?", ["Adicionar ao que já existe", "Substituir toda a base"], horizontal=True,
    help="Ao adicionar, o período coberto pelo arquivo novo substitui o mesmo período da base atual "
         "(enviar o mesmo mês duas vezes não duplica nada).",
)
retreinar = st.checkbox("Re-treinar automaticamente os modelos já treinados", value=True, disabled=not ss.modelos,
                        help="Disponível depois que algum modelo foi treinado na página Previsão.")

def aplicar(analisados: list[dict]) -> None:
    """Junta os arquivos já analisados à base, mostra o resumo e re-treina se pedido."""
    algum_ok = False
    for analisado in analisados:
        try:
            resumo = estado.aplicar_upload(analisado, modo.startswith("Adicionar"))
        except ValueError as erro:
            st.error(f"**{analisado['arquivo']}**: {erro}")
            continue
        algum_ok = True
        descartadas = resumo["linhas_lidas"] - resumo["linhas_validas"]
        mensagem = (f"**{analisado['arquivo']}**: {num(resumo['linhas_validas'])} chamados válidos de "
                    f"{num(resumo['linhas_lidas'])} lidos, de {data_br(resumo['periodo'][0])} a "
                    f"{data_br(resumo['periodo'][1])}")
        if descartadas:
            mensagem += f" ({num(descartadas)} descartados por falta de data ou bairro fora dos 75 oficiais)"
        if resumo["regional_deduzida"]:
            mensagem += f". Regional preenchida pelo bairro em {num(resumo['regional_deduzida'])} chamados"
        st.success(mensagem + ".")
    if algum_ok and retreinar and ss.modelos:
        with st.spinner("Re-treinando os modelos com os dados novos..."):
            feitos = retreinar_modelos_existentes()
        st.success("Modelos re-treinados: " + ", ".join(feitos))


# Passo 1: ler e conferir os arquivos, sem mexer na base atual
if st.button("Processar arquivos", type="primary", disabled=not arquivos):
    ss.pop("uploads_pendentes", None)
    analisados = []
    for arquivo in arquivos:
        with st.spinner(f"Lendo {arquivo.name}..."):
            try:
                analisados.append(estado.analisar_upload(arquivo.name, arquivo.getvalue()))
            except ErroDeArquivo as erro:
                st.error(f"**{arquivo.name}**: {erro}")
    if any(a["faltando"] for a in analisados):
        ss.uploads_pendentes = analisados  # espera a decisão do usuário (abaixo)
    elif analisados:
        aplicar(analisados)

# Passo 2 (só quando falta coluna opcional): o usuário decide se continua
if ss.get("uploads_pendentes"):
    aviso = st.empty()  # some da tela assim que o usuário decide
    with aviso.container(border=True):
        st.warning("⚠️ **Alguns arquivos não têm todas as colunas opcionais.** Eles podem ser usados, mas veja o que "
                   "muda antes de continuar. Nada foi carregado ainda.")
        for analisado in ss.uploads_pendentes:
            if analisado["faltando"]:
                st.markdown(f"**{analisado['arquivo']}** — falta: `{'`, `'.join(analisado['faltando'])}`")
                st.markdown("\n".join(f"- {EFEITO_SEM_COLUNA[c]}" for c in analisado["faltando"]))
            else:
                st.markdown(f"**{analisado['arquivo']}** — completo ✅")
        coluna_prosseguir, coluna_cancelar = st.columns(2)
        prosseguir = coluna_prosseguir.button("Prosseguir assim mesmo", type="primary", width="stretch")
        cancelar = coluna_cancelar.button("Cancelar", width="stretch")
    if prosseguir:
        aviso.empty()
        aplicar(ss.pop("uploads_pendentes"))
    elif cancelar:
        aviso.empty()
        ss.pop("uploads_pendentes")
        st.info("Envio cancelado. A base continua como estava.")

st.subheader("Base em uso")
inicio, fim = estado.periodo_chamados()
c1, c2, c3 = st.columns(3)
c1.metric("Chamados", num(len(estado.chamados())))
c2.metric("Período", f"{data_br(inicio)} a {data_br(fim)}")
c3.metric("Analisados (30+ dias)", num(len(estado.base())))
st.caption(f"Origem: {ss.origem}")
if st.button("Restaurar a base padrão"):
    estado.restaurar_padrao()
    st.rerun()

if ss.historico_uploads:
    st.subheader("Arquivos enviados nesta sessão")
    historico = pd.DataFrame(ss.historico_uploads)
    colunas = {"enviado_em": "Enviado em", "arquivo": "Arquivo", "linhas_lidas": "Lidas", "linhas_validas": "Válidas",
               "bairro_desconhecido": "Bairro fora dos 75", "colunas_faltando": "Colunas opcionais faltando",
               "regional_deduzida": "Regional preenchida pelo bairro"}
    st.dataframe(historico[[c for c in colunas if c in historico]].rename(columns=colunas),
                 hide_index=True, width="stretch")

with st.expander("Ver as últimas 200 linhas da base (já limpa)"):
    st.dataframe(estado.chamados().tail(200), width="stretch")
