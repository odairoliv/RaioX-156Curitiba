"""Machine Learning (classificação): este chamado vai atrasar? Treino, avaliação e simulador."""
import streamlit as st

from src import estado
from src import graficos as g
from src.componentes import aviso_modelo, controles_modelo, num, pct, status_dados_na_barra_lateral
from src.config import PRAZO_ATRASO_DIAS
from src.modelos.catalogo import MODELOS_CLASSIFICACAO
from src.modelos.classificacao import DIAS_DA_SEMANA, prever_chamado
from src.modelos.registro import ler_historico
from src.treino import treinar_atraso

ss = st.session_state

status_dados_na_barra_lateral()

st.title("⏱️ Previsão de atraso")
st.caption(f"O chamado será resolvido em até {PRAZO_ATRASO_DIAS} dias ou vai atrasar? "
           "Ajuda a prefeitura a priorizar e o cidadão a saber o que esperar.")

with st.container(border=True):
    st.markdown("**⚙️ Configurar o classificador**")
    modelo_nome, parametros = controles_modelo(MODELOS_CLASSIFICACAO, "cls")
    amostra = st.slider("Tamanho da amostra de treino", 5_000, 200_000, 40_000, step=5_000, key="cls_amostra",
                        help="Mais dados = modelo melhor, porém mais lento (principalmente o KNN).")
    rotulo = "Re-treinar" if estado.modelo("classificacao") else "Treinar"
    clicou_treinar = st.button(f"🧠 {rotulo} classificador", type="primary")

base = estado.base_atraso()
with st.expander("Como funciona? (explicação simples)"):
    st.markdown(
        f"- Usamos os chamados com pelo menos {PRAZO_ATRASO_DIAS} dias de idade ({num(len(base))} chamados). "
        f"Cada um recebe o rótulo **atrasou** (mais de {PRAZO_ATRASO_DIAS} dias sem resposta) ou **no prazo**.\n"
        "- As características são: assunto, detalhe do assunto, órgão, tipo, canal, bairro, regional, dia da "
        "semana e mês. Elas viram colunas de 0/1 (*one-hot encoding*).\n"
        "- Separamos **75% para treino e 25% para teste**. As métricas abaixo são calculadas só no teste, "
        "com chamados que o modelo nunca viu.\n"
        f"- Hoje {pct(base['atrasou'].mean() * 100)} dos chamados atrasam. Por isso comparamos a acurácia com "
        "um *chute* que sempre diz \"no prazo\"."
    )

if clicou_treinar:
    with st.spinner(f"Treinando {modelo_nome} com {num(amostra)} chamados..."):
        treinar_atraso(modelo_nome, parametros, amostra)
    st.toast("Classificador treinado e salvo!", icon="✅")

aviso_modelo("classificacao")
resultado = estado.modelo("classificacao")
if resultado is None:
    st.stop()

# ---------- Avaliação ----------
m = resultado.metricas
st.subheader("Qualidade do classificador")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Acurácia", pct(m["acuracia"] * 100), f"chute: {pct(m['acuracia_chute'] * 100)}", delta_color="off",
          help="% de acertos no teste.")
c2.metric("Precisão", pct(m["precisao"] * 100), help="Dos que o modelo disse que iam atrasar, quantos atrasaram.")
c3.metric("Recall", pct(m["recall"] * 100), help="Dos que atrasaram de verdade, quantos o modelo encontrou.")
c4.metric("F1", num(m["f1"], 3), help="Equilíbrio entre precisão e recall (0 a 1).")

esquerda, direita = st.columns(2)
esquerda.plotly_chart(g.matriz_confusao(resultado.detalhes["matriz_confusao"]), width="stretch")
importancia = resultado.detalhes["importancia"]
if importancia is not None:
    direita.plotly_chart(g.barras_simples(importancia.iloc[::-1], "O que mais pesa na decisão (%)", formato=".1f"),
                         width="stretch")
else:
    direita.info(f"O {resultado.modelo_nome} não informa a importância das características diretamente. "
                 "Experimente Árvore de Decisão, Random Forest ou Regressão Logística para ver esse gráfico.")

(no_prazo_ok, alarme_falso), (atraso_perdido, atraso_ok) = resultado.detalhes["matriz_confusao"]
total_teste = no_prazo_ok + alarme_falso + atraso_perdido + atraso_ok
with st.container(border=True):
    st.markdown("**Como ler estes resultados**")
    linhas = [
        f"O modelo foi testado em **{num(total_teste)} chamados** que ele não viu no treino. "
        f"Neles, {num(atraso_perdido + atraso_ok)} atrasaram de verdade e {num(no_prazo_ok + alarme_falso)} foram "
        "resolvidos no prazo.",
        f"- **Acurácia {pct(m['acuracia'] * 100)}:** de cada 100 chamados, o modelo acertou a resposta de "
        f"{num(m['acuracia'] * 100)}. Mas atenção: quem \"chuta\" sempre *no prazo* já acerta "
        f"{pct(m['acuracia_chute'] * 100)}, porque a maioria não atrasa. Por isso a acurácia sozinha engana.",
        f"- **Recall {pct(m['recall'] * 100)}:** de cada 100 chamados que **atrasaram de verdade**, o modelo "
        f"conseguiu apontar {num(m['recall'] * 100)}. É a métrica mais útil para a prefeitura: quantos atrasos "
        "seriam evitados se ela agisse nos alertas.",
        f"- **Precisão {pct(m['precisao'] * 100)}:** de cada 100 chamados que o modelo **disse** que iam atrasar, "
        f"{num(m['precisao'] * 100)} atrasaram mesmo; os outros são alarmes falsos.",
        f"- **F1 {num(m['f1'], 2)}:** resume precisão e recall num número só (0 = péssimo, 1 = perfeito).",
        "- **Matriz de confusão** (cada quadrado é uma quantidade de chamados do teste): "
        f"**{num(no_prazo_ok)}** no prazo e previstos no prazo ✅; **{num(atraso_ok)}** atrasados e previstos como "
        f"atraso ✅; **{num(alarme_falso)}** alarmes falsos (previsto atraso, mas foi no prazo); "
        f"**{num(atraso_perdido)}** atrasos não detectados.",
        "- **O que mais pesa:** quanto cada característica contribuiu para as decisões do modelo (soma 100%).",
        "- **Trocar o parâmetro *Balancear classes*** muda o equilíbrio: com `balanced` o modelo encontra mais "
        "atrasos (recall maior) à custa de mais alarmes falsos (precisão menor).",
    ]
    st.markdown("\n".join(linhas))

# ---------- Simulador ----------
st.subheader("Simular um chamado novo")
nomes = estado.indicadores()["bairro"]
s1, s2, s3 = st.columns(3)
orgao = s1.selectbox("Órgão", base["Orgao"].value_counts().index, key="cls_sim_orgao",
                     help="Ordenados do órgão com mais chamados para o com menos.")
do_orgao = base[base["Orgao"] == orgao]
assunto = s2.selectbox("Assunto", do_orgao["Assunto"].value_counts().index, key="cls_sim_assunto")
do_assunto = do_orgao[do_orgao["Assunto"] == assunto]
detalhe = s3.selectbox("Detalhe", do_assunto["Subdivisao"].value_counts().index, key="cls_sim_detalhe")
s4, s5, s6, s7 = st.columns(4)
nome_bairro = s4.selectbox("Bairro", sorted(nomes), index=sorted(nomes).index("Centro"), key="cls_sim_bairro")
chave = nomes[nomes == nome_bairro].index[0]
tipo = s5.selectbox("Tipo", base["Tipo"].value_counts().index, key="cls_sim_tipo")
canal = s6.selectbox("Canal", base["Origem"].value_counts().index, key="cls_sim_canal")
dia = s7.selectbox("Dia da semana", DIAS_DA_SEMANA, key="cls_sim_dia")
regional = base.loc[base["chave"] == chave, "Regional"].mode()
chamado = {"Assunto": assunto, "Subdivisao": detalhe, "Orgao": orgao, "Tipo": tipo, "Origem": canal, "chave": chave,
           "Regional": regional.iloc[0] if len(regional) else "", "dia_semana": dia,
           "mes_nome": str(estado.periodo_chamados()[1].month)}

probabilidade = prever_chamado(resultado.pipeline, chamado)
historico = do_assunto[do_assunto["Subdivisao"] == detalhe]["atrasou"]
with st.container(border=True):
    r1, r2 = st.columns([1, 2])
    r1.metric("Chance de atraso", pct(probabilidade * 100, 0))
    if probabilidade >= 0.6:
        r2.error(f"🔴 **Alta chance de atraso.** Chamados parecidos costumam passar de {PRAZO_ATRASO_DIAS} dias.")
    elif probabilidade >= 0.4:
        r2.warning("🟡 **Pode atrasar.** Vale acompanhar o andamento.")
    else:
        r2.success(f"🟢 **Deve ser resolvido em até {PRAZO_ATRASO_DIAS} dias.**")
    r2.progress(probabilidade)
    if len(historico):
        r2.caption(f"No histórico, {pct(historico.mean() * 100, 0)} dos {num(len(historico))} chamados com esse "
                   "assunto e detalhe atrasaram (em toda a cidade).")
st.caption(f"**Como ler:** a *chance de atraso* é a probabilidade, calculada pelo modelo treinado, de um chamado com "
           f"essas características levar mais de {PRAZO_ATRASO_DIAS} dias para ser respondido. Acima de 60% = alta; "
           "40% a 60% = incerto; abaixo de 40% = deve sair no prazo. Modelos diferentes podem dar chances diferentes "
           "para o mesmo chamado — compare treinando outro algoritmo.")

with st.expander("Histórico de treinos (salvos em modelos_salvos/)"):
    st.dataframe(ler_historico("classificacao"), hide_index=True, width="stretch")
