"""Machine learning: "vai atrasar?" (classificação) e "quantos dias?" (regressão), com simulador."""
import streamlit as st

from src import estado
from src import graficos as g
from src.analise import DIAS_DA_SEMANA
from src.bairros import tabela_bairros
from src.componentes import (aviso_modelo, controles_modelo, nomes_caracteristicas, num, pct,
                             seletor_caracteristicas, status_dados_na_barra_lateral)
from src.config import PRAZO_ATRASO_DIAS
from src.modelos.catalogo import MODELOS_CLASSIFICACAO, MODELOS_REGRESSAO
from src.modelos.classificacao import prever_chamado
from src.modelos.regressao import TOLERANCIA_DIAS, prever_dias
from src.modelos.registro import ler_historico
from src.treino import treinar

OPCOES = {
    "Vai atrasar? (classificação)": ("classificacao", MODELOS_CLASSIFICACAO),
    "Quantos dias? (regressão)": ("regressao", MODELOS_REGRESSAO),
}

status_dados_na_barra_lateral()
st.title("🧠 Previsão")
st.markdown(
    f"Dois modelos, treinados com os chamados antigos:\n"
    f"- **Vai atrasar?** — classificação: prevê *sim* ou *não* (mais de {PRAZO_ATRASO_DIAS} dias para responder).\n"
    "- **Quantos dias?** — regressão: prevê o número de dias até a resposta.\n\n"
    "Os dois usam as mesmas informações do chamado: órgão, assunto, detalhe, tipo, canal, bairro, regional, dia da "
    "semana e mês. Elas viram colunas 0/1 (*one-hot encoding*) e os dados são divididos em **75% para treinar e 25% "
    "para testar** — as métricas abaixo são calculadas só com chamados que o modelo nunca viu."
)

escolha = st.segmented_control("Modelo", list(OPCOES), default=list(OPCOES)[0], key="tipo_modelo") or list(OPCOES)[0]
tipo, catalogo = OPCOES[escolha]

with st.container(border=True):
    st.markdown("**⚙️ Configurar e treinar**")
    modelo_nome, parametros = controles_modelo(catalogo, tipo)
    caracteristicas = seletor_caracteristicas(tipo)
    amostra = st.slider("Quantos chamados usar no treino", 5_000, 200_000, 40_000, step=5_000, key=f"{tipo}_amostra",
                        help="Mais dados = modelo melhor, porém mais lento (principalmente o KNN).")
    rotulo = "Re-treinar" if estado.modelo(tipo) else "Treinar"
    if st.button(f"🧠 {rotulo} o modelo", type="primary"):
        with st.spinner(f"Treinando {modelo_nome} com {num(amostra)} chamados..."):
            treinar(tipo, modelo_nome, parametros, amostra, caracteristicas)
        st.toast("Modelo treinado e salvo!", icon="✅")

aviso_modelo(tipo)
resultado = estado.modelo(tipo)

if resultado is not None:
    m, detalhes = resultado.metricas, resultado.detalhes
    st.caption(f"Características usadas neste treino: {nomes_caracteristicas(detalhes['caracteristicas'])}.")
    st.subheader("Qualidade do modelo")

    if tipo == "classificacao":
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Acurácia", pct(m["acuracia"] * 100), f"chute: {pct(m['acuracia_chute'] * 100)}", delta_color="off")
        c2.metric("Recall", pct(m["recall"] * 100))
        c3.metric("Precisão", pct(m["precisao"] * 100))
        c4.metric("F1", num(m["f1"], 2))
        esquerda, direita = st.columns(2)
        esquerda.plotly_chart(g.matriz_confusao(detalhes["matriz_confusao"]), width="stretch")
        (no_prazo_ok, alarme_falso), (atraso_perdido, atraso_ok) = detalhes["matriz_confusao"]
        explicacao = [
            f"O modelo foi testado em **{num(detalhes['n_teste'])} chamados**: "
            f"{num(atraso_perdido + atraso_ok)} atrasaram de verdade e {num(no_prazo_ok + alarme_falso)} não.",
            f"- **Acurácia {pct(m['acuracia'] * 100)}:** de cada 100 chamados, acertou {num(m['acuracia'] * 100)}. "
            f"Quem \"chuta\" sempre *no prazo* já acerta {pct(m['acuracia_chute'] * 100)}, porque a maioria não atrasa "
            "— por isso a acurácia sozinha engana.",
            f"- **Recall {pct(m['recall'] * 100)}:** de cada 100 chamados que **atrasaram**, o modelo apontou "
            f"{num(m['recall'] * 100)}. É o mais útil para a prefeitura: quantos atrasos daria para evitar.",
            f"- **Precisão {pct(m['precisao'] * 100)}:** de cada 100 que o modelo **disse** que iam atrasar, "
            f"{num(m['precisao'] * 100)} atrasaram mesmo; o resto é alarme falso.",
            f"- **F1 {num(m['f1'], 2)}:** resume precisão e recall (0 = péssimo, 1 = perfeito). É a métrica para "
            "comparar algoritmos e grupos de características.",
            f"- **Matriz:** {num(no_prazo_ok)} acertos de *no prazo*, {num(atraso_ok)} acertos de *atraso*, "
            f"{num(alarme_falso)} alarmes falsos e {num(atraso_perdido)} atrasos não detectados.",
            "- *Balancear classes = balanced* faz o modelo encontrar mais atrasos (recall maior) em troca de mais "
            "alarmes falsos (precisão menor).",
        ]
    else:
        acerto = m[f"pct_erro_ate_{TOLERANCIA_DIAS}_dias"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Erro médio (MAE)", f"{num(m['mae'], 1)} dias", f"chute: {num(m['mae_chute'], 1)} dias",
                  delta_color="off")
        c2.metric("Erro mediano", f"{num(m['erro_mediano'], 1)} dias")
        c3.metric(f"Erro de até {TOLERANCIA_DIAS} dias", pct(acerto))
        c4.metric("R²", num(m["r2"], 2))
        esquerda, direita = st.columns(2)
        esquerda.plotly_chart(g.real_vs_previsto(detalhes["real_vs_previsto"]), width="stretch")
        explicacao = [
            f"O modelo foi testado em **{num(detalhes['n_teste'])} chamados já respondidos**. Metade deles foi "
            f"respondida em até {num(m['mediana_dias_real'])} dias.",
            f"- **Erro médio {num(m['mae'], 1)} dias:** em média, a previsão errou {num(m['mae'], 1)} dias para mais ou "
            f"para menos. O \"chute\" (prever sempre {num(m['mediana_dias_real'])} dias, sem olhar o chamado) erra "
            f"{num(m['mae_chute'], 1)} dias — o modelo é **{num((1 - m['mae'] / m['mae_chute']) * 100)}% melhor**.",
            f"- **Erro mediano {num(m['erro_mediano'], 1)} dias:** metade das previsões errou menos que isso. É menor "
            "que o erro médio porque alguns chamados demoram muito mais que o normal (meses).",
            f"- **{pct(acerto)} das previsões** erraram no máximo {TOLERANCIA_DIAS} dias.",
            f"- **R² {num(m['r2'], 2)}:** quanto das diferenças de prazo o modelo explica (1 = tudo). É baixo: o prazo "
            "exato depende de coisas que não estão na base (fila da equipe, clima, complexidade de cada caso). Por isso "
            "a previsão serve como **estimativa**, e o classificador *vai atrasar?* é mais confiável.",
            "- **Gráfico:** cada ponto é um chamado de teste; quanto mais perto da linha pontilhada, melhor a previsão.",
            "- O modelo aprende o *logaritmo* dos dias, para que os poucos chamados muito demorados não puxem todas as "
            "previsões para cima.",
        ]

    direita.plotly_chart(g.barras_simples(detalhes["importancia"], "O que mais pesa na previsão (%)"),
                         width="stretch")
    explicacao.append("- **O que mais pesa:** embaralhamos uma informação de cada vez (ex.: trocamos os bairros dos "
                      "chamados de lugar) e medimos quanto o modelo piora. Quanto maior a piora, mais o modelo depende "
                      "daquela informação (total = 100%). Se *Detalhe*, *Assunto* e *Órgão* dominam e *Bairro* e "
                      "*Regional* pesam pouco, isso apoia a H2.")
    with st.container(border=True):
        st.markdown("**Como ler estes resultados**")
        st.markdown("\n".join(explicacao))

# ---------- Simulador ----------
st.header("Simular um chamado")
modelos_prontos = {t: estado.modelo(t) for t in ("classificacao", "regressao") if estado.modelo(t)}
if not modelos_prontos:
    st.info("Treine pelo menos um dos modelos para usar o simulador.")
    st.stop()

base = estado.base()
nomes_bairros = tabela_bairros().set_index("chave")["bairro"]
s1, s2, s3 = st.columns(3)
orgao = s1.selectbox("Órgão", base["Orgao"].value_counts().index, key="sim_orgao",
                     help="Do órgão com mais chamados para o com menos.")
do_orgao = base[base["Orgao"] == orgao]
assunto = s2.selectbox("Assunto", do_orgao["Assunto"].value_counts().index, key="sim_assunto")
do_assunto = do_orgao[do_orgao["Assunto"] == assunto]
detalhe = s3.selectbox("Detalhe", do_assunto["Subdivisao"].value_counts().index, key="sim_detalhe")
s4, s5, s6, s7 = st.columns(4)
lista_bairros = sorted(nomes_bairros)
nome_bairro = s4.selectbox("Bairro", lista_bairros, index=lista_bairros.index("Centro"), key="sim_bairro")
chave = nomes_bairros[nomes_bairros == nome_bairro].index[0]
tipo_chamado = s5.selectbox("Tipo", base["Tipo"].value_counts().index, key="sim_tipo")
canal = s6.selectbox("Canal", base["Origem"].value_counts().index, key="sim_canal")
dia = s7.selectbox("Dia da semana", DIAS_DA_SEMANA, key="sim_dia")
regional = base.loc[base["chave"] == chave, "Regional"].mode()
chamado = {"Assunto": assunto, "Subdivisao": detalhe, "Orgao": orgao, "Tipo": tipo_chamado, "Origem": canal,
           "chave": chave, "Regional": regional.iloc[0] if len(regional) else "", "dia_semana": dia,
           "mes_nome": str(estado.periodo_chamados()[1].month)}

with st.container(border=True):
    r1, r2 = st.columns(2)
    if "classificacao" in modelos_prontos:
        chance = prever_chamado(modelos_prontos["classificacao"].pipeline, chamado)
        r1.metric("Chance de atrasar", pct(chance * 100, 0))
        r1.progress(chance)
        if chance >= 0.6:
            r1.error(f"🔴 Alta chance de passar de {PRAZO_ATRASO_DIAS} dias.")
        elif chance >= 0.4:
            r1.warning("🟡 Pode atrasar — vale acompanhar.")
        else:
            r1.success(f"🟢 Deve ser respondido em até {PRAZO_ATRASO_DIAS} dias.")
    else:
        r1.info("Treine o modelo *Vai atrasar?* para ver a chance de atraso.")
    if "regressao" in modelos_prontos:
        dias = prever_dias(modelos_prontos["regressao"].pipeline, chamado)
        r2.metric("Dias previstos até a resposta", f"{num(dias)} dias")
        r2.caption(f"Estimativa — o modelo erra em média {num(modelos_prontos['regressao'].metricas['mae'], 0)} dias.")
    else:
        r2.info("Treine o modelo *Quantos dias?* para ver a estimativa de prazo.")
    historico = do_assunto[do_assunto["Subdivisao"] == detalhe]
    if len(historico):
        st.caption(f"Para comparar: no histórico, {num(len(historico))} chamados com esse assunto e detalhe tiveram "
                   f"{pct(historico['atrasou'].mean() * 100, 0)} de atraso e mediana de "
                   f"{num(historico['dias_resposta'].median())} dias até a resposta (em toda a cidade).")
st.caption("Mude o **bairro** mantendo o mesmo assunto: se a previsão quase não muda, o tipo de problema pesa mais "
           "que o lugar (H2).")

with st.expander("Histórico de treinos (salvos em modelos_salvos/)"):
    st.dataframe(ler_historico(tipo), hide_index=True, width="stretch")
