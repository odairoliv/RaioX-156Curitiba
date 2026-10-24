"""Machine learning: "vai atrasar?" (classificação) e "quantos dias?" (regressão), com simulador."""
import pandas as pd
import streamlit as st

from src import estado
from src import graficos as g
from src.analise import DIAS_DA_SEMANA
from src.bairros import tabela_bairros
from src.componentes import (aviso_modelo, controles_modelo, nomes_caracteristicas, num, pct,
                             seletor_caracteristicas, status_dados_na_barra_lateral)
from src.config import PRAZO_ATRASO_DIAS
from src.modelos.catalogo import MODELOS_CLASSIFICACAO, MODELOS_REGRESSAO
from src.modelos.classificacao import metricas_no_limiar, prever_chamado, prever_varios
from src.modelos.comum import CARACTERISTICAS, MINIMO_POR_VALOR
from src.modelos.regressao import TOLERANCIA_DIAS, prever_dias, prever_dias_varios
from src.modelos.registro import ler_historico
from src.treino import treinar

OPCOES = {
    "Vai atrasar? (classificação)": ("classificacao", MODELOS_CLASSIFICACAO),
    "Quantos dias? (regressão)": ("regressao", MODELOS_REGRESSAO),
}
ss = st.session_state
nomes_bairros = tabela_bairros().set_index("chave")["bairro"]

status_dados_na_barra_lateral()
st.title("🧠 Previsão")
st.markdown(
    "O computador estuda **centenas de milhares de chamados antigos** — o que foi pedido, a quem, onde, por qual "
    "canal — e quanto tempo cada um levou. Depois tenta adivinhar o que vai acontecer com um chamado novo.\n"
    f"- **Vai atrasar?** responde *sim* ou *não* (mais de {PRAZO_ATRASO_DIAS} dias para a resposta).\n"
    "- **Quantos dias?** responde com um número de dias.\n\n"
    "Para saber se ele aprendeu de verdade, separamos **1 em cada 4 chamados** antes do treino. O modelo nunca vê "
    "esses chamados; no final ele tenta adivinhá-los e comparamos com o que realmente aconteceu. **Todos os "
    "resultados abaixo vêm desse teste.**"
)

escolha = st.segmented_control("Modelo", list(OPCOES), default=list(OPCOES)[0], key="tipo_modelo") or list(OPCOES)[0]
tipo, catalogo = OPCOES[escolha]

with st.container(border=True):
    st.markdown("**⚙️ Configurar e treinar**")
    modelo_nome, parametros = controles_modelo(catalogo, tipo)
    caracteristicas = seletor_caracteristicas(tipo)
    amostra = st.slider("Quantos chamados usar (75% para treinar, 25% para testar)", 5_000, 200_000, 40_000,
                        step=5_000, key=f"{tipo}_amostra",
                        help="Mais dados = modelo melhor, porém mais lento (principalmente o KNN).")
    rotulo = "Re-treinar" if estado.modelo(tipo) else "Treinar"
    if st.button(f"🧠 {rotulo} o modelo", type="primary"):
        with st.spinner(f"Treinando {modelo_nome} com {num(amostra)} chamados..."):
            treinar(tipo, modelo_nome, parametros, amostra, caracteristicas)
        st.toast("Modelo treinado e salvo!", icon="✅")

aviso_modelo(tipo)
resultado = estado.modelo(tipo)


def _bairro(chave: str) -> str:
    return nomes_bairros.get(chave, chave)


# ======================================================================
# Resultados — Vai atrasar?
# ======================================================================
if resultado is not None and tipo == "classificacao":
    detalhes = resultado.detalhes
    st.caption(f"Este treino usou: {nomes_caracteristicas(detalhes['caracteristicas'])}.")

    st.subheader("🎚️ Quando dar o alerta?")
    st.markdown(
        "Para cada chamado o modelo calcula uma **chance de atraso** (de 0% a 100%). Você decide a partir de qual "
        "chance ele deve levantar a bandeira 🚩. É como o alarme de um carro: **sensível demais** dispara à toa "
        "(muitos alarmes falsos); **pouco sensível** deixa passar problemas de verdade."
    )
    limiar = st.slider("Alertar quando a chance de atraso for pelo menos", 5, 95, 50, step=5, format="%d%%",
                       key="classificacao_limiar") / 100
    y_teste, chance = pd.Series(detalhes["y_teste"]), detalhes["chance_teste"]
    m = metricas_no_limiar(y_teste, chance, limiar)
    (no_prazo_ok, alarme), (perdido, atraso_ok) = m["matriz"]
    total = no_prazo_ok + alarme + perdido + atraso_ok
    por_100 = lambda v: v / total * 100  # noqa: E731
    atrasam, alertas = por_100(perdido + atraso_ok), por_100(alarme + atraso_ok)

    with st.container(border=True):
        st.markdown(f"#### Em 100 chamados novos, com o alerta em {limiar:.0%}:")
        st.markdown("\n".join([
            f"- Cerca de **{num(atrasam)}** vão atrasar de verdade.",
            f"- O modelo levanta a bandeira 🚩 em **{num(alertas)}** chamados:",
            f"    - **{num(por_100(atraso_ok))}** realmente atrasam ✅ (alertas certos);",
            f"    - **{num(por_100(alarme))}** acabam saindo no prazo ⚠️ (alarmes falsos).",
            f"- Dos {num(atrasam)} que atrasam, o modelo **encontrou {num(por_100(atraso_ok))}** e "
            f"**deixou passar {num(por_100(perdido))}** ❌.",
        ]))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Atrasos encontrados", pct(m["recall"] * 100),
              help="Recall. De cada 100 chamados que atrasaram de verdade, quantos o modelo marcou com 🚩.")
    c2.metric("Alertas certos", pct(m["precisao"] * 100),
              help="Precisão. De cada 100 bandeiras 🚩 que o modelo levantou, quantas eram atrasos de verdade.")
    c3.metric("Acertos no geral", pct(m["acuracia"] * 100), f"chute: {pct(m['acuracia_chute'] * 100)}",
              delta_color="off", help="Acurácia. Quem 'chuta' sempre 'no prazo' já acerta a maioria, porque a "
                                      "maioria dos chamados não atrasa. Por isso esta métrica sozinha engana.")
    c4.metric("Nota geral (F1)", num(m["f1"], 2),
              help="Junta 'atrasos encontrados' e 'alertas certos' num número de 0 a 1. Use para comparar algoritmos.")

    esquerda, direita = st.columns(2)
    esquerda.plotly_chart(g.curva_limiar(y_teste.to_numpy(), chance, limiar), width="stretch")
    direita.plotly_chart(g.matriz_confusao(m["matriz"]), width="stretch")
    st.caption("**Gráfico da esquerda:** baixando o limiar (para a esquerda) o modelo encontra mais atrasos, mas uma "
               "parte maior dos alertas é falsa. Não existe limiar que melhore as duas coisas ao mesmo tempo — a "
               "escolha depende do uso. Para **acompanhar** chamados (custo baixo), vale um limiar baixo; para "
               "**mandar uma equipe extra** (custo alto), um limiar alto.")

    st.subheader("Exemplos reais do teste")
    exemplos = detalhes["exemplos"].copy()
    exemplos["alerta"] = exemplos["chance"] >= limiar
    exemplos["resultado"] = [
        "✅ acertou" if bool(a) == bool(r) else ("⚠️ alarme falso" if a else "❌ não detectou")
        for a, r in zip(exemplos["alerta"], exemplos["atrasou"])]
    # 2 exemplos de cada situação, quando houver
    escolhidos = pd.concat([exemplos[exemplos["resultado"] == r].head(2)
                            for r in ["✅ acertou", "⚠️ alarme falso", "❌ não detectou"]]
                           + [exemplos[(exemplos["resultado"] == "✅ acertou") & ~exemplos["alerta"]].head(2)])
    st.dataframe(pd.DataFrame({
        "Assunto": escolhidos["Assunto"], "Detalhe": escolhidos["Subdivisao"],
        "Bairro": escolhidos["chave"].map(_bairro), "Chance calculada": escolhidos["chance"] * 100,
        "Modelo disse": escolhidos["alerta"].map({True: "🚩 vai atrasar", False: "no prazo"}),
        "O que aconteceu": [("ainda sem resposta" if pd.isna(d) else f"{num(d)} dias") + (" (atrasou)" if a else "")
                            for d, a in zip(escolhidos["dias_resposta"], escolhidos["atrasou"])],
        "Resultado": escolhidos["resultado"],
    }), hide_index=True, width="stretch",
        column_config={"Chance calculada": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100)})
    st.caption("Chamados de verdade que o modelo não viu no treino. Mude o limiar acima e veja os resultados mudarem.")

# ======================================================================
# Resultados — Quantos dias?
# ======================================================================
if resultado is not None and tipo == "regressao":
    m, detalhes = resultado.metricas, resultado.detalhes
    st.caption(f"Este treino usou: {nomes_caracteristicas(detalhes['caracteristicas'])}.")
    acerto = m[f"pct_erro_ate_{TOLERANCIA_DIAS}_dias"]
    melhora = (1 - m["mae"] / m["mae_chute"]) * 100

    with st.container(border=True):
        st.markdown("#### Em 100 chamados já respondidos:")
        st.markdown("\n".join([
            f"- Em **{num(acerto)}** a previsão errou **no máximo {TOLERANCIA_DIAS} dias** (para mais ou para menos).",
            f"- Em metade deles o erro foi de até **{num(m['erro_mediano'])} dias**.",
            f"- O erro médio foi de **{num(m['mae'], 1)} dias**. Ele é maior que o erro típico porque alguns chamados "
            "levam meses e o modelo não tem como prever esses casos raros.",
            f"- Para comparar: quem \"chuta\" sempre {num(m['mediana_dias_real'])} dias (o prazo mais comum) erra em "
            f"média {num(m['mae_chute'], 1)} dias. O modelo erra **{num(melhora)}% menos** que o chute.",
        ]))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Previsões com erro de até {TOLERANCIA_DIAS} dias", pct(acerto))
    c2.metric("Erro típico (mediana)", f"{num(m['erro_mediano'], 1)} dias",
              help="Metade das previsões errou menos que isso.")
    c3.metric("Erro médio (MAE)", f"{num(m['mae'], 1)} dias", f"chute: {num(m['mae_chute'], 1)} dias",
              delta_color="off", help="Média do erro, para mais ou para menos, em todos os chamados de teste.")
    c4.metric("R²", num(m["r2"], 2),
              help="Quanto das diferenças de prazo o modelo consegue explicar (1 = tudo, 0 = nada). É baixo porque o "
                   "prazo exato depende de coisas que não estão na base: fila da equipe, clima, complexidade do caso.")

    grafico, _ = st.columns([3, 1])
    grafico.plotly_chart(g.real_vs_previsto(detalhes["real_vs_previsto"]), width="stretch")
    grafico.caption("Cada ponto é um chamado de teste. Pontos sobre a linha pontilhada = previsão perfeita; acima "
                     "dela o modelo previu mais dias do que levou, abaixo previu menos.")

    st.subheader("Exemplos reais do teste")
    exemplos = detalhes["exemplos"].head(8)
    st.dataframe(pd.DataFrame({
        "Assunto": exemplos["Assunto"], "Detalhe": exemplos["Subdivisao"], "Bairro": exemplos["chave"].map(_bairro),
        "Levou (dias)": exemplos["dias_resposta"].astype(int), "Modelo previu (dias)": exemplos["previsto"].round(0).astype(int),
        "Erro (dias)": (exemplos["previsto"] - exemplos["dias_resposta"]).abs().round(0).astype(int),
    }), hide_index=True, width="stretch")
    st.caption("Chamados de verdade que o modelo não viu no treino. A previsão serve como **estimativa** do prazo; "
               "para decidir se um chamado vai atrasar, o modelo *Vai atrasar?* é mais confiável.")

# ---------- O que mais pesa (vale para os dois) ----------
if resultado is not None:
    st.subheader("O que mais pesa na previsão")
    col_grafico, col_texto = st.columns([3, 2])
    col_grafico.plotly_chart(g.barras_simples(resultado.detalhes["importancia"], "Importância de cada informação (%)"),
                             width="stretch")
    col_texto.markdown(
        "Para medir, **embaralhamos uma informação de cada vez** — por exemplo, trocamos os bairros dos chamados "
        "de lugar, como se cada um fosse de um bairro qualquer — e vemos **quanto o modelo piora**.\n\n"
        "- Se embaralhar o bairro quase não muda nada, o modelo **não depende** do bairro.\n"
        "- Se embaralhar o assunto estraga tudo, o assunto é **essencial**.\n\n"
        "Quando *Detalhe*, *Assunto* e *Órgão* somam muito mais que *Bairro* e *Regional*, isso apoia a **H2**: o "
        "tipo de problema pesa mais que o lugar."
    )

# ======================================================================
# Simulador
# ======================================================================
st.header("Simular um chamado")
prontos = {t: estado.modelo(t) for t in ("classificacao", "regressao") if estado.modelo(t)}
if not prontos:
    st.info("Treine pelo menos um dos modelos para usar o simulador.")
    st.stop()

usadas = set().union(*(r.detalhes["caracteristicas"] for r in prontos.values()))
ignoradas = [CARACTERISTICAS[c] for c in CARACTERISTICAS if c not in usadas and c != "Regional"]
if ignoradas:
    st.info(f"🔒 Os modelos treinados **não usam**: {', '.join(n.lower() for n in ignoradas)}. Esses campos ficam "
            "travados porque mudá-los não alteraria o resultado. Para usá-los, treine de novo com esses grupos.")

base = estado.base()
regional_do_bairro = base.groupby("chave")["Regional"].agg(lambda s: s.mode().iloc[0])
travado = lambda coluna: coluna not in usadas  # noqa: E731

s1, s2, s3 = st.columns(3)
orgao = s1.selectbox("Órgão", base["Orgao"].value_counts().index, key="sim_orgao", disabled=travado("Orgao"),
                     help="Do órgão com mais chamados para o com menos.")
do_orgao = base[base["Orgao"] == orgao]
assunto = s2.selectbox("Assunto", do_orgao["Assunto"].value_counts().index, key="sim_assunto",
                       disabled=travado("Assunto"))
do_assunto = do_orgao[do_orgao["Assunto"] == assunto]
contagem_detalhes = do_assunto["Subdivisao"].value_counts()
detalhe = s3.selectbox("Detalhe", contagem_detalhes.index, key="sim_detalhe", disabled=travado("Subdivisao"),
                       format_func=lambda d: f"{d} ({num(contagem_detalhes[d])} chamados)")
s4, s5, s6, s7 = st.columns(4)
lista_bairros = sorted(nomes_bairros)
nome_bairro = s4.selectbox("Bairro", lista_bairros, index=lista_bairros.index("Centro"), key="sim_bairro",
                           disabled=travado("chave"))
chave = nomes_bairros[nomes_bairros == nome_bairro].index[0]
tipo_chamado = s5.selectbox("Tipo", base["Tipo"].value_counts().index, key="sim_tipo", disabled=travado("Tipo"))
canal = s6.selectbox("Canal", base["Origem"].value_counts().index, key="sim_canal", disabled=travado("Origem"))
dia = s7.selectbox("Dia da semana", DIAS_DA_SEMANA, key="sim_dia", disabled=travado("dia_semana"))
chamado = {"Assunto": assunto, "Subdivisao": detalhe, "Orgao": orgao, "Tipo": tipo_chamado, "Origem": canal,
           "chave": chave, "Regional": regional_do_bairro.get(chave, ""), "dia_semana": dia,
           "mes_nome": str(estado.periodo_chamados()[1].month)}

limiar = ss.get("classificacao_limiar", 50) / 100
with st.container(border=True):
    r1, r2 = st.columns(2)
    if "classificacao" in prontos:
        chance = prever_chamado(prontos["classificacao"].pipeline, chamado)
        r1.metric("Chance de atrasar", pct(chance * 100, 0))
        r1.progress(chance)
        if chance >= limiar:
            r1.error(f"🚩 Acima do limiar de {limiar:.0%}: o modelo **alerta** que deve passar de {PRAZO_ATRASO_DIAS} dias.")
        else:
            r1.success(f"Abaixo do limiar de {limiar:.0%}: o modelo espera resposta em até {PRAZO_ATRASO_DIAS} dias.")
    else:
        r1.info("Treine o modelo *Vai atrasar?* para ver a chance de atraso.")
    if "regressao" in prontos:
        dias = prever_dias(prontos["regressao"].pipeline, chamado)
        r2.metric("Dias previstos até a resposta", f"{num(dias)} dias")
        r2.caption(f"Estimativa — em {pct(prontos['regressao'].metricas[f'pct_erro_ate_{TOLERANCIA_DIAS}_dias'], 0)} "
                   f"dos testes o erro foi de até {TOLERANCIA_DIAS} dias.")
    else:
        r2.info("Treine o modelo *Quantos dias?* para ver a estimativa de prazo.")
    historico = do_assunto[do_assunto["Subdivisao"] == detalhe]
    if len(historico):
        st.caption(f"Para comparar: no histórico, {num(len(historico))} chamados com esse assunto e detalhe tiveram "
                   f"{pct(historico['atrasou'].mean() * 100, 0)} de atraso e mediana de "
                   f"{num(historico['dias_resposta'].median())} dias até a resposta (em toda a cidade).")

# ---------- O mesmo chamado em todos os bairros ----------
st.subheader("E se fosse em outro bairro?")
if "chave" not in usadas:
    st.markdown("Os modelos treinados **não usam o bairro**, então o resultado seria igual em qualquer bairro.")
else:
    todos = pd.DataFrame([{**chamado, "chave": c, "Regional": regional_do_bairro.get(c, "")} for c in nomes_bairros.index])
    comparacao = pd.DataFrame({"Bairro": nomes_bairros.values})
    if "classificacao" in prontos:
        comparacao["Chance de atrasar (%)"] = prever_varios(prontos["classificacao"].pipeline, todos) * 100
    if "regressao" in prontos:
        comparacao["Dias previstos"] = prever_dias_varios(prontos["regressao"].pipeline, todos)
    coluna = comparacao.columns[1]
    menor, maior = comparacao.loc[comparacao[coluna].idxmin()], comparacao.loc[comparacao[coluna].idxmax()]
    unidade = "%" if "Chance" in coluna else " dias"
    st.markdown(
        f"O **mesmo pedido** ({assunto.lower()} — {detalhe.lower()}) feito em cada um dos 75 bairros: o resultado "
        f"vai de **{num(menor[coluna])}{unidade}** ({menor['Bairro']}) a **{num(maior[coluna])}{unidade}** "
        f"({maior['Bairro']}). Quanto menor essa diferença, menos o lugar importa para este tipo de pedido (H2)."
    )
    with st.expander("Ver todos os bairros"):
        st.dataframe(comparacao.sort_values(coluna), hide_index=True, width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.0f") for c in comparacao.columns[1:]})

with st.expander("Por que às vezes mudar um campo não muda o resultado?"):
    st.markdown(
        "1. **O modelo pode não usar aquela informação.** Se ele foi treinado sem o grupo *Onde*, por exemplo, o "
        "bairro é ignorado — por isso o campo aparece travado 🔒.\n"
        "2. **Árvores fazem poucas perguntas.** Uma Árvore de Decisão com profundidade 8 faz no máximo 8 perguntas "
        "seguidas (\"é da Secretaria do Urbanismo?\", \"é poda?\"...). Se nenhuma for sobre o bairro ou sobre aquele "
        "detalhe, mudá-los não altera nada. Gradient Boosting e Regressão Logística usam mais informações.\n"
        f"3. **Detalhes raros são agrupados.** Detalhes com menos de {MINIMO_POR_VALOR} chamados no treino viram "
        "\"outros\" e o modelo trata todos eles igual. O número entre parênteses no campo *Detalhe* mostra quantos "
        "chamados existem de cada um.\n"
        "4. **A informação realmente pesa pouco.** Pelos resultados da H2, o bairro muda pouco a previsão quando o "
        "tipo de problema já é conhecido — o que é, em si, uma resposta à pergunta do trabalho."
    )

with st.expander("Histórico de treinos (salvos em modelos_salvos/)"):
    st.dataframe(ler_historico(tipo), hide_index=True, width="stretch")
