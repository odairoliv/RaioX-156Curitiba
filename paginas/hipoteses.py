"""Análise das hipóteses H1–H4 com gráficos e conclusões calculadas a partir dos dados atuais."""
import pandas as pd
import streamlit as st

from src import analise, estado
from src import graficos as g
from src.componentes import num, pct, status_dados_na_barra_lateral
from src.config import MINIMO_CHAMADOS_GRUPO, PRAZO_ATRASO_DIAS

DIFERENCA_RELEVANTE = 5  # pontos percentuais: abaixo disso consideramos a diferença pequena
PLURAIS = {"Solicitação": "solicitações", "Reclamação": "reclamações", "Denúncia": "denúncias"}

status_dados_na_barra_lateral()
st.title("🔎 Análise das hipóteses")

base = estado.base()
media = base["atrasou"].mean() * 100
st.markdown(
    f"Base analisada: **{num(len(base))} chamados** (solicitações, reclamações e denúncias abertos há pelo menos "
    f"{PRAZO_ATRASO_DIAS} dias). **{pct(media)}** deles atrasaram, isto é, levaram mais de {PRAZO_ATRASO_DIAS} dias "
    f"para ser respondidos. Nos gráficos, grupos com menos de {MINIMO_CHAMADOS_GRUPO} chamados ficam de fora."
)


def conclusao(texto: str, confirmada: bool | None) -> None:
    """Caixa com a conclusão da hipótese (verde = confirmada, amarelo = em parte, cinza = não)."""
    if confirmada is True:
        st.success("✅ " + texto)
    elif confirmada is None:
        st.warning("🟡 " + texto)
    else:
        st.info("❌ " + texto)


# ======================================================================
st.header("H1 — A região onde a pessoa mora muda a chance de atraso?")
regionais = analise.taxa_de_atraso(base, "Regional")
bairros = analise.por_bairro(base)
mapa, barras = st.columns([3, 2])
mapa.plotly_chart(g.mapa_atraso(bairros), width="stretch")
mapa.caption("Cada bairro está colorido pela % dos seus chamados que atrasaram (mais escuro = mais atraso). "
             "Passe o mouse para ver os números.")
barras.plotly_chart(g.barras_pct(regionais, "% de atraso por regional", media), width="stretch")

melhor, pior = regionais.index[0], regionais.index[-1]
diferenca_h1 = analise.amplitude(regionais)
razao = regionais.loc[pior, "pct_atraso"] / regionais.loc[melhor, "pct_atraso"]
conclusao(
    f"**Nos números brutos, sim.** Na regional **{pior}**, {pct(regionais.loc[pior, 'pct_atraso'])} dos chamados "
    f"atrasam; na **{melhor}**, {pct(regionais.loc[melhor, 'pct_atraso'])} — uma diferença de "
    f"{num(diferenca_h1, 1)} pontos percentuais ({num(razao, 1)} vezes mais). Mas será que é a região ou o tipo de "
    "problema que cada região costuma pedir? Isso é testado na H2.",
    diferenca_h1 >= DIFERENCA_RELEVANTE,
)
with st.expander("Ver os bairros com mais e menos atraso"):
    colunas = {"bairro": "Bairro", "regional": "Regional", "chamados": "Chamados", "pct_atraso": "% atrasados",
               "mediana_dias": "Dias (mediana)"}
    formato = {"Chamados": "{:,.0f}", "% atrasados": "{:.1f}%", "Dias (mediana)": "{:.0f}"}
    e, d = st.columns(2)
    e.markdown("**Menos atraso**")
    e.dataframe(bairros.head(10)[list(colunas)].rename(columns=colunas).style.format(formato), hide_index=True)
    d.markdown("**Mais atraso**")
    d.dataframe(bairros.tail(10).iloc[::-1][list(colunas)].rename(columns=colunas).style.format(formato), hide_index=True)

# ======================================================================
st.header("H2 — O tipo de problema pesa mais que a região?")
orgaos = analise.taxa_de_atraso(base, "Orgao")
diferenca_orgaos = analise.amplitude(orgaos)
st.plotly_chart(g.barras_pct(orgaos, "% de atraso por órgão responsável", media, cor=g.AZUL), width="stretch")
st.markdown(
    f"Entre **órgãos** a % de atraso vai de {pct(orgaos['pct_atraso'].min())} a {pct(orgaos['pct_atraso'].max())} "
    f"(diferença de **{num(diferenca_orgaos, 1)} pontos**). Entre **regionais**, a diferença é de "
    f"**{num(diferenca_h1, 1)} pontos**."
)

st.subheader("Comparação justa entre regiões")
st.markdown(
    "Cada região pede coisas diferentes: o Centro tem muita *abordagem social* (respondida em 1 dia), outras regiões "
    "têm mais *poda de árvore* (que demora semanas). Para comparar **maçã com maçã**, calculamos a % de atraso que "
    "cada região teria **se pedisse a mesma mistura de assuntos que a cidade inteira** (barra azul). Se as barras "
    "azuis ficam parecidas, a diferença entre regiões vinha do tipo de problema; se continuam diferentes, a região "
    "realmente importa."
)
justa = analise.comparacao_justa(base, "Regional")
st.plotly_chart(g.comparacao_justa(justa, "Atraso real x atraso com a mesma mistura de assuntos", media),
                width="stretch")
diferenca_ajustada = analise.amplitude(justa, "ajustado")
explicado = (1 - diferenca_ajustada / diferenca_h1) * 100 if diferenca_h1 else 0
pior_ajustado, melhor_ajustado = justa.index[-1], justa.index[0]
conclusao(
    f"**Sim, o tipo de problema pesa mais.** A diferença entre órgãos ({num(diferenca_orgaos, 1)} pontos) é muito "
    f"maior que entre regionais ({num(diferenca_h1, 1)}). E na comparação justa a diferença entre regionais cai de "
    f"{num(diferenca_h1, 1)} para **{num(diferenca_ajustada, 1)} pontos**: cerca de **{num(explicado)}%** da "
    "diferença entre regiões é explicada pelo tipo de problema que cada uma pede.",
    diferenca_orgaos > diferenca_h1,
)
conclusao(
    f"**Mas a região não some (H1 em parte confirmada).** Mesmo comparando os mesmos tipos de problema, "
    f"**{pior_ajustado}** ({pct(justa.loc[pior_ajustado, 'ajustado'])}) continua atrasando mais que "
    f"**{melhor_ajustado}** ({pct(justa.loc[melhor_ajustado, 'ajustado'])}).",
    None if diferenca_ajustada >= DIFERENCA_RELEVANTE else False,
)

with st.expander("Ver um assunto específico em cada regional"):
    assuntos = base["Assunto"].value_counts()
    assunto = st.selectbox("Assunto", assuntos[assuntos >= 2000].index, key="sim_assunto_h2")
    do_assunto = analise.taxa_de_atraso(base[base["Assunto"] == assunto], "Regional", minimo=50)
    st.plotly_chart(g.barras_pct(do_assunto, f"{assunto}: % de atraso por regional",
                                 do_assunto["atrasados"].sum() / do_assunto["chamados"].sum() * 100), width="stretch")
    st.caption("Aqui todas as regionais estão sendo comparadas no mesmo tipo de pedido.")
st.caption("Outra forma de testar a H2 está na página **Previsão**: treine o modelo só com as características "
           "*Onde* e depois só com *O quê* e compare as métricas.")

# ======================================================================
st.header("H3 — Denúncias demoram mais que solicitações e reclamações?")
tipos = analise.taxa_de_atraso(base, "Tipo")
st.plotly_chart(g.barras_pct(tipos, "% de atraso por tipo de chamado", media), width="stretch")
denuncia = tipos.loc["Denúncia", "pct_atraso"] if "Denúncia" in tipos.index else float("nan")
outros = tipos.drop("Denúncia", errors="ignore")["pct_atraso"]
h3_ok = bool(pd.notna(denuncia) and denuncia - outros.max() >= DIFERENCA_RELEVANTE)
conclusao(
    f"**{'Sim' if h3_ok else 'Não'}.** Denúncias atrasam em {pct(denuncia)} dos casos, contra "
    + ", ".join(f"{pct(v)} em {PLURAIS.get(t, t.lower())}" for t, v in outros.items())
    + ". Denúncias normalmente exigem fiscalização no local, o que explica a demora.",
    h3_ok,
)

# ======================================================================
st.header("H4 — O canal usado muda a velocidade do atendimento?")
canais = analise.taxa_de_atraso(base, "Origem")
st.plotly_chart(g.barras_pct(canais, "% de atraso por canal de abertura", media), width="stretch")
diferenca_canais = analise.amplitude(canais)
h4_ok = diferenca_canais >= DIFERENCA_RELEVANTE
conclusao(
    f"**{'Sim' if h4_ok else 'Pouco'}.** Pelo canal **{canais.index[0]}**, {pct(canais['pct_atraso'].iloc[0])} "
    f"atrasam; pelo **{canais.index[-1]}**, {pct(canais['pct_atraso'].iloc[-1])} (diferença de "
    f"{num(diferenca_canais, 1)} pontos). Atenção: cada canal recebe assuntos diferentes, então parte da diferença "
    "pode vir do tipo de problema, como visto na H2.",
    h4_ok,
)

# ======================================================================
st.header("Contexto — o atraso está mudando ao longo do tempo?")
mensal = analise.evolucao_mensal(base)
st.plotly_chart(g.linha_evolucao(mensal, "% de chamados atrasados por mês de abertura"), width="stretch")
st.caption("Não é uma hipótese, mas ajuda a interpretar: se a % de atraso muda muito de um mês para outro, vale "
           "re-treinar os modelos quando chegarem dados novos.")

# ======================================================================
st.header("Resumo")
st.dataframe(pd.DataFrame([
    {"Hipótese": "H1 — a região muda a chance de atraso",
     "Resultado": "Confirmada em parte" if diferenca_ajustada >= DIFERENCA_RELEVANTE else "Não confirmada",
     "Evidência": f"Diferença de {num(diferenca_h1, 1)} pontos entre regionais; {num(diferenca_ajustada, 1)} "
                  "comparando os mesmos assuntos"},
    {"Hipótese": "H2 — o tipo de problema pesa mais que a região",
     "Resultado": "Confirmada" if diferenca_orgaos > diferenca_h1 else "Não confirmada",
     "Evidência": f"Diferença de {num(diferenca_orgaos, 1)} pontos entre órgãos; o tipo de problema explica "
                  f"~{num(explicado)}% da diferença entre regiões"},
    {"Hipótese": "H3 — denúncias demoram mais",
     "Resultado": "Confirmada" if h3_ok else "Não confirmada",
     "Evidência": f"Denúncias: {pct(denuncia)} de atraso; outros tipos: até {pct(outros.max())}"},
    {"Hipótese": "H4 — o canal muda a velocidade",
     "Resultado": "Confirmada" if h4_ok else "Não confirmada",
     "Evidência": f"Diferença de {num(diferenca_canais, 1)} pontos entre canais"},
]), hide_index=True, width="stretch")
