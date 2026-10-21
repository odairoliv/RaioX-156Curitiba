"""Pontos de atenção para a gestão pública: o que está crescendo, o que está parado e onde agir."""
import pandas as pd
import streamlit as st

from src import categorias as cat
from src import estado
from src import graficos as g
from src.componentes import num, pct, seletor_pesos, status_dados_na_barra_lateral
from src.config import PRAZO_ATRASO_DIAS
from src.indicadores import apenas_demandas, unidade_taxa

ACOES_SUGERIDAS = {
    cat.ZELADORIA: "Programar mutirão de manutenção (pavimento, iluminação, calçadas, drenagem).",
    cat.TRANSITO: "Reforçar a fiscalização de trânsito e revisar a sinalização.",
    cat.LIMPEZA: "Intensificar limpeza, poda de árvores e fiscalização de terrenos.",
    cat.SEGURANCA: "Ampliar a abordagem social e as rondas da Guarda Municipal.",
    cat.SAUDE: "Ações de vigilância em saúde (ex.: combate à dengue, vigilância sanitária).",
}
JANELA_DIAS = 28  # compara as últimas 4 semanas com as 8 anteriores

pesos = seletor_pesos()
status_dados_na_barra_lateral()
with st.sidebar:
    st.subheader("Sensibilidade dos alertas")
    alta_minima = st.slider("Alta mínima para alertar (%)", 10, 100, 50, step=5)
    volume_minimo = st.slider("Mínimo de chamados nas últimas 4 semanas", 5, 100, 30, step=5)

st.title("🚨 Pontos de atenção")
st.caption("Painel para a gestão pública. Os alertas são recalculados a cada novo arquivo enviado.")

chamados = estado.chamados()
demandas = apenas_demandas(chamados)
demandas = demandas[demandas["categoria"].isin(cat.CATEGORIAS_DA_NOTA)]
notas = estado.notas(pesos)
nomes = notas["bairro"]
data_ref = chamados["data"].max()

# ---------- 1. Crescimento recente ----------
inicio_recente = data_ref - pd.Timedelta(days=JANELA_DIAS - 1)
inicio_anterior = inicio_recente - pd.Timedelta(days=2 * JANELA_DIAS)
recente = demandas[demandas["data"] >= inicio_recente].groupby(["chave", "categoria"], observed=True).size()
anterior = (demandas[(demandas["data"] >= inicio_anterior) & (demandas["data"] < inicio_recente)]
            .groupby(["chave", "categoria"], observed=True).size() / 2)
crescimento = pd.DataFrame({"recente": recente, "anterior": anterior}).fillna(0)
crescimento["variacao"] = (crescimento["recente"] / crescimento["anterior"].where(crescimento["anterior"] > 0) - 1) * 100
crescimento = crescimento[(crescimento["recente"] >= volume_minimo) & (crescimento["variacao"] >= alta_minima)]
crescimento = crescimento.sort_values("variacao", ascending=False).reset_index()

# ---------- 2. Chamados parados ----------
limite = data_ref - pd.Timedelta(days=PRAZO_ATRASO_DIAS)
parados = demandas[demandas["pendente"] & (demandas["data"] <= limite)]

# ---------- 3. Lista unificada de alertas ----------
alertas = []
for _, linha in crescimento.iterrows():
    alertas.append({
        "Prioridade": "🔴 Alta" if linha["variacao"] >= 2 * alta_minima else "🟡 Média",
        "Bairro": nomes[linha["chave"]], "Alerta": f"Alta em {linha['categoria'].lower()}",
        "Detalhe": f"{num(linha['recente'])} chamados nas últimas 4 semanas (+{num(linha['variacao'])}% sobre a média anterior)",
        "Ação sugerida": ACOES_SUGERIDAS[linha["categoria"]],
    })
for chave, linha in notas[notas["situacao"] == "Crítica"].sort_values("nota_final").iterrows():
    alertas.append({
        "Prioridade": "🔴 Alta" if linha["nota_final"] < 3 else "🟡 Média", "Bairro": linha["bairro"],
        "Alerta": "Nota geral crítica", "Detalhe": f"Nota {num(linha['nota_final'], 1)} ({linha['posicao']}º de 75)",
        "Ação sugerida": "Ver o Raio-X do bairro para identificar as dimensões mais fracas.",
    })
parados_bairro = parados.groupby("chave", observed=True).size().sort_values(ascending=False)
for chave, qtd in parados_bairro.head(5).items():
    alertas.append({
        "Prioridade": "🟡 Média", "Bairro": nomes[chave], "Alerta": "Chamados parados",
        "Detalhe": f"{num(qtd)} chamados abertos há mais de {PRAZO_ATRASO_DIAS} dias",
        "Ação sugerida": "Revisar a fila de atendimento dos órgãos responsáveis.",
    })
alertas = pd.DataFrame(alertas)

c1, c2, c3 = st.columns(3)
c1.metric("Alertas de crescimento", len(crescimento))
c2.metric("Bairros em situação crítica", int((notas["situacao"] == "Crítica").sum()))
c3.metric(f"Chamados parados há +{PRAZO_ATRASO_DIAS} dias", num(len(parados)))

with st.expander("Como cada alerta é definido", expanded=True):
    st.markdown("\n".join([
        f"- **Alta em uma categoria:** compara as demandas das **últimas 4 semanas** "
        f"({inicio_recente:%d/%m} a {data_ref:%d/%m/%Y}) com a média de 4 semanas das **8 semanas anteriores** "
        f"({inicio_anterior:%d/%m} a {inicio_recente - pd.Timedelta(days=1):%d/%m}). Vira alerta quando a alta é de "
        f"pelo menos **{alta_minima}%** e o bairro teve pelo menos **{volume_minimo} chamados** nas últimas 4 semanas "
        f"(ajuste na barra lateral). Prioridade **alta** quando a alta passa de {2 * alta_minima}%.",
        "- **Nota geral crítica:** nota final abaixo de 4 no perfil escolhido. Prioridade alta abaixo de 3.",
        f"- **Chamados parados:** demandas ainda *Abertas*, *Em andamento*, *Em avaliação* ou com *Falta de dados* "
        f"criadas há mais de {PRAZO_ATRASO_DIAS} dias (até {limite:%d/%m/%Y}). Lista os 5 bairros com mais casos.",
        "- **Ação sugerida:** recomendação padrão para o tipo de problema — serve de ponto de partida para a equipe.",
    ]))

st.subheader("Lista de alertas")
if alertas.empty:
    st.success("Nenhum alerta com os critérios atuais.")
else:
    alertas = alertas.sort_values("Prioridade", key=lambda s: s.str.contains("Alta"), ascending=False)
    st.dataframe(alertas, hide_index=True, width="stretch")
    st.download_button("⬇️ Baixar relatório de alertas (CSV)", alertas.to_csv(sep=";", index=False).encode("utf-8-sig"),
                       f"alertas_bairros_{data_ref:%Y%m%d}.csv", "text/csv")

aba_calor, aba_orgaos, aba_ruas, aba_parados = st.tabs(
    ["🌡️ Mapa de calor", "🏢 Órgãos", "🛣️ Ruas críticas", "⏳ Chamados parados"])

with aba_calor:
    ind = estado.indicadores()
    taxas = ind[[f"taxa_{c}" for c in ["zeladoria", "transito", "limpeza", "seguranca", "saude"]]]
    taxas.columns = cat.CATEGORIAS_DA_NOTA
    taxas.index = ind["bairro"]
    n = st.slider("Quantos bairros mostrar", 10, 75, 25, step=5)
    topo = taxas.loc[taxas.sum(axis=1).sort_values(ascending=False).index[:n]]
    st.plotly_chart(g.mapa_de_calor(topo, f"Chamados por mês ({unidade_taxa()}) — bairros com maior demanda",
                                    "Chamados/mês"), width="stretch")
    st.caption("Cada linha é um bairro e cada coluna uma categoria. O número é quantas demandas daquela categoria o "
               "bairro registra por mês para cada mil moradores; quanto mais escuro, maior. Mostra os bairros com "
               "maior soma das 5 categorias — útil para ver *qual* problema puxa a demanda de cada bairro.")

with aba_orgaos:
    respondidos = demandas.dropna(subset=["dias_resposta"])
    maduros = demandas[demandas["data"] <= limite]
    atrasou = (maduros["dias_resposta"] > PRAZO_ATRASO_DIAS) | (maduros["pendente"] & maduros["dias_resposta"].isna())
    orgaos = pd.DataFrame({
        "Chamados": demandas.groupby("Orgao", observed=True).size(),
        "Mediana de dias": respondidos.groupby("Orgao", observed=True)["dias_resposta"].median(),
        "% com atraso": atrasou.groupby(maduros["Orgao"], observed=True).mean() * 100,
        "Parados": parados.groupby("Orgao", observed=True).size(),
    }).fillna({"Parados": 0})
    orgaos = orgaos[orgaos["Chamados"] >= 200].sort_values("% com atraso", ascending=False)
    st.plotly_chart(g.barras_simples(orgaos["% com atraso"].head(10), f"Órgãos com mais chamados acima de {PRAZO_ATRASO_DIAS} dias (%)",
                                     cor="#d03b3b", formato=".1f"), width="stretch")
    st.dataframe(orgaos.style.format({"Chamados": "{:,.0f}", "Mediana de dias": "{:.0f}", "% com atraso": "{:.1f}%",
                                      "Parados": "{:,.0f}"}), width="stretch")
    st.caption(f"Órgãos com pelo menos 200 demandas. **Chamados:** demandas no período. **Mediana de dias:** tempo "
               f"típico até a resposta. **% com atraso:** entre as demandas com mais de {PRAZO_ATRASO_DIAS} dias de "
               f"idade, quantas passaram de {PRAZO_ATRASO_DIAS} dias sem resposta. **Parados:** ainda abertos após "
               f"{PRAZO_ATRASO_DIAS} dias.")

with aba_ruas:
    categoria = st.selectbox("Categoria", cat.CATEGORIAS_DA_NOTA)
    ruas = (demandas[demandas["categoria"] == categoria]
            .assign(bairro=lambda d: d["chave"].astype(str).map(nomes))
            .groupby(["Logradouro", "bairro"], observed=True).size().sort_values(ascending=False))
    ruas = ruas[ruas.index.get_level_values(0).astype(str).str.strip() != ""].head(15)
    ruas.index = [f"{rua} ({bairro})" for rua, bairro in ruas.index]
    st.plotly_chart(g.barras_simples(ruas, f"Ruas com mais demandas de {categoria.lower()}",
                                     cor=g.CORES_CATEGORIAS[categoria]), width="stretch")
    st.caption("Quantidade total de demandas da categoria registradas em cada rua (com o bairro entre parênteses) no "
               "período inteiro. Ruas longas que cruzam vários bairros aparecem separadas por bairro.")

with aba_parados:
    st.markdown(f"Chamados ainda **abertos ou em andamento** criados até {limite:%d/%m/%Y}.")
    por_assunto = parados.groupby("Assunto", observed=True).size().sort_values(ascending=False).head(10)
    st.plotly_chart(g.barras_simples(por_assunto, "Assuntos com mais chamados parados", cor=g.LARANJA), width="stretch")
    st.caption(f"Total: {num(len(parados))} chamados ({pct(len(parados) / max(len(demandas), 1) * 100)} das demandas).")
