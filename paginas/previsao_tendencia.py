"""Machine Learning (regressão): previsão de chamados das próximas semanas e da nota projetada."""
import pandas as pd
import streamlit as st

from src import categorias as cat
from src import estado
from src import graficos as g
from src.componentes import aviso_modelo, controles_modelo, num, seletor_pesos, status_dados_na_barra_lateral
from src.indicadores import notas_projetadas
from src.modelos.catalogo import MODELOS_REGRESSAO
from src.modelos.previsao import prever_futuro, tendencia_por_bairro
from src.modelos.registro import ler_historico
from src.treino import treinar_tendencia

ss = st.session_state

pesos = seletor_pesos()
status_dados_na_barra_lateral()

st.title("📈 Previsão de tendência")
st.caption("Quantos chamados cada bairro deve ter nas próximas semanas — e como isso mexe na nota.")

with st.container(border=True):
    st.markdown("**⚙️ Configurar o modelo (regressão)**")
    modelo_nome, parametros = controles_modelo(MODELOS_REGRESSAO, "prev")
    rotulo = "Re-treinar" if estado.modelo("previsao") else "Treinar"
    clicou_treinar = st.button(f"🧠 {rotulo} modelo", type="primary")

with st.expander("Como funciona? (explicação simples)"):
    st.markdown(
        "1. Contamos os chamados de cada **bairro** e **categoria**, semana a semana.\n"
        "2. O modelo aprende a relação entre **as 4 semanas anteriores** e **a semana seguinte** "
        "(além do tamanho típico daquela série).\n"
        "3. **Avaliação justa:** escondemos as últimas semanas, o modelo prevê sem ver a resposta e comparamos "
        "com o que aconteceu de verdade. Também comparamos com uma **regra simples** (repetir a média das "
        "últimas 4 semanas): um bom modelo precisa ser melhor que ela.\n"
        "4. Depois da avaliação o modelo é re-treinado com **todas** as semanas e prevê o futuro passo a passo "
        "(a previsão de uma semana vira entrada da próxima)."
    )

if clicou_treinar:
    with st.spinner(f"Treinando {modelo_nome}..."):
        treinar_tendencia(modelo_nome, parametros)
    st.toast("Modelo treinado e salvo!", icon="✅")

aviso_modelo("previsao")
resultado = estado.modelo("previsao")
if resultado is None:
    st.stop()

# ---------- Avaliação ----------
m = resultado.metricas
st.subheader("Qualidade do modelo")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Erro médio (MAE)", num(m["mae"], 2), help="Em média, quantos chamados o modelo erra por bairro, categoria e semana.")
c2.metric("Erro da regra simples", num(m["mae_regra_simples"], 2))
c3.metric("Ganho sobre a regra simples", f"{num(m['ganho_sobre_regra_simples_%'], 1)}%",
          delta="melhor" if m["ganho_sobre_regra_simples_%"] > 0 else "pior",
          delta_color="normal" if m["ganho_sobre_regra_simples_%"] > 0 else "inverse")
c4.metric("R²", num(m["r2"], 3), help="Quanto da variação o modelo explica (1 = perfeito).")

esquerda, direita = st.columns([3, 2])
esquerda.plotly_chart(g.previsao_vs_real(resultado.detalhes["teste_cidade"],
                                         f"Teste nas últimas {resultado.detalhes['semanas_teste']} semanas (cidade inteira)"),
                      width="stretch")
direita.plotly_chart(g.barras_simples(resultado.detalhes["erro_por_categoria"], "Erro médio por categoria",
                                      formato=".2f"), width="stretch")

with st.container(border=True):
    st.markdown("**Como ler estes resultados**")
    media = resultado.detalhes.get("media_real_teste")
    proporcao = f" — cerca de {num(m['mae'] / media * 100)}% de uma semana típica" if media else ""
    ganho = m["ganho_sobre_regra_simples_%"]
    unidade = ("- **Unidade de previsão:** quantos chamados um bairro terá numa categoria numa semana "
               "(ex.: *trânsito no Batel na semana de 15/09*).")
    if media:
        unidade += f" No teste, uma combinação bairro × categoria teve em média **{num(media, 1)} chamados por semana**."
    st.markdown("\n".join([
        unidade,
        f"- **Erro médio (MAE) = {num(m['mae'], 2)}:** em média, a previsão errou {num(m['mae'], 1)} chamados "
        f"para mais ou para menos em cada bairro × categoria × semana{proporcao}.",
        f"- **Regra simples = {num(m['mae_regra_simples'], 2)}:** é o erro de quem só repete a média das últimas 4 "
        f"semanas. O modelo é **{num(abs(ganho), 1)}% {'melhor' if ganho > 0 else 'pior'}** que essa regra. Um ganho "
        "pequeno significa que a demanda é estável e o passado recente já explica quase tudo — resultado esperado "
        "com poucos meses de dados.",
        f"- **R² = {num(m['r2'], 2)}:** o modelo explica {num(m['r2'] * 100)}% das diferenças entre os valores reais. "
        "É alto em boa parte porque bairros grandes sempre têm mais chamados que os pequenos.",
        "- **Gráfico da esquerda:** soma da cidade nas semanas de teste — linha preta é o real, azul é o modelo e "
        "pontilhada é a regra simples (quedas bruscas costumam ser feriados, como 7 de setembro).",
        "- **Gráfico da direita:** erro médio por categoria; categorias com mais chamados têm erros maiores em "
        "números absolutos.",
    ]))

# ---------- Previsão ----------
st.subheader("Previsão das próximas semanas")
serie = estado.serie()
f1, f2, f3 = st.columns([1, 2, 2])
horizonte = f1.slider("Semanas à frente", 2, 13, 8, key="prev_horizonte")
nomes = estado.indicadores()["bairro"]
escopo = f2.selectbox("Onde", ["Cidade inteira", *sorted(nomes)], key="prev_escopo")
categorias = f3.multiselect("Categorias", cat.TODAS_CATEGORIAS, default=cat.CATEGORIAS_DA_NOTA, key="prev_categorias")
categorias = categorias or cat.CATEGORIAS_DA_NOTA


@st.cache_data(show_spinner="Calculando a previsão...", max_entries=10)
def calcular_futuro(id_modelo: str, versao: str, semanas: int, _pipeline, _serie) -> pd.DataFrame:
    return prever_futuro(_pipeline, _serie, semanas)


futuro = calcular_futuro(resultado.treinado_em.isoformat(), ss.versao_dados, horizonte, resultado.pipeline, serie)
filtro_hist = serie["categoria"].isin(categorias)
filtro_fut = futuro["categoria"].isin(categorias)
if escopo != "Cidade inteira":
    chave = nomes[nomes == escopo].index[0]
    filtro_hist &= serie["chave"] == chave
    filtro_fut &= futuro["chave"] == chave
historico = serie[filtro_hist].groupby("semana")["qtd"].sum()
previsto = futuro[filtro_fut].groupby("semana")["qtd"].sum()
st.plotly_chart(g.historico_e_previsao(historico, previsto, f"{escopo}: chamados por semana"), width="stretch")

variacao = previsto.mean() / historico.tail(4).mean() - 1 if historico.tail(4).mean() else 0
# Cada semana conta no mês onde cai a maior parte dos seus dias (a quinta-feira)
mensal = previsto.groupby((previsto.index + pd.Timedelta(days=3)).to_period("M")).sum()
p1, p2 = st.columns(2)
p1.metric("Média semanal prevista", num(previsto.mean()), f"{num(variacao * 100, 1)}% vs últimas 4 semanas",
          delta_color="inverse")
p2.markdown("**Previsão por mês** (meses parciais incluem só as semanas previstas)")
p2.dataframe(pd.DataFrame({"Mês": mensal.index.strftime("%m/%Y"), "Chamados previstos": mensal.round().astype(int).values}),
             hide_index=True, width="stretch")
st.caption("**Como ler:** a linha azul é o número real de demandas por semana (só as categorias escolhidas); a linha "
           "tracejada laranja é a previsão. *Média semanal prevista* compara a média das semanas previstas com a média "
           "das 4 últimas semanas reais (seta para baixo = queda de demanda, o que é bom). Previsões longas tendem a "
           "ficar \"retas\": o modelo volta para o nível típico de cada bairro quando não há novidade nos dados.")

# ---------- Tendência por bairro ----------
st.subheader("Onde a demanda deve crescer")
tendencia = tendencia_por_bairro(serie, futuro, categorias).join(nomes)
tendencia["variacao_%"] = tendencia["variacao_%"].clip(-50, 50)
mapa, tabela = st.columns([3, 2])
mapa.plotly_chart(g.mapa_bairros(tendencia, "variacao_%", "Variação %", escala=g.ESCALA_VARIACAO, faixa=(-30, 30),
                                 formato="+.1f", altura=480), width="stretch")
tabela.markdown("**Maiores altas previstas** (bairros com pelo menos 5 chamados por semana)")
tabela.dataframe(tendencia[tendencia["media_recente"] >= 5].sort_values("variacao_%", ascending=False).head(10)
                 [["bairro", "media_recente", "media_prevista", "variacao_%"]]
                 .rename(columns={"bairro": "Bairro", "media_recente": "Média recente/sem.",
                                  "media_prevista": "Média prevista/sem.", "variacao_%": "Variação %"})
                 .style.format({"Média recente/sem.": "{:.1f}", "Média prevista/sem.": "{:.1f}", "Variação %": "{:+.1f}%"}),
                 hide_index=True, width="stretch")
st.caption("**Como ler:** a cor é a variação entre a média semanal prevista e a média das 4 últimas semanas reais, "
           "em cada bairro. Vermelho = demanda deve crescer (atenção); azul = deve cair. Valores limitados a ±50%.")

# ---------- Nota projetada ----------
st.subheader("Nota projetada dos bairros")
st.caption("**Como ler:** recalculamos a nota de cada bairro trocando as taxas de chamados atuais pelas taxas "
           "previstas e comparando com a situação de HOJE dos 75 bairros. Infraestrutura, saneamento e atendimento "
           "ficam como estão. *Diferença* positiva = o bairro deve melhorar de posição; negativa = deve piorar.")
contagens = futuro.pivot_table(index="chave", columns="categoria", values="qtd", aggfunc="sum")
projetadas = notas_projetadas(estado.indicadores(), contagens, horizonte * 7 / 30.44, pesos)
atuais = estado.notas(pesos)
comparacao = pd.DataFrame({"Bairro": atuais["bairro"], "Nota atual": atuais["nota_final"],
                           "Nota projetada": projetadas["nota_final"]})
comparacao["Diferença"] = comparacao["Nota projetada"] - comparacao["Nota atual"]
esq, dir_ = st.columns(2)
formato = {"Nota atual": "{:.1f}", "Nota projetada": "{:.1f}", "Diferença": "{:+.2f}"}
esq.markdown("**Bairros que devem piorar**")
esq.dataframe(comparacao.sort_values("Diferença").head(10).style.format(formato), hide_index=True, width="stretch")
dir_.markdown("**Bairros que devem melhorar**")
dir_.dataframe(comparacao.sort_values("Diferença", ascending=False).head(10).style.format(formato),
               hide_index=True, width="stretch")

with st.expander("Histórico de treinos (salvos em modelos_salvos/)"):
    st.dataframe(ler_historico("previsao"), hide_index=True, width="stretch")
