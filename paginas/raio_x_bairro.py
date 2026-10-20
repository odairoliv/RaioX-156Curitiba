"""Raio-X de um bairro: nota, resumo em linguagem simples, comparação e detalhes."""
import pandas as pd
import streamlit as st

from src import estado
from src import graficos as g
from src.categorias import CATEGORIAS_DA_NOTA
from src.componentes import num, pct, seletor_pesos, situacao_com_icone, status_dados_na_barra_lateral
from src.indicadores import DIMENSOES, apenas_demandas, pontos_fortes_e_fracos, unidade_taxa
from src.preparacao import ITENS_SANEAMENTO

pesos = seletor_pesos()
status_dados_na_barra_lateral()

st.title("🔎 Raio-X do bairro")

notas = estado.notas(pesos)
ind = estado.indicadores()
nome_para_chave = dict(zip(notas["bairro"], notas.index))
nomes_ordenados = sorted(nome_para_chave)

escolha, comparar = st.columns([1, 2])
nome = escolha.selectbox("Bairro", nomes_ordenados, index=nomes_ordenados.index("Batel"))
comparados = comparar.multiselect("Comparar com (até 2)", [n for n in nomes_ordenados if n != nome],
                                  max_selections=2)
chave = nome_para_chave[nome]
nota, info = notas.loc[chave], ind.loc[chave]
inicio, fim = estado.periodo_chamados()
meses = max((fim - inicio).days / 30.44, 1)

# ---------- Cabeçalho com a nota ----------
with st.container(border=True):
    c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1])
    c1.metric(f"Nota de {nome}", f"{num(nota['nota_final'], 1)} / 10")
    c1.markdown(f"**{situacao_com_icone(nota['situacao'])}** · {nota['posicao']}º de 75 bairros")
    c2.metric("População (Censo 2022)", num(info["populacao"]))
    c3.metric("Demandas por mês", num(info["total_demandas"] / meses))
    dias = info["mediana_dias_resposta"]
    c4.metric("Tempo típico de resposta", f"{num(dias)} {'dia' if dias == 1 else 'dias'}",
              help="Mediana: metade dos chamados é respondida em até este número de dias.")

# ---------- Resumo em linguagem simples ----------
fortes, fracos = pontos_fortes_e_fracos(nota)
melhor_que = (notas["nota_final"] < nota["nota_final"]).mean() * 100
texto = [f"**{nome}** ({info['regional']}) tem nota **{num(nota['nota_final'], 1)}**, "
         f"melhor que cerca de **{num(melhor_que)}%** dos bairros de Curitiba no perfil *{st.session_state.preset_pesos}*."]
if fortes:
    texto.append("**Pontos fortes:** " + ", ".join(f"{n.lower()} ({num(v, 1)})" for n, v in fortes) + ".")
if fracos:
    texto.append("**Pontos de atenção:** " + ", ".join(f"{n.lower()} ({num(v, 1)})" for n, v in fracos) + ".")
texto.append(f"Cerca de **{pct(info['pct_atrasados'], 0)}** dos chamados do bairro levaram mais de 30 dias para "
             "ser respondidos.")
if info["populacao"] < 2000:
    texto.append("⚠️ *Bairro com poucos moradores: as taxas por habitante variam muito e devem ser lidas com cuidado.*")
st.info("\n\n".join(texto))

# ---------- Radar de comparação ----------
st.subheader("Notas por dimensão")
series_radar = {nome: nota, "Média da cidade": notas[[d.chave for d in DIMENSOES]].mean()}
for outro in comparados:
    series_radar[outro] = notas.loc[nome_para_chave[outro]]
radar, tabela = st.columns([3, 2])
radar.plotly_chart(g.radar(series_radar, DIMENSOES), width="stretch")
comparacao = pd.DataFrame({rotulo: [s.get(d.chave) for d in DIMENSOES] + [s.get("nota_final", None)]
                           for rotulo, s in series_radar.items()},
                          index=[d.nome for d in DIMENSOES] + ["Nota final"])
comparacao.loc["Nota final", "Média da cidade"] = notas["nota_final"].mean()
tabela.dataframe(comparacao.style.format("{:.1f}", na_rep="—"), width="stretch")
tabela.caption("Notas de 0 a 10. Quanto maior, melhor o bairro em relação aos outros.")

# ---------- Detalhes ----------
chamados_bairro = estado.chamados()[estado.chamados()["chave"] == chave]
demandas = apenas_demandas(chamados_bairro)
aba_problemas, aba_evolucao, aba_infra, aba_saneamento = st.tabs(
    ["🧾 Principais demandas", "📈 Evolução", "🏫 Infraestrutura", "🚰 Saneamento e moradia"])

with aba_problemas:
    esquerda, direita = st.columns(2)
    esquerda.plotly_chart(g.barras_simples(demandas["Assunto"].value_counts().head(10), "Assuntos mais pedidos"),
                          width="stretch")
    ruas = demandas[demandas["categoria"].isin(CATEGORIAS_DA_NOTA)]["Logradouro"].value_counts()
    ruas = ruas[ruas.index.astype(str).str.strip() != ""].head(10)
    direita.plotly_chart(g.barras_simples(ruas, "Ruas com mais demandas", cor=g.LARANJA), width="stretch")
    taxas = pd.DataFrame({
        "Bairro": [info[f"taxa_{d.chave}"] for d in DIMENSOES[:5]],
        "Média da cidade": [ind[f"taxa_{d.chave}"].median() for d in DIMENSOES[:5]],
    }, index=[d.nome for d in DIMENSOES[:5]])
    st.markdown(f"**Chamados por mês ({unidade_taxa()})** — bairro x mediana dos bairros")
    st.dataframe(taxas.style.format("{:.2f}"), width="stretch")

with aba_evolucao:
    semanal = demandas.groupby(["semana", "categoria"], observed=True).size().rename("qtd").reset_index()
    st.plotly_chart(g.linhas_por_categoria(semanal, "semana", f"Demandas por semana em {nome}"), width="stretch")
    if info["elogios"]:
        st.caption(f"💬 O bairro também registrou **{num(info['elogios'])} elogios** no período.")

with aba_infra:
    equipamentos = estado.equipamentos()
    if equipamentos is None:
        st.info("Envie a base de equipamentos urbanos para ver esta parte.")
    else:
        do_bairro = equipamentos[equipamentos["chave"] == chave]
        esquerda, direita = st.columns([2, 3])
        esquerda.metric("Equipamentos públicos no bairro", num(len(do_bairro)),
                        help="Escolas, unidades de saúde, segurança, transporte, esporte, cultura e assistência.")
        esquerda.metric(f"Por mil habitantes", num(info["taxa_infraestrutura"], 2),
                        delta=f"{num(info['taxa_infraestrutura'] - ind['taxa_infraestrutura'].median(), 2)} vs mediana",
                        delta_color="normal")
        direita.plotly_chart(g.barras_simples(do_bairro["tema"].value_counts(), "Equipamentos por tema"), width="stretch")
        with st.expander("Lista de equipamentos"):
            st.dataframe(do_bairro[["tema", "tipo_equipamento", "nome"]].sort_values(["tema", "nome"])
                         .rename(columns={"tema": "Tema", "tipo_equipamento": "Tipo", "nome": "Nome"}),
                         hide_index=True, width="stretch")

with aba_saneamento:
    if estado.esaude() is None or pd.isna(info.get("pct_saneamento")):
        st.info("Não há moradores suficientes avaliados pelo E-Saúde neste bairro para calcular o indicador.")
    else:
        itens = pd.DataFrame({
            nome: [info[f"pct_{item}"] for item in ITENS_SANEAMENTO],
            "Média da cidade": [ind[f"pct_{item}"].mean() for item in ITENS_SANEAMENTO],
        }, index=list(ITENS_SANEAMENTO))
        st.dataframe(itens.style.format("{:.1f}%"), width="stretch")
        st.caption(f"Com base em {num(info['moradores_avaliados'])} moradores atendidos pelo E-Saúde "
                   "(cadastro das unidades de saúde). Mostra a % de moradores com cada item adequado.")
