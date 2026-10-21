"""Explica os dados, os indicadores, como a nota é calculada (com exemplo real) e os limites da análise."""
import pandas as pd
import streamlit as st

from src import estado
from src.bairros import fonte_populacao, tabela_bairros, usa_populacao
from src.categorias import ASSUNTOS_POR_CATEGORIA
from src.componentes import data_br, num, seletor_pesos, status_dados_na_barra_lateral
from src.dicionario import BASES, GLOSSARIO, INDICADORES, comparar_com_bairros, formatar_valor, indicadores_da_dimensao
from src.indicadores import DIMENSAO_POR_CHAVE, DIMENSOES, PRESETS_PESOS, POPULACAO_MINIMA_CONFIAVEL, unidade_taxa
from src.leitura import NOMES_TIPOS

ss = st.session_state
pesos = seletor_pesos()
status_dados_na_barra_lateral()

st.title("📘 Metodologia e dados")
st.markdown(
    "O sistema responde a três perguntas para cada bairro de Curitiba:\n"
    "1. **Quanto a população pede ajuda à prefeitura**, e sobre o quê? (chamados da Central 156)\n"
    "2. **Como a prefeitura responde?** (tempo de resposta e atrasos)\n"
    "3. **Que estrutura o bairro tem?** (equipamentos públicos e saneamento das moradias)\n\n"
    "Cada resposta vira um **indicador** (um número com unidade, ex.: *8,7 chamados por mês por mil moradores*). "
    "Os indicadores são transformados em **notas de 0 a 10** e combinados numa **nota final**."
)

aba_dados, aba_indicadores, aba_nota, aba_limites = st.tabs(
    ["🗂️ Os dados", "📏 Indicadores", "🧮 Como a nota é calculada", "⚠️ Limitações e qualidade"])

# ======================================================================
with aba_dados:
    st.subheader("Bases usadas")
    for tipo, info in BASES.items():
        df = ss.bases.get(tipo)
        with st.container(border=True):
            st.markdown(f"#### {info['nome']}")
            st.markdown(f"{info['registro']}  \n**Origem dos dados no app:** {ss.origens.get(tipo, '—')}")
            if df is not None:
                resumo = f"**{num(len(df))} registros válidos**"
                if tipo == "156":
                    resumo += f", de {data_br(df['data'].min())} a {data_br(df['data'].max())}"
                st.markdown(resumo + ". Fonte: [Portal de Dados Abertos de Curitiba](https://dadosabertos.curitiba.pr.gov.br/).")
            st.dataframe(pd.DataFrame(info["colunas"].items(), columns=["Coluna", "Significado"]),
                         hide_index=True, width="stretch")

    st.subheader("População por bairro")
    bairros = tabela_bairros()
    with st.container(border=True):
        if usa_populacao():
            st.markdown(
                f"**Sim, o sistema usa a população de cada bairro** — não só os chamados. Ela serve para "
                f"transformar quantidades em **taxas por mil habitantes** e assim comparar bairros de tamanhos "
                f"muito diferentes (ex.: Riviera tem {num(bairros.set_index('chave').loc['RIVIERA', 'populacao'])} "
                f"moradores; a Cidade Industrial, "
                f"{num(bairros.set_index('chave').loc['CIDADE INDUSTRIAL DE CURITIBA', 'populacao'])}).\n\n"
                f"- **Fonte atual:** {fonte_populacao()}\n"
                f"- **Total:** {num(bairros['populacao'].sum())} habitantes nos 75 bairros.\n"
                "- **Arquivo:** `data/populacao_bairros.csv` (colunas `bairro;populacao`)."
            )
        else:
            st.warning("O arquivo de população não está completo: as taxas estão sendo calculadas **por km²**.")
        st.markdown(
            "**Como atualizar a população**\n"
            "1. **Pelo app:** na página *Enviar dados*, envie um CSV com as colunas `bairro;populacao` e os 75 "
            "bairros (ex.: estimativas mais recentes do IPPUC). O arquivo do projeto é substituído e todas as taxas "
            "e notas são recalculadas.\n"
            "2. **Direto do IBGE:** rode `python scripts/atualizar_populacao.py`. O script consulta a API do IBGE "
            "(tabela SIDRA 9923, Censo 2022, nível Bairro). Quando sair um censo novo por bairro, informe a tabela e o "
            "ano: `python scripts/atualizar_populacao.py <tabela> <ano>`.\n"
            "3. **À mão:** edite `data/populacao_bairros.csv`. A primeira linha (`# Fonte: ...`) descreve a origem."
        )
        with st.expander("Ver a população de cada bairro"):
            st.dataframe(bairros[["bairro", "regional", "populacao", "area_km2"]].sort_values("populacao", ascending=False)
                         .rename(columns={"bairro": "Bairro", "regional": "Regional", "populacao": "População",
                                          "area_km2": "Área (km²)"}),
                         hide_index=True, width="stretch",
                         column_config={"População": st.column_config.NumberColumn(format="%d"),
                                        "Área (km²)": st.column_config.NumberColumn(format="%.2f")})

    st.subheader("Mapa dos bairros")
    st.markdown("Limites e área dos 75 bairros oficiais do **IPPUC**, em GeoJSON "
                "([geo-curitiba](https://github.com/gustavokira/geo-curitiba)) — `data/geo/bairros.geo.json`.")

    st.subheader("Glossário")
    st.dataframe(pd.DataFrame(GLOSSARIO.items(), columns=["Termo", "Significado"]), hide_index=True, width="stretch")

# ======================================================================
with aba_indicadores:
    st.markdown(
        f"Cada **dimensão** da nota é medida por um ou mais **indicadores**. As taxas são calculadas "
        f"**{unidade_taxa()}** e **por mês**, para que bairros grandes e pequenos — e períodos de tamanhos "
        "diferentes — possam ser comparados."
    )
    linhas = [{
        "Dimensão": DIMENSAO_POR_CHAVE[i.dimensao].nome, "Indicador": i.nome, "O que mede": i.o_que_mede,
        "Como é calculado": i.como_calcula, "Unidade": i.unidade, "Fonte": i.fonte,
        "Melhor quando": "maior" if i.maior_melhor else "menor",
    } for i in INDICADORES]
    st.dataframe(pd.DataFrame(linhas), hide_index=True, width="stretch",
                 column_config={"O que mede": st.column_config.TextColumn(width="large"),
                                "Como é calculado": st.column_config.TextColumn(width="large")})
    st.markdown(
        "**Por que \"menor é melhor\" nas dimensões de chamados?** Mais chamados por morador significa mais "
        "problemas relatados no bairro (buracos, lixo, insegurança...). Já em *Infraestrutura* e *Saneamento*, "
        "quanto maior, melhor.\n\n"
        "**Atendimento da prefeitura** usa dois indicadores (tempo típico de resposta e % de atrasados); a nota da "
        "dimensão é a média das duas notas."
    )
    st.subheader("Quais assuntos do 156 entram em cada categoria?")
    for categoria, assuntos in ASSUNTOS_POR_CATEGORIA.items():
        with st.expander(categoria):
            st.markdown(", ".join(assuntos))
    st.caption("Os demais assuntos (IPTU, cartão transporte, escolas etc.) ficam em **Outros serviços** e não entram "
               "na nota. Pedidos de coleta (podas, entulho) são um serviço de rotina e também ficam fora da nota.")

# ======================================================================
with aba_nota:
    st.markdown(
        "**Passo 1 — calcular o indicador.** Ex.: chamados de trânsito por mês para cada mil moradores.\n\n"
        "**Passo 2 — transformar em nota (0 a 10) comparando os 75 bairros.** A nota é a fração de bairros que "
        "estão *piores* que ele, vezes 10. Quem está melhor que todos tira ~10; quem está pior que todos tira ~0; "
        "quem está no meio tira ~5. Empates contam como meio ponto.\n\n"
        "**Passo 3 — nota final.** Média ponderada das 8 notas, com os pesos do perfil escolhido na barra lateral. "
        "Se um bairro não tem dados numa dimensão, o peso dela é redistribuído entre as outras.\n\n"
        "**Passo 4 — situação.** 🟢 **Boa**: 6,5 ou mais · 🟡 **Atenção**: de 4 a 6,5 · 🔴 **Crítica**: abaixo de 4."
    )

    st.subheader("Exemplo com números reais")
    ind = estado.indicadores()
    notas = estado.notas(pesos)
    nomes = sorted(notas["bairro"])
    bairro = st.selectbox("Bairro do exemplo", nomes, index=nomes.index("Água Verde"))
    chave = notas.index[notas["bairro"] == bairro][0]
    dimensao = DIMENSAO_POR_CHAVE["transito"]
    indicador = indicadores_da_dimensao("transito")[0]
    c = comparar_com_bairros(ind, chave, indicador)
    st.markdown(
        f"- **Passo 1:** {bairro} registrou {num(ind.loc[chave, 'qtd_' + dimensao.nome])} chamados de "
        f"{dimensao.nome.lower()} no período, tem {num(ind.loc[chave, 'populacao'])} moradores → "
        f"**{formatar_valor(indicador, c['valor'])} {indicador.unidade}** "
        f"(mediana dos bairros: {formatar_valor(indicador, c['mediana'])}).\n"
        f"- **Passo 2:** {c['piores']} dos outros {c['total'] - 1} bairros têm taxa maior (estão piores) → "
        f"nota de {dimensao.nome.lower()} = **{num(notas.loc[chave, 'transito'], 1)}**."
    )
    calculo = pd.DataFrame({
        "Dimensão": [d.nome for d in DIMENSOES],
        "Nota": [notas.loc[chave, d.chave] for d in DIMENSOES],
        "Peso": [pesos[d.chave] for d in DIMENSOES],
    })
    calculo["Nota × peso"] = calculo["Nota"] * calculo["Peso"]
    validos = calculo["Nota"].notna()
    st.markdown(f"- **Passo 3:** soma de (nota × peso) ÷ soma dos pesos = "
                f"{num(calculo.loc[validos, 'Nota × peso'].sum(), 2)} ÷ {num(calculo.loc[validos, 'Peso'].sum(), 1)} = "
                f"**{num(notas.loc[chave, 'nota_final'], 2)}** → situação **{notas.loc[chave, 'situacao']}**.")
    st.dataframe(calculo, hide_index=True, width="stretch",
                 column_config={"Nota": st.column_config.NumberColumn(format="%.2f"),
                                "Nota × peso": st.column_config.NumberColumn(format="%.2f")})

    st.subheader("Pesos de cada perfil")
    st.dataframe(pd.DataFrame(PRESETS_PESOS).rename(index={d.chave: d.nome for d in DIMENSOES}), width="stretch")
    st.caption("Peso 2 = a dimensão conta o dobro de uma com peso 1. Peso 0 = a dimensão é ignorada.")

# ======================================================================
with aba_limites:
    st.subheader("Limitações (importante para interpretar)")
    st.markdown(
        "- **O 156 mede quem reclama, não só onde há problema.** Bairros onde a população usa mais o 156 podem "
        "parecer piores; bairros com menos acesso ao serviço podem parecer melhores do que são.\n"
        "- **Bairros centrais recebem muita gente de fora** (trabalho, comércio, lazer), mas a taxa é dividida só "
        "pelos moradores. Por isso Centro, Batel e vizinhos têm muitos chamados por morador.\n"
        f"- **Bairros com menos de {num(POPULACAO_MINIMA_CONFIAVEL)} moradores** (Riviera e Lamenha Pequena) têm "
        "taxas instáveis e ficam fora dos rankings.\n"
        "- **As notas são relativas.** Quando todos os bairros têm valores parecidos (ex.: saneamento, quase todos "
        "acima de 95%), uma diferença pequena muda bastante a nota. Por isso o Raio-X mostra também o valor real.\n"
        "- O **E-Saúde** cobre só moradores atendidos na rede municipal, não toda a população.\n"
        "- A **população é do Censo 2022**; bairros que cresceram desde então podem ter taxas um pouco superestimadas.\n"
        "- As **previsões** usam poucos meses de histórico; efeitos sazonais (ex.: verão e dengue) só serão "
        "aprendidos com mais dados — por isso o re-treino com arquivos novos é importante."
    )
    st.subheader("Qualidade dos dados carregados")
    rotulos = {"linhas_lidas": "Linhas lidas", "linhas_validas": "Linhas válidas", "sem_data": "Sem data",
               "bairro_desconhecido": "Bairro fora dos 75", "fora_dos_temas": "Fora dos temas de serviço",
               "moradores_unicos": "Moradores únicos"}
    for tipo, relatorio in ss.relatorios.items():
        with st.container(border=True):
            st.markdown(f"**{NOMES_TIPOS[tipo]}** — {ss.origens[tipo]}")
            partes = {k: v for k, v in relatorio.items() if isinstance(v, (int, float))}
            for coluna, (chave_rel, valor) in zip(st.columns(len(partes)), partes.items()):
                coluna.metric(rotulos.get(chave_rel, chave_rel), num(valor))

    st.subheader("Bibliotecas usadas")
    st.markdown("**pandas** (leitura e limpeza), **plotly** (gráficos e mapas interativos), **scikit-learn** "
                "(modelos), **streamlit** (interface web), **joblib** (salvar modelos), **lxml** (leitura de XML).")
