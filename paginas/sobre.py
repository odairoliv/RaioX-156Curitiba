"""O que é a base, como cada número é calculado e quais são os limites da análise."""
import pandas as pd
import streamlit as st

from src import estado
from src.componentes import data_br, num, pct, status_dados_na_barra_lateral
from src.config import MINIMO_CHAMADOS_GRUPO, PRAZO_ATRASO_DIAS

ss = st.session_state
status_dados_na_barra_lateral()
st.title("📘 Sobre os dados")

chamados, base = estado.chamados(), estado.base()
inicio, fim = estado.periodo_chamados()

st.header("A base: Central 156 de Curitiba")
st.markdown(
    "A Central 156 é o canal da prefeitura para os moradores pedirem serviços, reclamarem ou denunciarem problemas "
    "(por telefone, aplicativo, portal ou chat). A prefeitura publica todos os chamados no "
    "[Portal de Dados Abertos de Curitiba](https://dadosabertos.curitiba.pr.gov.br/). **Cada linha da base é um chamado.**\n\n"
    f"Base atual: **{num(len(chamados))} chamados**, de {data_br(inicio)} a {data_br(fim)}, nos 75 bairros. "
    f"Origem: {ss.origem}."
)
st.dataframe(pd.DataFrame([
    ("DataCriacao / DataResposta", "Data de abertura e data da resposta da prefeitura", "Dias até a resposta; atraso"),
    ("Bairro / Regional", "Onde está o problema (10 regionais agrupam os 75 bairros)", "H1 — região"),
    ("Orgao", "Secretaria responsável pelo atendimento", "H2 — tipo de problema"),
    ("Assunto / Subdivisao", "Tema do pedido (ex.: Iluminação Pública) e o detalhe (ex.: lâmpada apagada)", "H2 — tipo de problema"),
    ("Tipo", "Solicitação, Reclamação, Denúncia, Elogio, Sugestão, Informação", "H3"),
    ("Origem", "Canal usado: Telefone, App 156, Portal, Chat, App Curitiba", "H4"),
    ("Situacao", "Aberto, Em andamento, Concluído, Respondido...", "Chamados ainda sem resposta"),
], columns=["Coluna", "Significado", "Uso no trabalho"]), hide_index=True, width="stretch")

st.header("Definições")
respondidos = base["dias_resposta"].notna()
st.markdown(
    f"- **Demanda:** chamado do tipo *Solicitação*, *Reclamação* ou *Denúncia* — os que pedem uma ação. Elogios, "
    "sugestões e pedidos de informação ficam fora da análise.\n"
    f"- **Dias até a resposta:** data da resposta − data de abertura.\n"
    f"- **Atrasado:** chamado que levou **mais de {PRAZO_ATRASO_DIAS} dias** para ser respondido, ou que continua "
    f"aberto depois de {PRAZO_ATRASO_DIAS} dias.\n"
    f"- **Base analisada:** demandas abertas há pelo menos {PRAZO_ATRASO_DIAS} dias (até "
    f"{data_br(fim - pd.Timedelta(days=PRAZO_ATRASO_DIAS))}). As mais recentes ficam de fora porque ainda não "
    f"tiveram tempo de atrasar. São **{num(len(base))} chamados**: {pct(respondidos.mean() * 100)} já respondidos e "
    f"{pct((~respondidos).mean() * 100)} ainda abertos (todos esses contam como atrasados).\n"
    "- **% de atraso de um grupo:** chamados atrasados do grupo ÷ chamados do grupo × 100. Grupos com menos de "
    f"{MINIMO_CHAMADOS_GRUPO} chamados não aparecem nos gráficos (a porcentagem ficaria instável).\n"
    "- **Mediana:** o valor do meio quando os números são ordenados; não é distorcida por poucos casos extremos."
)

st.header("Como funciona a comparação justa (H2)")
st.markdown(
    "1. Calculamos a % de atraso de cada **assunto** na cidade inteira (ex.: poda de árvore 60%, abordagem social 1%).\n"
    "2. Para cada região, o **esperado** é a média dessas porcentagens considerando os assuntos que a região pediu. "
    "Uma região que pede muita poda tem um esperado alto.\n"
    "3. **Ajustado = real ÷ esperado × média da cidade.** É a % de atraso que a região teria se pedisse a mesma "
    "mistura de assuntos que a cidade toda.\n\n"
    "Se o ajustado de uma região continua acima da média, ela atrasa mais **mesmo nos mesmos tipos de problema** — "
    "ou seja, a diferença não é explicada pelo assunto (técnica conhecida como *padronização*)."
)

st.header("Machine learning")
st.markdown(
    "- **Vai atrasar? (classificação):** aprende a separar chamados *atrasados* e *no prazo*. Algoritmos: Árvore de "
    "Decisão, KNN, Random Forest, Regressão Logística e Naive Bayes.\n"
    "- **Quantos dias? (regressão):** aprende a prever o número de dias até a resposta, usando só chamados já "
    "respondidos. Algoritmos: Regressão Linear, Ridge, Árvore de Decisão, KNN, Random Forest e Gradient Boosting.\n"
    "- **Entradas dos dois:** órgão, assunto, detalhe, tipo, canal, bairro, regional, dia da semana e mês, "
    "transformadas em colunas 0/1 (*one-hot encoding*). É possível escolher quais grupos usar — treinar só com "
    "*Onde* ou só com *O quê* é um teste direto da H2.\n"
    "- **Avaliação:** 75% dos chamados para treinar e 25% para testar. Os modelos são salvos em `modelos_salvos/` "
    "com o histórico de cada treino; ao enviar dados novos eles podem ser re-treinados."
)

st.header("Limitações")
st.markdown(
    "- O 156 registra **quem pede**. Não sabemos de problemas que ninguém registrou.\n"
    "- A data de resposta é quando a prefeitura **respondeu** o chamado, que nem sempre coincide com a solução do "
    "problema no local.\n"
    "- O bairro é o **local do problema**, que pode não ser onde a pessoa mora.\n"
    "- O prazo exato depende de fatores que não estão na base (tamanho da fila, equipes, clima, complexidade do "
    "caso). Por isso o modelo *quantos dias?* dá uma **estimativa** com erro de alguns dias.\n"
    "- Com poucos meses de dados, efeitos de estação do ano ainda não aparecem; o re-treino com arquivos novos "
    "melhora isso."
)

st.header("Qualidade dos dados carregados")
rel = ss.relatorio
c1, c2, c3, c4 = st.columns(4)
c1.metric("Linhas lidas", num(rel["linhas_lidas"]))
c2.metric("Sem data válida", num(rel["sem_data"]))
c3.metric("Bairro fora dos 75", num(rel["bairro_desconhecido"]))
c4.metric("Linhas válidas", num(rel["linhas_validas"]))

st.caption("Bibliotecas: pandas (limpeza), plotly (gráficos e mapa), scikit-learn (modelos), streamlit (interface), "
           "joblib (salvar modelos), lxml (XML). Mapa dos bairros: IPPUC (GeoJSON).")
