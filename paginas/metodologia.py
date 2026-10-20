"""Explica de onde vêm os dados, como a nota é calculada e quais são os limites da análise."""
import pandas as pd
import streamlit as st

from src import estado
from src.categorias import ASSUNTOS_POR_CATEGORIA
from src.componentes import num, status_dados_na_barra_lateral
from src.indicadores import DIMENSOES, PRESETS_PESOS, unidade_taxa
from src.leitura import NOMES_TIPOS

ss = st.session_state
status_dados_na_barra_lateral()

st.title("📘 Metodologia e dados")

st.header("Fontes")
st.markdown(
    "| Base | O que traz | Uso no sistema |\n|---|---|---|\n"
    "| **Central 156** (Portal de Dados Abertos) | Solicitações, reclamações e denúncias dos cidadãos: assunto, "
    "órgão, bairro, rua, datas de abertura e resposta | Base principal: demandas por bairro, atendimento, previsões |\n"
    "| **Equipamentos urbanos** (Portal de Dados Abertos) | Escolas, unidades de saúde, módulos de segurança, "
    "transporte, esporte, cultura... | Dimensão *Infraestrutura disponível* |\n"
    "| **E-Saúde** (Portal de Dados Abertos) | Cadastro de moradores atendidos: água, esgoto, lixo, tipo de moradia "
    "| Dimensão *Saneamento e moradia* (cada morador contado uma vez) |\n"
    "| **Limites dos bairros** (IPPUC, via GeoJSON) | Desenho e área dos 75 bairros | Mapas |\n"
    "| **IBGE — Censo 2022**, tabela 9923 | População residente por bairro | Taxas por mil habitantes |"
)

st.header("Como a nota é calculada")
st.markdown(
    f"1. Para cada bairro calculamos um indicador por dimensão — por exemplo, **chamados de trânsito por mês "
    f"{unidade_taxa()}**.\n"
    "2. Comparamos os 75 bairros: a nota da dimensão é a **porcentagem de bairros que estão piores**, "
    "em escala de 0 a 10. Nota 8 = melhor que cerca de 80% dos bairros.\n"
    "3. A **nota final** é a média ponderada das dimensões, com os pesos do perfil escolhido. Se uma dimensão não "
    "tem dados no bairro, o peso dela é redistribuído.\n"
    "4. Situação: 🟢 **Boa** (6,5 ou mais), 🟡 **Atenção** (4 a 6,5), 🔴 **Crítica** (menos de 4)."
)
st.dataframe(pd.DataFrame([{"Dimensão": d.nome, "O que mede": d.descricao,
                            "Melhor quando": "maior" if d.maior_melhor else "menor", "Base": NOMES_TIPOS[d.fonte]}
                           for d in DIMENSOES]), hide_index=True, width="stretch")
st.markdown("**Pesos de cada perfil**")
st.dataframe(pd.DataFrame(PRESETS_PESOS).rename(index={d.chave: d.nome for d in DIMENSOES}), width="stretch")

with st.expander("Quais assuntos do 156 entram em cada categoria?"):
    for categoria, assuntos in ASSUNTOS_POR_CATEGORIA.items():
        st.markdown(f"**{categoria}:** {', '.join(assuntos)}")
    st.markdown("Os demais assuntos (IPTU, cartão transporte, escolas etc.) ficam em **Outros serviços** e não entram "
                "na nota. Elogios, sugestões e pedidos de informação também não são contados como demanda.")

st.header("Limitações (importante para interpretar)")
st.markdown(
    "- **O 156 mede quem reclama, não só onde há problema.** Bairros onde a população usa mais o 156 podem "
    "parecer piores; bairros com menos acesso ao serviço podem parecer melhores do que são.\n"
    "- **Bairros centrais recebem muita gente de fora** (trabalho, comércio). Por isso Centro, Rebouças e "
    "vizinhos têm muitos chamados por morador.\n"
    "- **Bairros com poucos moradores** (ex.: Riviera, com 442 habitantes) têm taxas instáveis.\n"
    "- O E-Saúde cobre só moradores **atendidos** na rede municipal, não toda a população.\n"
    "- As notas são **relativas**: mostram a posição do bairro em relação aos outros, não um valor absoluto.\n"
    "- As previsões usam poucos meses de histórico. Efeitos sazonais (ex.: verão e dengue) só serão aprendidos "
    "com mais dados — por isso o re-treino com arquivos novos é importante."
)

st.header("Qualidade dos dados carregados")
for tipo, relatorio in ss.relatorios.items():
    with st.container(border=True):
        st.markdown(f"**{NOMES_TIPOS[tipo]}** — {ss.origens[tipo]}")
        partes = {k: v for k, v in relatorio.items() if isinstance(v, (int, float))}
        colunas = st.columns(len(partes))
        rotulos = {"linhas_lidas": "Linhas lidas", "linhas_validas": "Linhas válidas", "sem_data": "Sem data",
                   "bairro_desconhecido": "Bairro desconhecido", "fora_dos_temas": "Fora dos temas",
                   "moradores_unicos": "Moradores únicos"}
        for coluna, (chave, valor) in zip(colunas, partes.items()):
            coluna.metric(rotulos.get(chave, chave), num(valor))

st.header("Bibliotecas usadas")
st.markdown("**pandas** (leitura e limpeza), **plotly** (gráficos e mapas interativos), **scikit-learn** (modelos), "
            "**streamlit** (interface web), **joblib** (salvar modelos), **lxml** (leitura de XML).")
