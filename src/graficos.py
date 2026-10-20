"""Funções que montam os gráficos (Plotly). As páginas só chamam estas funções.

Regras de cor usadas no projeto:
- cada categoria tem SEMPRE a mesma cor (não muda conforme o filtro);
- notas usam uma escala divergente: vermelho (ruim) -> cinza (meio) -> azul (bom);
- as situações Boa/Atenção/Crítica usam verde/amarelo/vermelho e sempre vêm com texto.
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src import categorias as cat
from src.bairros import carregar_geojson

CORES_CATEGORIAS = {
    cat.ZELADORIA: "#2a78d6",
    cat.TRANSITO: "#eb6834",
    cat.LIMPEZA: "#1baf7a",
    cat.SEGURANCA: "#eda100",
    cat.SAUDE: "#e87ba4",
    cat.COLETA: "#898781",
    cat.OUTROS: "#c3c2b7",
}
CORES_SITUACAO = {"Boa": "#0ca30c", "Atenção": "#fab219", "Crítica": "#d03b3b", "Sem dados": "#c3c2b7"}
ICONES_SITUACAO = {"Boa": "🟢", "Atenção": "🟡", "Crítica": "🔴", "Sem dados": "⚪"}
ESCALA_NOTA = [[0, "#b8322f"], [0.25, "#ec8f8c"], [0.5, "#e1e0d9"], [0.75, "#6da7ec"], [1, "#1c5cab"]]
ESCALA_VARIACAO = [[0, "#1c5cab"], [0.5, "#e1e0d9"], [1, "#b8322f"]]  # queda = azul, alta = vermelho
AZUL, CINZA, LARANJA = "#2a78d6", "#898781", "#eb6834"

CENTRO_CURITIBA = {"lat": -25.48, "lon": -49.28}


def _ajustar(fig: go.Figure, altura: int = 380, legenda_embaixo: bool = True) -> go.Figure:
    """Estilo comum: margens enxutas, grade discreta, legenda horizontal."""
    fig.update_layout(height=altura, margin=dict(l=10, r=10, t=40, b=10), hoverlabel=dict(namelength=-1))
    if legenda_embaixo:
        fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.15, x=0, title_text=""))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(137,135,129,0.25)", zeroline=False)
    return fig


def mapa_bairros(dados: pd.DataFrame, coluna: str, titulo_cor: str, escala=None, faixa=None,
                 formato: str = ".1f", altura: int = 560) -> go.Figure:
    """Mapa coroplético dos 75 bairros. `dados` precisa ter o índice = chave do bairro."""
    tabela = dados.reset_index().rename(columns={"index": "chave"})
    fig = px.choropleth_map(
        tabela, geojson=carregar_geojson(), locations="chave", featureidkey="properties.chave",
        color=coluna, color_continuous_scale=escala or ESCALA_NOTA, range_color=faixa,
        hover_name="bairro", hover_data={"chave": False, coluna: f":{formato}"},
        labels={coluna: titulo_cor}, map_style="carto-positron", zoom=9.6, center=CENTRO_CURITIBA,
        opacity=0.8,
    )
    fig.update_traces(marker_line_width=0.6, marker_line_color="white")
    fig.update_layout(height=altura, margin=dict(l=0, r=0, t=0, b=0),
                      coloraxis_colorbar=dict(title=titulo_cor, thickness=12))
    return fig


def barras_ranking(dados: pd.DataFrame, coluna: str, titulo: str, n: int = 10, maiores: bool = True,
                   cor: str = AZUL, formato: str = ".1f") -> go.Figure:
    """Barras horizontais com os N bairros de maior (ou menor) valor na coluna."""
    ordenado = dados.sort_values(coluna, ascending=not maiores).head(n).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=ordenado[coluna], y=ordenado["bairro"], orientation="h", marker_color=cor,
        text=ordenado[coluna].map(lambda v: f"{v:{formato}}"), textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:" + formato + "}<extra></extra>",
    ))
    fig.update_layout(title=titulo, xaxis_title=None, yaxis_title=None)
    return _ajustar(fig, altura=max(300, 32 * n + 60), legenda_embaixo=False)


def linhas_por_categoria(serie: pd.DataFrame, coluna_tempo: str, titulo: str, categorias=None) -> go.Figure:
    """Evolução no tempo, uma linha por categoria (cores fixas)."""
    categorias = categorias or cat.CATEGORIAS_DA_NOTA
    dados = serie[serie["categoria"].isin(categorias)]
    fig = px.line(dados, x=coluna_tempo, y="qtd", color="categoria", markers=True,
                  color_discrete_map=CORES_CATEGORIAS, category_orders={"categoria": categorias},
                  labels={"qtd": "Chamados", coluna_tempo: "", "categoria": "Categoria"})
    fig.update_traces(line_width=2, marker_size=6)
    fig.update_layout(title=titulo, hovermode="x unified")
    return _ajustar(fig)


def pizza(contagem: pd.Series, titulo: str) -> go.Figure:
    """Gráfico de rosca (até ~6 fatias; o resto vira 'Outros')."""
    contagem = contagem.sort_values(ascending=False)
    if len(contagem) > 6:
        contagem = pd.concat([contagem.head(5), pd.Series({"Outros": contagem.iloc[5:].sum()})])
    fig = go.Figure(go.Pie(labels=contagem.index, values=contagem.values, hole=0.5, sort=False,
                           marker=dict(colors=[AZUL, LARANJA, "#1baf7a", "#eda100", "#e87ba4", CINZA],
                                       line=dict(color="white", width=2)),
                           textinfo="percent", hovertemplate="%{label}: %{value:,} (%{percent})<extra></extra>"))
    fig.update_layout(title=titulo)
    return _ajustar(fig, altura=340)


def barras_empilhadas(tabela: pd.DataFrame, titulo: str, rotulo_y: str) -> go.Figure:
    """Barras empilhadas: linhas da tabela no eixo X, colunas = categorias."""
    fig = go.Figure()
    for categoria in [c for c in cat.TODAS_CATEGORIAS if c in tabela.columns]:
        fig.add_bar(name=categoria, x=tabela.index, y=tabela[categoria], marker_color=CORES_CATEGORIAS[categoria],
                    marker_line=dict(color="white", width=1),
                    hovertemplate="%{x}<br>" + categoria + ": %{y:,.1f}<extra></extra>")
    fig.update_layout(barmode="stack", title=titulo, yaxis_title=rotulo_y)
    return _ajustar(fig, altura=420)


def barras_simples(contagem: pd.Series, titulo: str, cor: str = AZUL, formato: str = ",.0f",
                   rotulo_x: str = "") -> go.Figure:
    """Barras horizontais a partir de uma Series (índice = rótulo)."""
    contagem = contagem.iloc[::-1]
    fig = go.Figure(go.Bar(x=contagem.values, y=[str(i) for i in contagem.index], orientation="h",
                           marker_color=cor, text=[f"{v:{formato}}" for v in contagem.values],
                           textposition="outside", cliponaxis=False,
                           hovertemplate="%{y}: %{x:" + formato + "}<extra></extra>"))
    fig.update_layout(title=titulo, xaxis_title=rotulo_x)
    return _ajustar(fig, altura=max(280, 30 * len(contagem) + 70), legenda_embaixo=False)


def radar(notas: dict[str, pd.Series], dimensoes: list) -> go.Figure:
    """Radar com as notas por dimensão. `notas` = {rótulo: Series(chave_dimensao -> nota)}."""
    cores = [AZUL, CINZA, LARANJA, "#1baf7a"]
    nomes = [d.nome for d in dimensoes]
    fig = go.Figure()
    for (rotulo, valores), cor in zip(notas.items(), cores):
        r = [valores.get(d.chave) for d in dimensoes]
        fig.add_scatterpolar(r=r + r[:1], theta=nomes + nomes[:1], name=rotulo, line=dict(color=cor, width=2),
                             fill="toself" if cor == AZUL else None, opacity=0.9,
                             hovertemplate="%{theta}: %{r:.1f}<extra>" + rotulo + "</extra>")
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 10], tickvals=[0, 2.5, 5, 7.5, 10])))
    return _ajustar(fig, altura=430)


def mapa_de_calor(tabela: pd.DataFrame, titulo: str, rotulo: str) -> go.Figure:
    """Mapa de calor (linhas = bairros, colunas = categorias), escala de um só tom."""
    fig = px.imshow(tabela, color_continuous_scale=["#f0efec", "#86b6ef", "#2a78d6", "#104281"],
                    aspect="auto", labels=dict(color=rotulo), text_auto=".1f")
    fig.update_layout(title=titulo, xaxis_title=None, yaxis_title=None)
    fig.update_xaxes(side="top")
    return _ajustar(fig, altura=max(400, 24 * len(tabela) + 120), legenda_embaixo=False)


def previsao_vs_real(tabela: pd.DataFrame, titulo: str) -> go.Figure:
    """Linhas Real x Modelo x Regra simples nas semanas de teste."""
    estilos = {"Real": dict(color="#0b0b0b", width=2.5), "Modelo": dict(color=AZUL, width=2),
               "Regra simples": dict(color=CINZA, width=2, dash="dot")}
    fig = go.Figure()
    for coluna, estilo in estilos.items():
        fig.add_scatter(x=tabela.index, y=tabela[coluna], name=coluna, mode="lines+markers", line=estilo,
                        hovertemplate="%{y:,.0f}<extra>" + coluna + "</extra>")
    fig.update_layout(title=titulo, hovermode="x unified", yaxis_title="Chamados na semana")
    return _ajustar(fig)


def historico_e_previsao(historico: pd.Series, futuro: pd.Series, titulo: str) -> go.Figure:
    """Série histórica (linha cheia) e previsão (linha tracejada) na mesma escala."""
    fig = go.Figure()
    fig.add_scatter(x=historico.index, y=historico.values, name="Histórico", mode="lines+markers",
                    line=dict(color=AZUL, width=2), hovertemplate="%{y:,.0f}<extra>Histórico</extra>")
    ligacao = pd.concat([historico.tail(1), futuro])  # liga a última semana real à previsão
    fig.add_scatter(x=ligacao.index, y=ligacao.values, name="Previsão", mode="lines+markers",
                    line=dict(color=LARANJA, width=2, dash="dash"), hovertemplate="%{y:,.0f}<extra>Previsão</extra>")
    fig.add_vrect(x0=futuro.index.min(), x1=futuro.index.max(), fillcolor=LARANJA, opacity=0.06, line_width=0)
    fig.update_layout(title=titulo, hovermode="x unified", yaxis_title="Chamados por semana")
    return _ajustar(fig)


def matriz_confusao(matriz, titulo: str = "Matriz de confusão") -> go.Figure:
    rotulos = ["No prazo", "Atrasou"]
    fig = px.imshow(matriz, x=[f"Previsto: {r}" for r in rotulos], y=[f"Real: {r}" for r in rotulos],
                    color_continuous_scale=["#f0efec", "#2a78d6"], text_auto=",")
    fig.update_layout(title=titulo, coloraxis_showscale=False)
    return _ajustar(fig, altura=340, legenda_embaixo=False)
