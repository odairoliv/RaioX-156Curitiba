"""Funções que montam os gráficos (Plotly). As páginas só chamam estas funções.

Cores: azul = valor principal; cinza = referência (média da cidade, chute);
laranja/vermelho = atraso. O mapa usa uma escala de um tom só (claro = pouco atraso,
escuro = muito atraso).
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.bairros import carregar_geojson

AZUL, CINZA, LARANJA, VERMELHO = "#2a78d6", "#898781", "#eb6834", "#d03b3b"
ESCALA_ATRASO = ["#fdf0e8", "#f6b38f", "#eb6834", "#b8322f"]
CENTRO_CURITIBA = {"lat": -25.48, "lon": -49.28}


def _pct(valor: float) -> str:
    return f"{valor:.1f}%".replace(".", ",")


def _ajustar(fig: go.Figure, altura: int = 380) -> go.Figure:
    """Estilo comum: margens enxutas, grade discreta, legenda horizontal embaixo."""
    fig.update_layout(height=altura, margin=dict(l=10, r=10, t=50, b=10), hoverlabel=dict(namelength=-1),
                      legend=dict(orientation="h", yanchor="top", y=-0.15, x=0, title_text=""))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(137,135,129,0.25)", zeroline=False)
    return fig


def mapa_atraso(tabela: pd.DataFrame, altura: int = 560) -> go.Figure:
    """Mapa dos bairros colorido pela % de chamados atrasados. Índice da tabela = chave do bairro."""
    dados = tabela.reset_index().rename(columns={"index": "chave"})
    fig = px.choropleth_map(
        dados, geojson=carregar_geojson(), locations="chave", featureidkey="properties.chave",
        color="pct_atraso", color_continuous_scale=ESCALA_ATRASO, hover_name="bairro",
        hover_data={"chave": False, "pct_atraso": ":.1f", "chamados": ":,", "mediana_dias": ":.0f", "regional": True},
        labels={"pct_atraso": "% atrasados", "chamados": "Chamados", "mediana_dias": "Dias (mediana)",
                "regional": "Regional"},
        map_style="carto-positron", zoom=9.6, center=CENTRO_CURITIBA, opacity=0.85,
    )
    fig.update_traces(marker_line_width=0.6, marker_line_color="white")
    fig.update_layout(height=altura, margin=dict(l=0, r=0, t=0, b=0),
                      coloraxis_colorbar=dict(title="% atrasados", thickness=12))
    return fig


def barras_pct(tabela: pd.DataFrame, titulo: str, media: float | None = None, coluna: str = "pct_atraso",
               cor: str = LARANJA) -> go.Figure:
    """Barras horizontais com a % de atraso de cada grupo e uma linha com a média da cidade."""
    tabela = tabela.sort_values(coluna)
    rotulos = [str(i) for i in tabela.index]
    fig = go.Figure(go.Bar(
        x=tabela[coluna], y=rotulos, orientation="h", marker_color=cor,
        text=[f"{v:.1f}%".replace(".", ",") for v in tabela[coluna]], textposition="outside", cliponaxis=False,
        customdata=tabela[["chamados"]] if "chamados" in tabela else None,
        hovertemplate="%{y}: %{x:.1f}% atrasados" + ("<br>%{customdata[0]:,} chamados" if "chamados" in tabela else "")
                      + "<extra></extra>",
    ))
    if media is not None:
        fig.add_vline(x=media, line_dash="dot", line_color=CINZA,
                      annotation_text=f"média da cidade: {media:.1f}%".replace(".", ","), annotation_position="top")
    fig.update_layout(title=titulo, xaxis_title="% de chamados atrasados", yaxis_title=None)
    return _ajustar(fig, altura=max(300, 34 * len(tabela) + 110))


def comparacao_justa(tabela: pd.DataFrame, titulo: str, media: float) -> go.Figure:
    """Para cada região: % de atraso real x % ajustada (como se todas tivessem os mesmos assuntos)."""
    tabela = tabela.sort_values("observado")
    fig = go.Figure()
    fig.add_bar(name="Real (observado)", y=tabela.index, x=tabela["observado"], orientation="h",
                marker_color=LARANJA, hovertemplate="%{y}: %{x:.1f}% real<extra></extra>",
                text=[_pct(v) for v in tabela["observado"]], textposition="outside", cliponaxis=False)
    fig.add_bar(name="Com a mesma mistura de assuntos da cidade (ajustado)", y=tabela.index, x=tabela["ajustado"],
                orientation="h", marker_color=AZUL, hovertemplate="%{y}: %{x:.1f}% ajustado<extra></extra>",
                text=[_pct(v) for v in tabela["ajustado"]], textposition="outside", cliponaxis=False)
    fig.add_vline(x=media, line_dash="dot", line_color=CINZA,
                  annotation_text=f"média: {media:.1f}%".replace(".", ","), annotation_position="top")
    fig.update_layout(title=titulo, barmode="group", xaxis_title="% de chamados atrasados", yaxis_title=None,
                      bargap=0.25, bargroupgap=0.05)
    return _ajustar(fig, altura=max(380, 52 * len(tabela) + 120))


def linha_evolucao(tabela: pd.DataFrame, titulo: str) -> go.Figure:
    """% de atraso mês a mês."""
    fig = go.Figure(go.Scatter(
        x=tabela.index, y=tabela["pct_atraso"], mode="lines+markers+text", line=dict(color=LARANJA, width=2),
        marker_size=8, text=[f"{v:.0f}%" for v in tabela["pct_atraso"]], textposition="top center",
        hovertemplate="%{x|%m/%Y}: %{y:.1f}% atrasados<extra></extra>",
    ))
    fig.update_layout(title=titulo, yaxis_title="% de chamados atrasados", xaxis_title=None,
                      yaxis_range=[0, max(tabela["pct_atraso"].max() * 1.25, 10)])
    return _ajustar(fig, altura=360)


def barras_simples(serie: pd.Series, titulo: str, cor: str = AZUL, formato: str = ".1f",
                   rotulo_x: str = "") -> go.Figure:
    """Barras horizontais a partir de uma Series (índice = rótulo). O maior valor fica em cima."""
    serie = serie.sort_values()
    fig = go.Figure(go.Bar(x=serie.values, y=[str(i) for i in serie.index], orientation="h", marker_color=cor,
                           text=[f"{v:{formato}}".replace(".", ",") for v in serie.values], textposition="outside",
                           cliponaxis=False, hovertemplate="%{y}: %{x:" + formato + "}<extra></extra>"))
    fig.update_layout(title=titulo, xaxis_title=rotulo_x)
    return _ajustar(fig, altura=max(260, 32 * len(serie) + 90))


def matriz_confusao(matriz) -> go.Figure:
    rotulos = ["No prazo", "Atrasou"]
    fig = px.imshow(matriz, x=[f"Previsto: {r}" for r in rotulos], y=[f"Real: {r}" for r in rotulos],
                    color_continuous_scale=["#f0efec", AZUL], text_auto=",")
    fig.update_layout(title="Matriz de confusão (chamados de teste)", coloraxis_showscale=False)
    return _ajustar(fig, altura=340)


def real_vs_previsto(tabela: pd.DataFrame, limite_dias: int = 120, amostra: int = 3000) -> go.Figure:
    """Dispersão dias reais x dias previstos (amostra) com a linha do acerto perfeito."""
    pontos = tabela.sample(min(amostra, len(tabela)), random_state=42)
    fig = go.Figure()
    fig.add_scatter(x=pontos["Real"], y=pontos["Previsto"], mode="markers", name="Chamados de teste",
                    marker=dict(color=AZUL, size=6, opacity=0.35),
                    hovertemplate="Real: %{x:.0f} dias<br>Previsto: %{y:.0f} dias<extra></extra>")
    fig.add_scatter(x=[0, limite_dias], y=[0, limite_dias], mode="lines", name="Acerto perfeito",
                    line=dict(color=CINZA, dash="dot"))
    fig.update_layout(title="Dias reais x dias previstos (amostra do teste)", xaxis_title="Dias reais até a resposta",
                      yaxis_title="Dias previstos", xaxis_range=[0, limite_dias], yaxis_range=[0, limite_dias])
    return _ajustar(fig, altura=400)
