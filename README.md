# 🏙️ Raio-X dos Bairros de Curitiba

Aplicação web em Python que analisa a **"saúde" dos 75 bairros de Curitiba** com dados abertos da
prefeitura. Ela atribui uma **nota de 0 a 10** a cada bairro, mostra **pontos de atenção** para a gestão
pública e usa **machine learning** para prever a **tendência dos próximos meses** e o **risco de atraso**
no atendimento de um chamado.

> Trabalho final da disciplina **Tópicos Especiais em Software** — Prof. Rhafael Freitas da Costa.

**Para quem é**
- **Cidadão:** vai se mudar ou se hospedar? Veja a nota, os pontos fortes e fracos e compare bairros.
- **Gestão pública:** onde as demandas estão crescendo, quais ruas concentram problemas, quais órgãos atrasam.

---

## Como rodar

Pré-requisito: **Python 3.12** (ou 3.11+).

```bash
# 1. Criar e ativar o ambiente virtual
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux/Mac

# 2. Instalar as dependências
pip install -r requirements.txt

# 3. Abrir o aplicativo (abre em http://localhost:8501)
streamlit run app.py
```

O sistema já vem com as bases carregadas (Central 156 de jan–set/2026, equipamentos urbanos e E-Saúde).
Para testar o upload, use os arquivos da pasta `data/exemplos/` (outubro/2026 em CSV; primeira semana de
outubro em JSON e XML).

Testes automáticos: `pytest` (leva cerca de 2 minutos).

---

## Como o trabalho atende aos requisitos

| Requisito do professor | Onde está |
|---|---|
| **Upload de .csv, .xml ou .json** com colunas consistentes | Página *Enviar dados* — `src/leitura.py` detecta formato, codificação (UTF-8/Latin-1), separador e o tipo de base pelas colunas |
| **Flexibilidade dos dados** (outras bases com as mesmas colunas) | Novo mês/período é *adicionado* à base (o período do arquivo substitui o mesmo período, sem duplicar) ou *substitui* tudo — `src/preparacao.py::juntar_156` |
| **Gráficos básicos** (barras, pizza) | Rankings, assuntos, ruas, tipos e canais, órgãos |
| **Gráficos avançados/interativos e mapas** | Mapa coroplético dos 75 bairros, radar de comparação, mapa de calor, séries temporais com previsão (Plotly) |
| **Predições personalizadas** | *Previsão de tendência* (por bairro e categoria, 2 a 13 semanas) e simulador de chamado em *Previsão de atraso* |
| **Escolha e configuração de classificadores/preditores** | 6 regressores e 5 classificadores; parâmetros ajustáveis na tela (profundidade, nº de vizinhos, nº de árvores...) — `src/modelos/catalogo.py` |
| **Treinamento dinâmico (re-treino)** | Botão *Re-treinar*; aviso automático quando chegam dados novos; opção de re-treinar ao enviar arquivos; histórico de treinos salvo em `modelos_salvos/` |
| **pandas / matplotlib-seaborn ou interativos / scikit-learn** | pandas (limpeza), Plotly (gráficos interativos), scikit-learn (Pipelines, OneHotEncoder, StandardScaler, métricas) |
| **Código limpo e modular** | Pastas separadas por responsabilidade (abaixo), comentários em português, testes em `tests/` |

---

## Bases de dados

| Base | Fonte | Uso |
|---|---|---|
| Central 156 — solicitações, reclamações e denúncias (620 mil registros em 2026) | [Portal de Dados Abertos de Curitiba](https://dadosabertos.curitiba.pr.gov.br/) | Base principal |
| Equipamentos urbanos (escolas, saúde, segurança, transporte...) | Portal de Dados Abertos de Curitiba | Dimensão *Infraestrutura* |
| E-Saúde — condições de moradia dos atendidos | Portal de Dados Abertos de Curitiba | Dimensão *Saneamento e moradia* |
| Limites dos 75 bairros (GeoJSON) | IPPUC, via [geo-curitiba](https://github.com/gustavokira/geo-curitiba) | Mapas e área |
| População por bairro | IBGE — Censo 2022, tabela SIDRA 9923 | Taxas por mil habitantes |

`scripts/preparar_dados.py` gera as bases de `data/base/` a partir dos arquivos baixados do portal
(também remove colunas pessoais do E-Saúde que o app não usa).

---

## Como a nota funciona

1. Os ~240 assuntos do 156 são agrupados em 5 categorias (`src/categorias.py`): zeladoria urbana,
   trânsito e mobilidade, limpeza e meio ambiente, segurança e vulnerabilidade, saúde pública.
2. Para cada bairro: **chamados por mês por mil habitantes** em cada categoria, tempo de resposta e % de
   chamados atrasados, equipamentos públicos por mil habitantes e % de moradores com saneamento adequado.
3. Cada indicador vira uma nota de 0 a 10 **comparando os bairros entre si** (nota 8 = melhor que ~80% dos bairros).
4. A nota final é uma **média ponderada**; os pesos mudam conforme o perfil (*Equilibrado*, *Vou morar*,
   *Vou visitar*, *Gestão pública*) ou podem ser ajustados manualmente.

**Limitações** (discutidas na página *Metodologia*): o 156 mede quem reclama, bairros centrais recebem
muita gente de fora, bairros com pouquíssimos moradores têm taxas instáveis e as notas são relativas.

## Machine learning

**Previsão de tendência (regressão).** Série semanal de chamados por bairro × categoria. Entradas: as 4
semanas anteriores, média, tendência e nível histórico. Avaliação nas últimas 6 semanas (sem olhar a
resposta) e comparação com uma *regra simples* (média das últimas 4 semanas). Depois o modelo é re-treinado
com tudo e prevê o futuro de forma recursiva. A previsão também gera a **nota projetada** de cada bairro.

**Previsão de atraso (classificação).** Rótulo: o chamado levou mais de 30 dias (ou continua sem resposta
após 30 dias)? Entradas categóricas (assunto, detalhe, órgão, tipo, canal, bairro, regional, dia, mês) com
*one-hot encoding*. Divisão 75% treino / 25% teste estratificada; métricas: acurácia (comparada ao "chute"),
precisão, recall, F1, matriz de confusão e importância das características.

---

## Estrutura do projeto

```
raio-x-bairros/
├── app.py                     # ponto de entrada (navegação entre páginas)
├── paginas/                   # uma página do app por arquivo
│   ├── inicio.py
│   ├── enviar_dados.py        # upload .csv/.json/.xml
│   ├── mapa_cidade.py         # mapa, rankings, gráficos gerais
│   ├── raio_x_bairro.py       # visão do cidadão + comparação
│   ├── pontos_atencao.py      # visão da gestão pública
│   ├── previsao_tendencia.py  # ML: regressão
│   ├── previsao_atraso.py     # ML: classificação + simulador
│   └── metodologia.py
├── src/
│   ├── config.py              # caminhos e constantes
│   ├── leitura.py             # leitura dos arquivos e identificação da base
│   ├── preparacao.py          # limpeza com pandas
│   ├── bairros.py             # lista oficial, nomes, população, GeoJSON
│   ├── categorias.py          # assunto do 156 -> categoria
│   ├── indicadores.py         # indicadores e notas
│   ├── graficos.py            # gráficos Plotly
│   ├── componentes.py         # pedaços de interface reutilizados
│   ├── estado.py              # dados e modelos da sessão (cache)
│   ├── treino.py              # atalhos de treino/re-treino
│   └── modelos/
│       ├── catalogo.py        # algoritmos e parâmetros ajustáveis
│       ├── previsao.py        # regressão (tendência)
│       ├── classificacao.py   # classificação (atraso)
│       └── registro.py        # salvar modelos e histórico de treinos
├── data/
│   ├── base/                  # bases padrão (compactadas)
│   ├── exemplos/              # arquivos para testar o upload
│   ├── geo/bairros.geo.json
│   └── populacao_bairros.csv
├── scripts/preparar_dados.py
└── tests/                     # pytest
```

## Sugestão de divisão de tarefas (defesa individual)

| Integrante | Parte do código para apresentar |
|---|---|
| 1 | Upload e limpeza dos dados — `leitura.py`, `preparacao.py`, página *Enviar dados* |
| 2 | Indicadores e nota — `categorias.py`, `indicadores.py`, `bairros.py`, página *Metodologia* |
| 3 | Visualizações — `graficos.py`, páginas *Mapa da cidade*, *Raio-X do bairro* e *Pontos de atenção* |
| 4 | ML de tendência — `modelos/previsao.py`, `modelos/catalogo.py`, página *Previsão de tendência* |
| 5 | ML de atraso e re-treino — `modelos/classificacao.py`, `registro.py`, `treino.py`, `estado.py` |

## Roteiro sugerido para a demonstração

1. **Início** → explicar o tema e as duas visões (cidadão × gestão).
2. **Mapa da cidade** → trocar o perfil para *Vou morar no bairro* e mostrar o mapa mudando.
3. **Raio-X do bairro** → escolher um bairro, ler o resumo e comparar com outros dois.
4. **Pontos de atenção** → alertas de crescimento, órgãos com atraso, ruas críticas.
5. **Previsão de tendência** → treinar Regressão Linear e depois Random Forest; comparar com a regra simples.
6. **Previsão de atraso** → treinar Árvore de Decisão, mudar a profundidade, simular um chamado.
7. **Enviar dados** → enviar `data/exemplos/156_outubro_2026.csv` com re-treino automático e mostrar
   que o período e os modelos foram atualizados.
