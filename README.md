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

### População por bairro

O sistema usa a **população residente de cada bairro** (não só os chamados) para calcular taxas
**por mil habitantes** — sem isso, bairros grandes como a Cidade Industrial (172.510 moradores) sempre
pareceriam piores que bairros pequenos como o Riviera (442 moradores).

- **Fonte:** IBGE, Censo Demográfico 2022, tabela SIDRA 9923 (nível territorial *Bairro*), obtida pela
  [API de agregados do IBGE](https://servicodados.ibge.gov.br/api/docs/agregados?versao=3). Total: 1.773.718 habitantes.
- **Arquivo:** `data/populacao_bairros.csv` (`bairro;populacao`). A primeira linha (`# Fonte: ...`) registra a origem
  e é mostrada no app.
- **Como atualizar:**
  1. **Pelo app** — página *Enviar dados*: envie um CSV `bairro;populacao` com os 75 bairros (ex.: estimativas do IPPUC).
  2. **Pelo IBGE** — `python scripts/atualizar_populacao.py` (Censo 2022) ou
     `python scripts/atualizar_populacao.py <tabela> <ano>` quando houver tabela nova por bairro.
  3. **À mão** — editando o CSV.

---

## Indicadores

Definidos em `src/indicadores.py` e descritos (para o app) em `src/dicionario.py`.
"Demanda" = chamado do tipo Solicitação, Reclamação ou Denúncia (elogios, sugestões e informações não contam).

| Dimensão | Indicador | Como é calculado | Unidade | Melhor quando |
|---|---|---|---|---|
| Zeladoria urbana | Chamados de zeladoria (buracos, iluminação, calçadas, drenagem) | demandas da categoria ÷ meses do período ÷ (população ÷ 1.000) | chamados/mês por mil hab. | menor |
| Trânsito e mobilidade | Chamados de trânsito (estacionamento irregular, semáforos, sinalização, ônibus) | idem | chamados/mês por mil hab. | menor |
| Limpeza e meio ambiente | Chamados de limpeza (varrição, árvores, poluição, terrenos, animais) | idem | chamados/mês por mil hab. | menor |
| Segurança e vulnerabilidade | Chamados de segurança (situação de rua, Guarda Municipal, invasões, violência) | idem | chamados/mês por mil hab. | menor |
| Saúde pública | Chamados de saúde (dengue, vigilância sanitária e de zoonoses, UMS, UPA) | idem | chamados/mês por mil hab. | menor |
| Atendimento da prefeitura | Tempo típico de resposta | mediana de (data da resposta − data de abertura) | dias | menor |
| Atendimento da prefeitura | Chamados atrasados | chamados com +30 dias sem resposta ÷ chamados com ao menos 30 dias de idade | % | menor |
| Infraestrutura disponível | Equipamentos públicos (educação, saúde, segurança, transporte, esporte, cultura, assistência...) | nº de equipamentos ÷ (população ÷ 1.000) | equipamentos por mil hab. | maior |
| Saneamento e moradia | Saneamento e moradia adequados | média dos % de moradores com água da rede, lixo coletado, esgoto adequado e moradia adequada (E-Saúde, cada morador uma vez, mínimo 30 por bairro) | % | maior |

## Como a nota funciona

1. Calcula-se cada indicador acima para os 75 bairros.
2. Cada indicador vira uma nota de 0 a 10 **comparando os bairros entre si**: nota = fração de bairros
   que estão *piores* × 10 (nota 8 = melhor que ~80% dos bairros). Atendimento = média das notas dos seus 2 indicadores.
3. A nota final é a **média ponderada** das 8 dimensões; os pesos mudam conforme o perfil (*Equilibrado*,
   *Vou morar*, *Vou visitar*, *Gestão pública*) ou podem ser ajustados manualmente.
4. Situação: 🟢 Boa (≥ 6,5), 🟡 Atenção (4 a 6,5), 🔴 Crítica (< 4).

As notas são **relativas** (posição entre os bairros). Por isso o app sempre mostra também o **valor real**
de cada indicador e a mediana da cidade (Raio-X do bairro, tabela do Mapa e página *Metodologia*, que traz um
exemplo de cálculo passo a passo com números reais).

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
│   ├── dicionario.py          # definição em texto de bases, termos e indicadores
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
├── scripts/
│   ├── preparar_dados.py      # gera data/base/ a partir dos downloads do portal
│   └── atualizar_populacao.py # baixa a população por bairro da API do IBGE
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
