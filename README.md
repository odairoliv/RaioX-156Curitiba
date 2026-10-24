# ⏱️ Seu chamado vai demorar?

**Análise e previsão de atrasos na Central 156 de Curitiba.**

Quando um morador pede algo à prefeitura pela Central 156 — trocar uma lâmpada, tapar um buraco, podar uma
árvore — quanto tempo leva para ser atendido? **E isso depende de onde a pessoa mora?**

Aplicação web em Python que permite enviar a base de chamados do 156 (.csv, .json ou .xml), testa quatro
hipóteses com gráficos e usa machine learning para prever **se um chamado vai atrasar** (classificação) e
**quantos dias deve levar** (regressão).

> Trabalho final da disciplina **Tópicos Especiais em Software** — Prof. Rhafael Freitas da Costa.

---

## Hipóteses

Um chamado é considerado **atrasado** quando levou **mais de 30 dias** para ser respondido (ou continua sem
resposta depois disso).

| | Hipótese | Como testamos | Resultado (jan–set/2026) |
|---|---|---|---|
| **H1** | A **região** onde a pessoa mora muda a chance de atraso | % de atraso por regional e mapa por bairro | **Em parte.** Cajuru 36,8% × Matriz 14,5%. Comparando os mesmos assuntos, a diferença cai de 22 para 12 pontos — mas CIC e Cajuru continuam acima da média |
| **H2** | O **tipo de problema** pesa mais que a região | % por órgão; comparação justa entre regiões; modelo treinado só com *Onde* × só com *O quê* | **Sim.** Entre órgãos a diferença chega a 70 pontos; o tipo de problema explica ~45% da diferença entre regiões; o modelo com *O quê* prevê bem melhor que com *Onde* |
| **H3** | **Denúncias** demoram mais | % por tipo de chamado | **Sim.** Denúncias 43% × reclamações 16% |
| **H4** | O **canal** muda a velocidade | % por canal | **Sim, com ressalva.** Chat 20% × Portal 34% (cada canal recebe assuntos diferentes) |

As conclusões são **recalculadas automaticamente** no app a cada novo arquivo enviado.

---

## Como rodar

Pré-requisito: **Python 3.11+**.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py            # abre em http://localhost:8501
```

O app já vem com a base de **janeiro a setembro de 2026** (585 mil chamados). Para testar o envio de dados novos
e o re-treino, use os arquivos de `data/exemplos/`: outubro/2026 em CSV e a primeira semana de outubro em JSON e XML.

Testes automáticos: `pytest` (cerca de 1 minuto).

---

## Páginas

| Página | O que faz |
|---|---|
| **Início** | A pergunta, as hipóteses e os números gerais |
| **Enviar dados** | Upload de .csv/.json/.xml; o arquivo pode **somar-se** à base (o período dele substitui o mesmo período) ou **substituí-la**; opção de re-treinar os modelos automaticamente |
| **Análise das hipóteses** | Mapa de % de atraso por bairro, gráficos por regional, órgão, tipo e canal, comparação justa e a conclusão de cada hipótese |
| **Previsão** | *Vai atrasar?* (classificação) e *Quantos dias?* (regressão): escolha do algoritmo, ajuste de parâmetros, escolha das características, treino/re-treino, métricas explicadas e simulador de chamado |
| **Sobre os dados** | O que é a base, o significado das colunas, definições, limitações e qualidade dos dados |

## Como o trabalho atende aos requisitos

| Requisito | Onde está |
|---|---|
| Upload de .csv, .xml ou .json com colunas consistentes | `src/leitura.py` detecta formato, codificação (UTF-8/Latin-1) e separador, e confere as colunas obrigatórias |
| Gráficos básicos e avançados/interativos, mapas | Barras, linha do tempo, mapa coroplético dos 75 bairros, matriz de confusão, dispersão real × previsto (Plotly) |
| Predições personalizadas | Simulador: escolha órgão, assunto, bairro, tipo, canal e dia → chance de atraso e dias previstos |
| Escolha e configuração de classificadores/preditores | 5 classificadores e 6 regressores com parâmetros ajustáveis (`src/modelos/catalogo.py`) |
| Treinamento dinâmico | Botão *Re-treinar*, aviso quando chegam dados novos, re-treino automático no upload, histórico em `modelos_salvos/` |
| pandas / gráficos / scikit-learn | pandas (limpeza), Plotly (gráficos), scikit-learn (Pipeline, OneHotEncoder, métricas, permutation importance) |

---

## Dados

**Central 156** — [Portal de Dados Abertos de Curitiba](https://dadosabertos.curitiba.pr.gov.br/). Cada linha é um
chamado. Colunas usadas:

| Coluna | Uso |
|---|---|
| `DataCriacao`, `DataResposta` | Dias até a resposta e se atrasou |
| `Bairro`, `Regional` | H1 (região) e mapa |
| `Orgao`, `Assunto`, `Subdivisao` | H2 (tipo de problema) |
| `Tipo` | H3 (solicitação, reclamação, denúncia) |
| `Origem` | H4 (canal) |
| `Situacao` | Chamados ainda abertos |

**Base analisada:** solicitações, reclamações e denúncias abertas há pelo menos 30 dias (as mais recentes ainda não
tiveram tempo de atrasar). Na base padrão são 515 mil chamados, dos quais 25,9% atrasaram.

**Mapa:** limites dos 75 bairros do IPPUC em GeoJSON ([geo-curitiba](https://github.com/gustavokira/geo-curitiba)).

`scripts/preparar_dados.py` gera `data/base/` e `data/exemplos/` a partir do arquivo baixado do portal.

## Comparação justa (H2)

Cada região pede coisas diferentes (o Centro tem muita abordagem social, resolvida em 1 dia; outras regiões têm
mais poda de árvore, que demora semanas). Para comparar "maçã com maçã":

1. calcula-se a % de atraso de cada **assunto** na cidade;
2. **esperado** da região = média dessas porcentagens para os assuntos que ela pediu;
3. **ajustado = real ÷ esperado × média da cidade** — o atraso que a região teria com a mesma mistura de assuntos da cidade.

Se os ajustados continuam diferentes, a região importa mesmo descontando o tipo de problema (técnica de *padronização*).

## Machine learning

- **Entradas:** órgão, assunto, detalhe, tipo, canal, bairro, regional, dia da semana e mês, em *one-hot encoding*.
  Os grupos podem ser ligados/desligados — treinar só com *Onde* ou só com *O quê* testa a H2 diretamente.
- **Vai atrasar? (classificação):** Árvore de Decisão, KNN, Random Forest, Regressão Logística, Naive Bayes.
  Métricas: acurácia (comparada ao "chute"), precisão, recall, F1 e matriz de confusão.
- **Quantos dias? (regressão):** Regressão Linear, Ridge, Árvore de Decisão, KNN, Random Forest, Gradient Boosting,
  treinados no logaritmo dos dias. Métricas: erro médio (comparado ao "chute" da mediana), erro mediano,
  % de previsões com erro de até 7 dias e R².
- **Avaliação:** 75% treino / 25% teste. **O que mais pesa:** importância por permutação (quanto o modelo piora
  quando uma informação é embaralhada) — vale igualmente para todos os algoritmos.
- **Resultados típicos:** classificação F1 ≈ 0,6; regressão erra em média ~14 dias (contra ~18 do chute), com
  57% das previsões a até 7 dias do real. O prazo exato depende de fatores fora da base, por isso a regressão é
  uma estimativa.

---

## Estrutura

```
├── app.py                    # ponto de entrada (navegação)
├── paginas/
│   ├── inicio.py
│   ├── enviar_dados.py       # upload .csv/.json/.xml
│   ├── hipoteses.py          # gráficos e conclusões H1–H4
│   ├── previsao.py           # classificação, regressão e simulador
│   └── sobre.py
├── src/
│   ├── config.py             # caminhos e constantes (prazo de 30 dias etc.)
│   ├── leitura.py            # leitura dos arquivos e validação das colunas
│   ├── preparacao.py         # limpeza com pandas
│   ├── bairros.py            # nomes dos bairros e GeoJSON
│   ├── analise.py            # cálculos das hipóteses (inclui a comparação justa)
│   ├── graficos.py           # gráficos Plotly
│   ├── componentes.py        # pedaços de interface reutilizados
│   ├── estado.py             # dados e modelos da sessão
│   ├── treino.py             # treinar / re-treinar
│   └── modelos/
│       ├── catalogo.py       # algoritmos e parâmetros ajustáveis
│       ├── comum.py          # características, one-hot, importância por permutação
│       ├── classificacao.py  # vai atrasar?
│       ├── regressao.py      # quantos dias?
│       └── registro.py       # salvar modelos e histórico de treinos
├── data/  base/ · exemplos/ · geo/
├── scripts/preparar_dados.py
└── tests/
```

## Sugestão de divisão para a defesa

| Integrante | Parte |
|---|---|
| 1 | Dados: upload, leitura e limpeza — `leitura.py`, `preparacao.py`, página *Enviar dados* |
| 2 | Análise das hipóteses — `analise.py`, página *Análise das hipóteses* (comparação justa) |
| 3 | Visualização — `graficos.py`, mapa e gráficos |
| 4 | Classificação — `classificacao.py`, `comum.py`, métricas e matriz de confusão |
| 5 | Regressão e re-treino — `regressao.py`, `treino.py`, `estado.py`, simulador |

## Roteiro da demonstração

1. **Início:** a pergunta e as 4 hipóteses.
2. **Análise:** mapa e regionais (H1) → órgãos e comparação justa (H2) → tipo (H3) e canal (H4) → resumo.
3. **Previsão – Vai atrasar?:** treinar com todas as características; depois só *Onde* e só *O quê* → comparar F1 (H2).
4. **Previsão – Quantos dias?:** treinar e mostrar erro médio × chute; simular um chamado e trocar só o bairro.
5. **Enviar dados:** enviar `data/exemplos/156_outubro_2026.csv` com re-treino automático e mostrar o período e
   os modelos atualizados.

## Limitações

- O 156 registra quem pede; problemas não registrados não aparecem.
- A data de resposta é quando a prefeitura respondeu, que nem sempre coincide com a solução no local.
- O bairro é o local do problema, não necessariamente onde a pessoa mora.
- Poucos meses de dados: efeitos de estação do ano ainda não aparecem — por isso o re-treino com arquivos novos.
