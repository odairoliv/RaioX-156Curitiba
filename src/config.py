"""Caminhos e constantes usados em todo o projeto."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
PASTA_DADOS = RAIZ / "data"
PASTA_EXEMPLOS = PASTA_DADOS / "exemplos"  # arquivos para testar o upload
PASTA_CACHE = PASTA_DADOS / "cache"        # base já limpa (gerado sozinho, fora do git)
ARQUIVO_GEOJSON = PASTA_DADOS / "geo" / "bairros.geo.json"
PASTA_MODELOS = RAIZ / "modelos_salvos"

# Base carregada ao abrir o app (o usuário pode substituí-la ou complementá-la pelo upload)
BASE_PADRAO = PASTA_DADOS / "base" / "156_jan_a_set_2026.csv.gz"

# Um chamado que levou mais do que este número de dias para ser respondido
# (ou continua sem resposta depois disso) é considerado ATRASADO
PRAZO_ATRASO_DIAS = 30

# Grupos com menos chamados que isso ficam fora dos gráficos de comparação (porcentagem instável)
MINIMO_CHAMADOS_GRUPO = 200
