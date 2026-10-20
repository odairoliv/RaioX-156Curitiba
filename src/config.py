"""Caminhos e constantes usados em todo o projeto."""
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
PASTA_DADOS = RAIZ / "data"
PASTA_BASE = PASTA_DADOS / "base"          # bases que já vêm com o projeto
PASTA_EXEMPLOS = PASTA_DADOS / "exemplos"  # arquivos para testar o upload
PASTA_CACHE = PASTA_DADOS / "cache"        # bases já limpas (gerado sozinho, fora do git)
ARQUIVO_GEOJSON = PASTA_DADOS / "geo" / "bairros.geo.json"
ARQUIVO_POPULACAO = PASTA_DADOS / "populacao_bairros.csv"
PASTA_MODELOS = RAIZ / "modelos_salvos"

# Bases padrão carregadas ao abrir o app (o usuário pode substituí-las pelo upload)
BASES_PADRAO = {
    "156": PASTA_BASE / "156_jan_a_set_2026.csv.gz",
    "equipamentos": PASTA_BASE / "equipamentos_urbanos_2026.csv.gz",
    "esaude": PASTA_BASE / "esaude_condicoes_moradia_jul_a_set_2026.csv.gz",
}

# Um chamado aberto há mais que este número de dias sem resposta é considerado "atrasado"
PRAZO_ATRASO_DIAS = 30
