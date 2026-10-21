"""Atualiza data/populacao_bairros.csv com a população por bairro do IBGE.

Fonte: API de agregados do IBGE (SIDRA), tabela 9923 do Censo Demográfico 2022
("População residente, por situação do domicílio"), nível territorial Bairro (N102),
município de Curitiba (código IBGE 4106902).

Uso:
    python scripts/atualizar_populacao.py                  # Censo 2022 (padrão)
    python scripts/atualizar_populacao.py 9923 2022        # tabela e ano explícitos

Quando o IBGE publicar uma tabela mais nova por bairro (ex.: um novo censo),
basta passar o número da tabela e o ano. Para usar outra fonte (ex.: estimativas
do IPPUC), edite o CSV à mão ou envie o arquivo na página "Enviar dados" do app.
"""
import json
import sys
import urllib.request
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.bairros import chave_bairro, tabela_bairros  # noqa: E402
from src.config import ARQUIVO_POPULACAO  # noqa: E402

CODIGO_CURITIBA = "4106902"
VARIAVEL_POPULACAO = "93"  # "População residente"


def baixar(tabela: str, ano: str) -> list[tuple[str, int]]:
    url = (f"https://servicodados.ibge.gov.br/api/v3/agregados/{tabela}/periodos/{ano}/"
           f"variaveis/{VARIAVEL_POPULACAO}?localidades=N102[N6[{CODIGO_CURITIBA}]]")
    with urllib.request.urlopen(url, timeout=60) as resposta:
        dados = json.load(resposta)
    series = dados[0]["resultados"][0]["series"]
    return [(s["localidade"]["nome"].split(" - ")[0], int(s["serie"][ano])) for s in series]


def main(tabela: str = "9923", ano: str = "2022") -> None:
    linhas = baixar(tabela, ano)
    encontrados = {chave_bairro(nome) for nome, _ in linhas}
    faltando = set(tabela_bairros()["chave"]) - encontrados
    if faltando:
        raise SystemExit(f"O IBGE não trouxe estes bairros (verifique os nomes em src/bairros.py): {faltando}")

    with open(ARQUIVO_POPULACAO, "w", encoding="utf-8", newline="") as arquivo:
        arquivo.write(f"# Fonte: IBGE, Censo Demográfico {ano}, tabela SIDRA {tabela} "
                      f"(população residente por bairro). Baixado em {date.today():%d/%m/%Y}.\n")
        arquivo.write("bairro;populacao\n")
        for nome, populacao in sorted(linhas):
            arquivo.write(f"{nome};{populacao}\n")
    total = sum(p for _, p in linhas)
    print(f"{len(linhas)} bairros salvos em {ARQUIVO_POPULACAO.name}. População total: {total:,}".replace(",", "."))


if __name__ == "__main__":
    main(*sys.argv[1:3])
