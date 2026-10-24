"""Gera a base que acompanha o projeto a partir do arquivo do 156 baixado do portal.

Uso (uma vez só, pela equipe):
    python scripts/preparar_dados.py "C:/Users/<voce>/Downloads"

O que ele faz:
- janeiro a setembro vira a base padrão (data/base/, compactada);
- outubro vira arquivo de exemplo (data/exemplos/) para demonstrar o envio de
  dados novos e o re-treino dos modelos;
- a primeira semana de outubro também é salva em JSON e XML, para mostrar os outros formatos.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.config import BASE_PADRAO, PASTA_EXEMPLOS  # noqa: E402
from src.leitura import ler_156  # noqa: E402


def main(pasta_downloads: str) -> None:
    candidatos = sorted(Path(pasta_downloads).glob("*156_-_Base_de_Dados.csv"))
    if not candidatos:
        raise SystemExit(f"Arquivo '*156_-_Base_de_Dados.csv' não encontrado em {pasta_downloads}")
    caminho = candidatos[-1]  # o mais recente (os nomes começam pela data)
    print(f"Lendo {caminho.name}...")
    chamados = ler_156(caminho.name, caminho.read_bytes())

    BASE_PADRAO.parent.mkdir(parents=True, exist_ok=True)
    PASTA_EXEMPLOS.mkdir(parents=True, exist_ok=True)
    datas = pd.to_datetime(chamados["DataCriacao"], format="%d/%m/%Y", errors="coerce")

    base = chamados[datas < "2026-10-01"]
    base.to_csv(BASE_PADRAO, sep=";", index=False, encoding="utf-8", compression="gzip")
    print(f"  -> {BASE_PADRAO.name} ({len(base):,} linhas)")

    outubro = chamados[datas >= "2026-10-01"]
    outubro.to_csv(PASTA_EXEMPLOS / "156_outubro_2026.csv", sep=";", index=False, encoding="utf-8")
    print(f"  -> 156_outubro_2026.csv ({len(outubro):,} linhas)")

    semana = chamados[(datas >= "2026-10-01") & (datas <= "2026-10-07")]
    semana.to_json(PASTA_EXEMPLOS / "156_semana_01a07_out_2026.json", orient="records", force_ascii=False, indent=1)
    semana.to_xml(PASTA_EXEMPLOS / "156_semana_01a07_out_2026.xml", index=False, root_name="chamados",
                  row_name="chamado")
    print(f"  -> exemplos JSON e XML ({len(semana):,} linhas)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
