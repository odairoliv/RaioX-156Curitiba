"""Gera as bases que acompanham o projeto a partir dos arquivos baixados do portal.

Uso (uma vez só, pela equipe):
    python scripts/preparar_dados.py "C:/Users/<voce>/Downloads"

O que ele faz:
- 156: janeiro a setembro vira a base padrão; outubro vira arquivo de exemplo para
  demonstrar o upload de "novos dados" e o re-treino dos modelos. Uma semana de
  outubro também é salva em JSON e XML para mostrar os outros formatos.
- Equipamentos urbanos: copiado em UTF-8 e compactado.
- E-Saúde: mantém só bairro, município, código do usuário e as colunas de
  moradia/saneamento (descartamos dados pessoais que o app não usa).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from src.config import PASTA_BASE, PASTA_EXEMPLOS  # noqa: E402
from src.leitura import ler_e_identificar  # noqa: E402

ARQUIVOS = {
    "156": "*156_-_Base_de_Dados.csv",
    "equipamentos": "*Unidades*Atendimento_Ativas*.csv",
    "esaude": "*E-Saude*.csv",
}
COLUNAS_ESAUDE = ["Data do Atendimento", "Bairro", "Municício", "cod_usuario", "Abastecimento",
                  "Destino Lixo", "Fezes/Urina", "Tipo de Habitação", "Energia Elétrica"]


def _ler(pasta: Path, padrao: str) -> pd.DataFrame:
    candidatos = sorted(pasta.glob(padrao))
    if not candidatos:
        raise SystemExit(f"Arquivo '{padrao}' não encontrado em {pasta}")
    caminho = candidatos[-1]  # o mais recente (os nomes começam pela data)
    print(f"Lendo {caminho.name}...")
    _, df = ler_e_identificar(caminho.name, caminho.read_bytes())
    return df


def _salvar_csv(df: pd.DataFrame, destino: Path) -> None:
    compressao = "gzip" if destino.suffix == ".gz" else None
    df.to_csv(destino, sep=";", index=False, encoding="utf-8", compression=compressao)
    print(f"  -> {destino.relative_to(destino.parents[2])} ({len(df):,} linhas)")


def main(pasta_downloads: str) -> None:
    pasta = Path(pasta_downloads)
    PASTA_BASE.mkdir(parents=True, exist_ok=True)
    PASTA_EXEMPLOS.mkdir(parents=True, exist_ok=True)

    chamados = _ler(pasta, ARQUIVOS["156"])
    datas = pd.to_datetime(chamados["DataCriacao"], format="%d/%m/%Y", errors="coerce")
    _salvar_csv(chamados[datas < "2026-10-01"], PASTA_BASE / "156_jan_a_set_2026.csv.gz")
    outubro = chamados[datas >= "2026-10-01"]
    _salvar_csv(outubro, PASTA_EXEMPLOS / "156_outubro_2026.csv")

    semana = chamados[(datas >= "2026-10-01") & (datas <= "2026-10-07")]
    semana.to_json(PASTA_EXEMPLOS / "156_semana_01a07_out_2026.json", orient="records",
                   force_ascii=False, indent=1)
    semana.to_xml(PASTA_EXEMPLOS / "156_semana_01a07_out_2026.xml", index=False,
                  root_name="chamados", row_name="chamado")
    print(f"  -> exemplos JSON e XML ({len(semana):,} linhas)")

    equipamentos = _ler(pasta, ARQUIVOS["equipamentos"])
    _salvar_csv(equipamentos, PASTA_BASE / "equipamentos_urbanos_2026.csv.gz")

    esaude = _ler(pasta, ARQUIVOS["esaude"])
    _salvar_csv(esaude[COLUNAS_ESAUDE], PASTA_BASE / "esaude_condicoes_moradia_jul_a_set_2026.csv.gz")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
