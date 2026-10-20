"""Dados compartilhados entre os testes (carregados uma vez só)."""
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.config import BASES_PADRAO, PASTA_EXEMPLOS  # noqa: E402
from src.leitura import ler_e_identificar  # noqa: E402
from src.preparacao import PREPARADORES  # noqa: E402


def _carregar(caminho: Path):
    tipo, bruto = ler_e_identificar(caminho.name, caminho.read_bytes())
    return PREPARADORES[tipo](bruto)[0]


@pytest.fixture(scope="session")
def bases():
    return {tipo: _carregar(Path(caminho)) for tipo, caminho in BASES_PADRAO.items()}


@pytest.fixture(scope="session")
def pasta_exemplos():
    return PASTA_EXEMPLOS
