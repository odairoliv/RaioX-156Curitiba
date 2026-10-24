"""Dados compartilhados entre os testes (carregados uma vez só)."""
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.analise import montar_base  # noqa: E402
from src.config import BASE_PADRAO, PASTA_EXEMPLOS  # noqa: E402
from src.leitura import ler_156  # noqa: E402
from src.preparacao import preparar_156  # noqa: E402


@pytest.fixture(scope="session")
def chamados():
    return preparar_156(ler_156(BASE_PADRAO.name, BASE_PADRAO.read_bytes()))[0]


@pytest.fixture(scope="session")
def base(chamados):
    return montar_base(chamados)


@pytest.fixture(scope="session")
def pasta_exemplos():
    return PASTA_EXEMPLOS
