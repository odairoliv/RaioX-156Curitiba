"""Tela de upload: quando falta coluna opcional, o usuário pode cancelar ou prosseguir.

Usa o AppTest do Streamlit, que roda a página "de verdade" e permite clicar nos botões.
"""
from streamlit.testing.v1 import AppTest

from src import estado

ARQUIVO_SEM_OPCIONAIS = ("DataCriacao;DataResposta;Assunto;Bairro;Situacao;Orgao;Tipo;Origem\n"
                         "02/10/2026;05/10/2026;Árvore;Centro;Concluído;SMMA;Solicitação;Telefone\n").encode("utf-8")


def _pagina():
    import runpy

    from src import estado
    estado.iniciar()
    runpy.run_path("paginas/enviar_dados.py")


def _abrir_com_arquivo_pendente() -> AppTest:
    app = AppTest.from_function(_pagina, default_timeout=120)
    app.run()
    app.session_state["uploads_pendentes"] = [estado.analisar_upload("sem_opcionais.csv", ARQUIVO_SEM_OPCIONAIS)]
    app.run()
    return app


def _botao(app: AppTest, rotulo: str):
    return next(b for b in app.button if b.label == rotulo)


def test_aviso_aparece_e_cancelar_nao_muda_a_base():
    app = _abrir_com_arquivo_pendente()
    assert any("colunas opcionais" in w.value for w in app.warning)
    linhas_antes = len(app.session_state["chamados"])

    _botao(app, "Cancelar").click().run()

    assert "uploads_pendentes" not in app.session_state
    assert len(app.session_state["chamados"]) == linhas_antes
    assert any("cancelado" in i.value for i in app.info)


def test_prosseguir_carrega_o_arquivo_mesmo_sem_as_opcionais():
    app = _abrir_com_arquivo_pendente()
    linhas_antes = len(app.session_state["chamados"])

    _botao(app, "Prosseguir assim mesmo").click().run()

    assert "uploads_pendentes" not in app.session_state
    assert len(app.session_state["chamados"]) == linhas_antes + 1
    assert any("Regional preenchida pelo bairro" in s.value for s in app.success)
