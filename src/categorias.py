"""Agrupa os ~240 assuntos da Central 156 em poucas categorias fáceis de entender.

Cada categoria vira uma dimensão da nota do bairro. Assuntos que não aparecem
aqui caem em "Outros serviços" e não entram na nota.
"""

ZELADORIA = "Zeladoria urbana"
TRANSITO = "Trânsito e mobilidade"
LIMPEZA = "Limpeza e meio ambiente"
SEGURANCA = "Segurança e vulnerabilidade"
SAUDE = "Saúde pública"
COLETA = "Coleta de resíduos"
OUTROS = "Outros serviços"

ASSUNTOS_POR_CATEGORIA = {
    ZELADORIA: [
        "Pavimentação", "Iluminação Pública", "Drenagem", "Macrodrenagem", "Passeio",
        "Cabos Danificados", "Fiscalização", "Acessibilidade", "Praças", "Jardinetes",
        "Pontes", "Passarelas", "Arruamento", "Edificações",
        "Segurança de Edificações e Imóveis", "Mobiliário Urbano", "Novo mobiliário urbano",
        "Bancos em ruas", "Placas de Nomenclatura de Ruas", "Largos", "Eixos de Animação",
    ],
    TRANSITO: [
        "Trânsito", "Semáforo", "Sinalização Horizontal", "Sinalização Vertical",
        "Sinalização de Estacionamento", "Lombada física", "Lombada Eletrônica",
        "Sentido de Tráfego", "Velocidade na Via", "Faixa de Travessia Elevada",
        "Trânsito em Escolas", "Radar de Velocidade", "Correção geométrica - Trânsito",
        "Minirrotatória", "Ciclovia em Ruas", "Sistema viário", "EstaR", "EstaR eletrônico",
        "Infrações de Trânsito", "Sinalização Luminosa e/ou Sonora da Linha Férrea",
        "Inibidor de Avanço de Sinal Vermelho", "Inibidor de Conversão-Retorno Proibido",
        "Inibidor de Parada na Faixa de Pedestre", "Motoristas de ônibus",
        "Cobrador de ônibus", "Ônibus", "Transporte coletivo",
        "Programação do transporte coletivo", "Itinerário do transporte coletivo",
        "Ponto de ônibus", "Estação tubo", "Terminais/Estações tubo (Obras)",
        "Mobilidade Urbana", "Táxi", "Ponto de táxi", "Educação e Cidadania no Trânsito",
    ],
    LIMPEZA: [
        "Limpeza", "Varrição", "Árvore", "Poluição",
        "Fiscalização de terrenos baldios ou edificados", "Fiscalização de Saneamento",
        "Fiscalização de fundo de vale (Córregos/Valas)", "Lixeiras em ruas", "Limpeza geral",
        "Animais", "Rede de Proteção Animal", "Bosques", "Parques",
    ],
    SEGURANCA: [
        "Abordagem Social - Situação de Rua", "Proteção ao cidadão - GM",
        "Proteção ao patrimônio - GM", "Violência contra criança e adolescente", "Invasão",
        "Fiscalização de mocós", "Fiscalização de loteamentos irregulares",
        "Disque Solidariedade",
    ],
    SAUDE: [
        "Dengue", "Vigilância de Zoonoses", "Vigilância Sanitária",
        "Vigilância em Saúde Ambiental", "Unidade de Saúde - UMS",
        "Unidade de Pronto Atendimento 24h - UPA", "SAMU",
        "Centro de Atenção Psicossocial - CAPS",
    ],
    # Pedidos de recolhimento (podas, entulho) são um serviço de rotina, não um problema
    # do bairro. Aparecem nos gráficos, mas ficam fora da nota.
    COLETA: ["Coleta"],
}

# Categorias que viram dimensões da nota (quanto MENOS chamados, melhor)
CATEGORIAS_DA_NOTA = [ZELADORIA, TRANSITO, LIMPEZA, SEGURANCA, SAUDE]
TODAS_CATEGORIAS = CATEGORIAS_DA_NOTA + [COLETA, OUTROS]

# Dicionário invertido: assunto -> categoria
_CATEGORIA_DO_ASSUNTO = {
    assunto: categoria
    for categoria, assuntos in ASSUNTOS_POR_CATEGORIA.items()
    for assunto in assuntos
}


def categoria_do_assunto(assunto: str) -> str:
    """Devolve a categoria de um assunto do 156 (ou "Outros serviços")."""
    return _CATEGORIA_DO_ASSUNTO.get(str(assunto).strip(), OUTROS)
