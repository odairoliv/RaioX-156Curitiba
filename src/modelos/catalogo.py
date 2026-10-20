"""Catálogo dos algoritmos que o usuário pode escolher e dos parâmetros ajustáveis.

Cada parâmetro é descrito por um dicionário. A tela lê esse catálogo e cria os
controles (sliders/caixas) automaticamente, então para oferecer um algoritmo novo
basta acrescentá-lo aqui.
"""
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def _inteiro(nome, rotulo, minimo, maximo, padrao, ajuda=""):
    return {"nome": nome, "rotulo": rotulo, "tipo": "int", "min": minimo, "max": maximo, "padrao": padrao, "ajuda": ajuda}


def _decimal(nome, rotulo, minimo, maximo, padrao, passo, ajuda=""):
    return {"nome": nome, "rotulo": rotulo, "tipo": "float", "min": minimo, "max": maximo, "padrao": padrao,
            "passo": passo, "ajuda": ajuda}


def _opcao(nome, rotulo, opcoes, padrao, ajuda=""):
    return {"nome": nome, "rotulo": rotulo, "tipo": "opcao", "opcoes": opcoes, "padrao": padrao, "ajuda": ajuda}


PROFUNDIDADE = _inteiro("max_depth", "Profundidade máxima da árvore", 1, 30, 8,
                        "Quantas perguntas em sequência a árvore pode fazer. Muito alto = decora os dados (overfitting).")
FOLHA = _inteiro("min_samples_leaf", "Mínimo de exemplos por folha", 1, 100, 5,
                 "Cada resposta final da árvore precisa de pelo menos este número de exemplos.")
VIZINHOS = _inteiro("n_neighbors", "Número de vizinhos (K)", 1, 50, 7,
                    "Quantos exemplos parecidos o KNN consulta para decidir.")
PESO_VIZINHOS = _opcao("weights", "Peso dos vizinhos", ["uniform", "distance"], "distance",
                       "'distance' dá mais importância aos vizinhos mais parecidos.")
ARVORES = _inteiro("n_estimators", "Número de árvores", 10, 300, 100,
                   "Quantas árvores a floresta combina. Mais árvores = mais estável, porém mais lento.")
BALANCEAR = _opcao("class_weight", "Balancear classes", ["balanced", "nenhum"], "balanced",
                   "Compensa o fato de haver menos chamados atrasados que no prazo.")

MODELOS_REGRESSAO = {
    "Regressão Linear": {"classe": LinearRegression, "parametros": [],
                         "descricao": "Traça a 'reta' que melhor liga as semanas anteriores à próxima. Simples e fácil de explicar."},
    "Ridge (Linear regularizada)": {"classe": Ridge, "parametros": [
        _decimal("alpha", "Força da regularização (alpha)", 0.01, 100.0, 1.0, 0.01,
                 "Quanto maior, mais 'conservador' o modelo fica.")],
        "descricao": "Regressão linear com um freio que evita coeficientes exagerados."},
    "Árvore de Decisão": {"classe": DecisionTreeRegressor, "parametros": [PROFUNDIDADE, FOLHA],
                          "descricao": "Faz perguntas do tipo 'semana passada teve mais de 30 chamados?' até chegar a um valor."},
    "KNN": {"classe": KNeighborsRegressor, "parametros": [VIZINHOS, PESO_VIZINHOS],
            "descricao": "Procura situações parecidas no passado e usa a média do que aconteceu depois."},
    "Random Forest": {"classe": RandomForestRegressor, "parametros": [ARVORES, PROFUNDIDADE, FOLHA],
                      "descricao": "Várias árvores de decisão votam juntas. Costuma ser mais preciso que uma árvore só."},
    "Gradient Boosting": {"classe": GradientBoostingRegressor, "parametros": [
        ARVORES, _inteiro("max_depth", "Profundidade de cada árvore", 1, 10, 3),
        _decimal("learning_rate", "Taxa de aprendizado", 0.01, 1.0, 0.1, 0.01,
                 "Quanto cada nova árvore corrige as anteriores.")],
        "descricao": "Árvores criadas em sequência, cada uma corrigindo os erros da anterior."},
}

MODELOS_CLASSIFICACAO = {
    "Árvore de Decisão": {"classe": DecisionTreeClassifier, "parametros": [PROFUNDIDADE, FOLHA, BALANCEAR],
                          "descricao": "Regras do tipo 'se o assunto é X e o órgão é Y, então atrasa'. Dá para visualizar as regras."},
    "KNN": {"classe": KNeighborsClassifier, "parametros": [VIZINHOS, PESO_VIZINHOS],
            "descricao": "Olha os K chamados mais parecidos e vê se a maioria atrasou."},
    "Random Forest": {"classe": RandomForestClassifier, "parametros": [ARVORES, PROFUNDIDADE, FOLHA, BALANCEAR],
                      "descricao": "Várias árvores votam. Mais robusto que uma árvore sozinha."},
    "Regressão Logística": {"classe": LogisticRegression, "parametros": [
        _decimal("C", "Inverso da regularização (C)", 0.01, 10.0, 1.0, 0.01,
                 "Valores menores deixam o modelo mais simples."), BALANCEAR],
        "descricao": "Calcula uma probabilidade de atraso somando o 'peso' de cada característica."},
    "Naive Bayes": {"classe": BernoulliNB, "parametros": [
        _decimal("alpha", "Suavização (alpha)", 0.01, 5.0, 1.0, 0.01,
                 "Evita probabilidade zero para combinações nunca vistas.")],
        "descricao": "Usa estatística (teorema de Bayes) assumindo que as características são independentes."},
}


def criar_modelo(catalogo: dict, nome: str, parametros: dict, semente: int = 42):
    """Instancia o algoritmo escolhido com os parâmetros da tela."""
    classe = catalogo[nome]["classe"]
    params = {k: (None if v == "nenhum" else v) for k, v in parametros.items()}
    if "random_state" in classe().get_params():
        params["random_state"] = semente
    if classe is LogisticRegression:
        params["max_iter"] = 1000
    elif "n_jobs" in classe().get_params():
        params["n_jobs"] = -1
    return classe(**params)


def parametros_padrao(catalogo: dict, nome: str) -> dict:
    return {p["nome"]: p["padrao"] for p in catalogo[nome]["parametros"]}
