# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


def carregar_dados(caminho_csv: str) -> pd.DataFrame:
    df = pd.read_csv(caminho_csv)
    df = df[["review_text_processed", "polarity"]].copy()
    df = df.dropna(subset=["review_text_processed", "polarity"])
    df["review_text_processed"] = df["review_text_processed"].astype(str)
    df["polarity"] = df["polarity"].astype(int)

    return df


def dividir_treino_teste(df: pd.DataFrame, teste_perc: float = 0.2, seed: int = 42):
    X = df["review_text_processed"]
    y = df["polarity"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=teste_perc,
        random_state=seed,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


def criar_classificador_bayesiano() -> Pipeline:
    modelo = Pipeline(
        steps=[
            ("vetorizador", CountVectorizer()),
            ("classificador", MultinomialNB()),
        ]
    )
    return modelo


def treinar_classificador(modelo: Pipeline, X_train, y_train) -> Pipeline:
    modelo.fit(X_train, y_train)
    return modelo


def prever(modelo: Pipeline, X_test):
    y_pred = modelo.predict(X_test)
    return y_pred


if __name__ == "__main__":
    caminho_csv = "data//buscape.csv"
    df = carregar_dados(caminho_csv)

    # Dividir em 80% treino e 20% teste
    X_train, X_test, y_train, y_test = dividir_treino_teste(df, teste_perc=0.2, seed=42)

    # Criar e treinar o classificador bayesiano
    modelo_nb = criar_classificador_bayesiano()
    modelo_nb = treinar_classificador(modelo_nb, X_train, y_train)

    # Obter previsões no conjunto de teste
    y_pred = prever(modelo_nb, X_test)

    # Avaliar desempenho no conjunto de teste
    print("Acurácia no conjunto de teste: {:.4f}".format(accuracy_score(y_test, y_pred)))
    print()
    print("Relatório de classificação:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["negativo (0)", "positivo (1)"],
            digits=4,
        )
    )

    resultados = pd.DataFrame(
        {
            "texto": X_test,
            "classe_real": y_test,
            "classe_prevista": y_pred,
        }
    )

    mapa_sentimento = {0: "negativo", 1: "positivo"}
    resultados["sentimento_real"] = resultados["classe_real"].map(mapa_sentimento)
    resultados["sentimento_previsto"] = resultados["classe_prevista"].map(mapa_sentimento)

    print("\nAlguns exemplos de previsões:")
    print(resultados.head(30))