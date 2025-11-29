# Trabalho Final - SI2  

### Desenvolvido por 
#### Caio Cezar Dias, Isabely Toledo de Melo, Rafael Alves Batista, Thaíssa Fernandes Silva e Geraldo Luis de Rezende Júnior

## Classificador Bayesiano para Análise de Sentimentos em Avaliações da Buscapé

Este projeto implementa um Sistema Inteligente baseado em um Classificador Bayesiano (Naive Bayes Multinomial) para realizar análise de sentimentos em avaliações de produtos do site Buscapé.  
O objetivo é classificar automaticamente cada avaliação como pertencente a uma de duas classes:  
- `0` – sentimento negativo  
- `1` – sentimento positivo  

O trabalho segue as diretrizes da disciplina de Sistemas Inteligentes (SI2), contemplando: escolha do problema, escolha e implementação do algoritmo, análise dos resultados e organização do código-fonte.

## 1. Algoritmo escolhido: Classificador Bayesiano (Naive Bayes)

O algoritmo utilizado é o Naive Bayes Multinomial, amplamente empregado em tarefas de classificação de texto devido a:

- Hipótese de independência condicional entre atributos (palavras);
- Bom desempenho mesmo com grandes vocabulários;
- Treinamento rápido e implementação simples;
- Boa interpretação estatística (probabilidades a posteriori).

No código, o modelo é construído com:

- `CountVectorizer` – responsável por transformar o texto em representação numérica (Bag-of-Words);
- `MultinomialNB` – classificador bayesiano para dados discretos (contagem de palavras);
- `Pipeline` – encadeia vetorização e classificação em um fluxo único.

## 2. Conjunto de dados

O conjunto de dados utilizado é o arquivo:

- `buscape.csv`

Este arquivo contém, entre outros, os seguintes atributos relevantes para o projeto:

- `review_text_processed`: texto da avaliação já pré-processado (limpeza, normalização etc.);  
- `polarity`: rótulo numérico do sentimento associado à avaliação (`0` para negativo, `1` para positivo).

O script `main.py` seleciona apenas essas duas colunas, remove valores ausentes e garante a tipagem correta antes da fase de treinamento e teste do modelo.

Observação: O pré-processamento textual (remoção de stopwords, normalização, etc.) é assumido como já realizado previamente no campo `review_text_processed` e não faz parte deste código.

## 3. Estrutura do projeto

Estrutura lógica do projeto:

```text
.
├── data/
│   └── buscape.csv        # Base de dados com avaliações e polaridade
├── main.py                # Código principal com carregamento, treino e avaliação do modelo
└── requirements.txt       # Dependências do projeto (bibliotecas Python)
```

Certifique-se de que o arquivo `buscape.csv` esteja dentro da pasta `data/` conforme indicado acima.

## 4. Descrição do código (`main.py`)

O arquivo `main.py` é organizado com as seguintes funções principais:

1. `carregar_dados(caminho_csv: str) -> pd.DataFrame`  
   - Lê o arquivo CSV com `pandas`;
   - Seleciona as colunas `review_text_processed` e `polarity`;
   - Remove registros com valores nulos nessas colunas;
   - Converte `review_text_processed` para `str` e `polarity` para `int`.

2. `dividir_treino_teste(df: pd.DataFrame, teste_perc: float = 0.2, seed: int = 42)`  
   - Separa a base em treino e teste usando `train_test_split` do `scikit-learn`;
   - Proporção padrão: 80% treino / 20% teste (`teste_perc = 0.2`);
   - Utiliza `stratify=y` para manter a proporção das classes no treino e teste.

3. `criar_classificador_bayesiano() -> Pipeline`  
   - Cria um `Pipeline` com:
     - `CountVectorizer`: vetorização do texto (Bag-of-Words);  
     - `MultinomialNB`: classificador Naive Bayes Multinomial.

4. `treinar_classificador(modelo: Pipeline, X_train, y_train) -> Pipeline`  
   - Treina o modelo com os dados de treino.

5. `prever(modelo: Pipeline, X_test)`  
   - Gera as previsões do modelo para o conjunto de teste.

Na execução principal, o script:

1. Carrega a base `data/buscape.csv`;
2. Divide os dados em treino e teste;
3. Cria e treina o classificador bayesiano;
4. Realiza previsões no conjunto de teste;
5. Calcula e imprime a acurácia e o relatório de classificação (`classification_report`);
6. Cria um `DataFrame` com exemplos de textos, classe real, classe prevista e sentimento textual correspondente.

## 5. Dependências e ambiente

As dependências principais estão listadas em `requirements.txt`:

```text
pandas
scikit-learn
```

### Versão recomendada do Python

- Python 3.10 ou superior (ou outra versão compatível com as bibliotecas acima).

### Configuração de ambiente virtual (opcional, mas recomendado)

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
# .venv\Scripts\activate     # Windows (PowerShell ou CMD)
```

Instalação das dependências:

```bash
pip install -r requirements.txt
```

## 6. Como executar o projeto

1. Certifique-se de que:

   - O arquivo `buscape.csv` esteja dentro da pasta `data/` no mesmo diretório de `main.py`;
   - As dependências de `requirements.txt` estejam instaladas.

2. Execute o script principal:

```bash
python main.py
```

3. Ao final da execução, serão exibidos no terminal:

   - Acurácia no conjunto de teste;  
   - Relatório de classificação contendo precisão, revocação e F1-score por classe;  
   - Uma amostra de previsões com:
     - texto da avaliação,
     - classe real,
     - classe prevista,
     - sentimentos mapeados para `negativo` ou `positivo`.
