# API de Análise de Sentimentos de Filmes v2.1

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange)

Uma API de alta performance para análise de sentimentos, construída com FastAPI e um pipeline de Machine Learning otimizado com Scikit-learn.

## 📜 Descrição Geral

Este projeto fornece um serviço de API robusto para classificar o sentimento (positivo ou negativo) de textos de críticas de filmes. A aplicação evoluiu de um script de análise exploratória para um sistema modular e operacional, que inclui:

-   **Pipeline de Pré-processamento:** Funções otimizadas para limpar e preparar dados de texto para modelagem.
-   **Benchmarking de Modelos:** Um script de treino que avalia sistematicamente múltiplos algoritmos (LinearSVC, Logistic Regression, MultinomialNB) para garantir que apenas o modelo de maior performance seja implementado.
-   **Otimização de Hiperparâmetros:** Uso de `GridSearchCV` para refinar a configuração do modelo campeão.
-   **Serviço de API:** Um endpoint de inferência assíncrono e de alta performance construído com FastAPI.
-   **Documentação Automática:** Interface interativa (Swagger UI) para testar a API diretamente do navegador.

## 🛠️ Stack Tecnológico

-   **Backend:** Python 3.10+
-   **API Framework:** FastAPI
-   **Servidor ASGI:** Uvicorn
-   **Machine Learning:** Scikit-learn, Pandas
-   **Processamento de Linguagem Natural:** NLTK
-   **Manipulação de Dados:** Contractions, BeautifulSoup

## 📂 Estrutura do Projeto

├── data/
│   └── IMDB Dataset.csv      # Dataset original
├── models/
│   └── best_optimized_model.joblib # Modelo campeão serializado
├── src/
│   ├── api.py                # Lógica do servidor FastAPI
│   ├── data_preprocessing.py # Funções de limpeza de texto
│   ├── predict.py            # Script para inferência via CLI
│   └── train.py              # Script de benchmarking e treino
├── .gitignore
├── README.md                 # Esta documentação
└── requirements.txt          # Dependências do projeto

## 🚀 Guia de Instalação e Uso

### 1. Pré-requisitos

-   Python 3.10 ou superior
-   `git` para clonar o repositório

### 2. Instalação

Clone o repositório e configure o ambiente virtual:

```bash
# 1. Clone o repositório
git clone <URL_DO_SEU_REPOSITORIO>
cd <NOME_DO_DIRETORIO>

# 2. Crie e ative um ambiente virtual
python -m venv .venv
source .venv/bin/activate
# Para Windows: .venv\Scripts\activate

# 3. Instale as dependências necessárias
pip install -r requirements.txt

###3. Treino do Modelo
Antes de executar a API, é necessário treinar os modelos. O script de treino executará um benchmark e guardará automaticamente o modelo de melhor performance.

Bash

python -m src.train
Este processo pode demorar alguns minutos. Ao final, o modelo vencedor será salvo em models/best_optimized_model.joblib.

###4. Execução da API
Com o modelo treinado, inicie o servidor da API com Uvicorn:

Bash

uvicorn src.api:app --reload
O servidor estará em execução e acessível em http://127.0.0.1:8000. A flag --reload reinicia o servidor automaticamente após alterações no código.

🔌 Endpoints da API
POST /predict
Este é o endpoint principal para classificação de sentimentos.

Método: POST

Corpo da Requisição (JSON):

JSON

{
  "text": "O texto da sua crítica aqui."
}
Resposta de Sucesso (JSON):

JSON

{
  "text": "O texto da sua crítica aqui.",
  "sentiment": "Positivo"
}
Como Testar:
1. Via Documentação Interativa (Recomendado):

Abra o seu navegador e aceda a http://127.0.0.1:8000/docs. A interface Swagger UI permite testar o endpoint de forma fácil e intuitiva.

2. Via cURL (Terminal):

Bash

curl -X 'POST' \
  '[http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict)' \
  -H 'Content-Type: application/json' \
  -d '{"text": "This movie was absolutely fantastic and a must-see."}'
GET /
Endpoint raiz para verificar o estado da API.

Método: GET

Resposta de Sucesso (JSON):

JSON

{
  "status": "API de Análise de Sentimentos está operacional."
}
