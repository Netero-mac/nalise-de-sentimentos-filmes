# API de Análise de Sentimentos de Filmes v2.1

Uma API de alta performance para análise de sentimentos, construída com FastAPI e um pipeline de Machine Learning otimizado com Scikit-learn para classificar críticas de filmes.

## 📜 Descrição Geral

Este projeto oferece um serviço de API robusto para classificar o sentimento (positivo ou negativo) em textos de críticas de filmes. A aplicação evoluiu de um script de análise exploratória para um sistema modular e pronto para produção.

## ✨ Principais Funcionalidades

  - **Pipeline de Pré-processamento:** Funções otimizadas para limpar e preparar dados de texto para modelagem.
  - **Benchmarking de Modelos:** Um script de treino que avalia sistematicamente múltiplos algoritmos (LinearSVC, Logistic Regression, MultinomialNB) para garantir que apenas o modelo de maior performance seja implementado.
  - **Otimização de Hiperparâmetros:** Uso de `GridSearchCV` para refinar a configuração do modelo campeão.
  - **Serviço de API:** Um endpoint de inferência assíncrono e de alta performance construído com FastAPI.
  - **Documentação Automática:** Interface interativa (Swagger UI) para testar a API diretamente do navegador.

## 🛠️ Stack Tecnológico

  - **Backend:** Python 3.10+
  - **Framework da API:** FastAPI
  - **Servidor ASGI:** Uvicorn
  - **Machine Learning:** Scikit-learn, Pandas
  - **Processamento de Linguagem Natural (PLN):** NLTK
  - **Manipulação de Dados:** Contractions, BeautifulSoup

## 📂 Estrutura do Projeto

```
├── data/
│   └── IMDB Dataset.csv              # Dataset original
├── models/
│   └── best_optimized_model.joblib   # Modelo campeão serializado
├── src/
│   ├── api.py                        # Lógica do servidor FastAPI
│   ├── data_preprocessing.py         # Funções de limpeza de texto
│   ├── predict.py                    # Script para inferência via CLI
│   └── train.py                      # Script de benchmarking e treino
├── .gitignore
├── README.md                         # Esta documentação
└── requirements.txt                  # Dependências do projeto
```

## 🚀 Guia de Instalação e Uso

### 1\. Pré-requisitos

  - Python 3.10 ou superior
  - `git` para clonar o repositório

### 2\. Instalação

Clone o repositório e configure o ambiente virtual:

```bash
# 1. Clone o repositório
git clone https://github.com/Netero-mac/nalise-de-sentimentos-filmes
cd nalise-de-sentimentos-filmes

# 2. Crie e ative um ambiente virtual
python -m venv .venv
# Em macOS/Linux:
source .venv/bin/activate
# Em Windows:
# .venv\Scripts\activate

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Baixe os recursos necessários do NLTK
python -m nltk.downloader stopwords
```

### 3\. Treinamento do Modelo

Antes de executar a API, é necessário treinar os modelos. O script de treino executará um benchmark e salvará automaticamente o modelo de melhor performance.

```bash
# Execute o script de treinamento
python -m src.train
```

Este processo pode demorar alguns minutos. Ao final, o modelo vencedor será salvo em `models/best_optimized_model.joblib`.

### 4\. Execução da API

Com o modelo treinado, inicie o servidor da API com Uvicorn:

```bash
# Inicie o servidor ASGI
uvicorn src.api:app --reload
```

O servidor estará em execução e acessível em `http://127.0.0.1:8000`. A flag `--reload` reinicia o servidor automaticamente após alterações no código.

## 🔌 Endpoints da API

### `GET /`

Endpoint raiz para verificar o estado da API.

  - **Método:** `GET`
  - **Resposta de Sucesso (200):**
    ```json
    {
      "status": "API de Análise de Sentimentos está operacional."
    }
    ```

### `POST /predict`

Endpoint principal para classificação de sentimentos.

  - **Método:** `POST`
  - **Corpo da Requisição (JSON):**
    ```json
    {
      "text": "O texto da sua crítica aqui."
    }
    ```
  - **Resposta de Sucesso (200):**
    ```json
    {
      "text": "O texto da sua crítica aqui.",
      "sentiment": "Positivo"
    }
    ```

-----

### Como Testar

#### 1\. Via Documentação Interativa (Recomendado)

Abra o seu navegador e acesse **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**. A interface do Swagger UI permite testar o endpoint de forma fácil e intuitiva.

#### 2\. Via cURL (Terminal)

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{"text": "This movie was absolutely fantastic and a must-see."}'
```

