# 🤖 API de Análise de Sentimentos v2.1 (Otimizada)

## 📜 Descrição

Este projeto implementa um serviço de API de alta performance para análise de sentimentos em avaliações de filmes, classificando-as como **positivo** ou **negativo**.

O sistema evoluiu de um script monolítico para uma arquitetura modular e robusta, que inclui:
-   **Otimização de Hiperparâmetros:** Utilização de `GridSearchCV` para encontrar a melhor configuração de modelo.
-   **Benchmarking Competitivo:** Um pipeline de treino que avalia sistematicamente múltiplos algoritmos para garantir que apenas o modelo de maior performance seja implementado.
-   **API Operacional:** Exposição do modelo através de um endpoint `POST /predict` utilizando FastAPI.

## ⚙️ Estrutura do Projeto

-   `data/`: Contém o dataset (`IMDB Dataset.csv`).
-   `models/`: Armazena o modelo campeão serializado (`best_optimized_model.joblib`).
-   `src/`: Código fonte do projeto.
    -   `data_preprocessing.py`: Funções para limpeza e pré-processamento de texto.
    -   `train.py`: Script de benchmarking para treinar, avaliar e selecionar o melhor modelo.
    -   `predict.py`: Script de inferência via linha de comando (CLI).
    -   `api.py`: Script do servidor da API (FastAPI).
-   `requirements.txt`: Lista de dependências Python.

## 🚀 Como Usar

### 1. Configuração do Ambiente

É altamente recomendável usar um ambiente virtual.

```bash
# Criar e ativar ambiente virtual (exemplo para fish shell)
python -m venv .venv
source .venv/bin/activate.fish

# Instalar dependências
pip install -r requirements.txt
