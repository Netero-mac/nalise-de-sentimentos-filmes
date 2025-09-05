# Comando para popular o README.md
cat <<'EOL' > README.md
# 🤖 Análise de Sentimentos em Avaliações de Filmes (v2.0)

## 📜 Descrição

Este projeto implementa um modelo de Machine Learning para análise de sentimentos em avaliações de filmes, classificando-as como **positivo** ou **negativo**. O sistema foi refatorizado para uma arquitetura modular e operacional.

## ⚙️ Estrutura do Projeto

- `data/`: Contém o dataset (`IMDB Dataset.csv`).
- `models/`: Armazena o modelo treinado e o vetorizador serializados (`.joblib`).
- `src/`: Código fonte do projeto.
  - `data_preprocessing.py`: Funções para limpeza e pré-processamento de texto.
  - `train.py`: Script para treinar o modelo, avaliá-lo com validação cruzada e guardar os artefactos.
  - `predict.py`: Script para carregar o modelo e fazer previsões em novos textos.
- `requirements.txt`: Lista de dependências Python.

## 🚀 Como Usar

### 1. Configuração do Ambiente

É altamente recomendável usar um ambiente virtual.

```bash
# Criar e ativar ambiente virtual
python -m venv .venv
source .venv/bin/activate

# Instalar dependências
pip install -r requirements.txt