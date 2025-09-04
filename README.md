# 🤖 Análise de Sentimentos em Avaliações de Filmes

Repositório do projeto de Machine Learning para classificação de sentimentos em textos, desenvolvido como parte do meu portfólio de projetos em Inteligência Artificial.

## 📜 Descrição

Este projeto implementa um modelo de Machine Learning capaz de realizar análise de sentimentos em avaliações de filmes. O objetivo é ler um texto e classificá-lo automaticamente como **positivo** ou **negativo**. Para isso, foi utilizado um dataset clássico do IMDb contendo 50.000 avaliações.

O desenvolvimento abrange o ciclo completo de um projeto de Processamento de Linguagem Natural (PLN): coleta e limpeza dos dados, extração de features com TF-IDF, treinamento de um modelo classificador e avaliação de sua performance.

---

## ✨ Features

- **Limpeza e Pré-processamento de Texto:** Remoção de pontuação, stop words e conversão para minúsculas para padronizar os dados.
- **Vetorização com TF-IDF:** Converte o texto limpo em uma representação numérica que o modelo consegue entender, dando importância às palavras mais relevantes.
- **Modelo de Classificação:** Utiliza o algoritmo **Multinomial Naive Bayes**, uma escolha eficiente e clássica para problemas de classificação de texto.
- **Avaliação de Performance:** Mede a eficácia do modelo através de métricas como acurácia, precisão e recall.

---

## 📊 Resultados

O modelo final, treinado e testado com dados nunca vistos, alcançou uma **acurácia de 88%**. Este resultado demonstra uma alta capacidade do modelo em generalizar e classificar corretamente novas avaliações de filmes.

**➡️ ATENÇÃO:** Altere o valor de `88%` para a acurácia exata que o seu modelo alcançou no final!

---

## 🛠️ Tecnologias Utilizadas

- **Python 3**
- **Pandas:** Para manipulação e carregamento dos dados.
- **Scikit-learn:** Para o pipeline de Machine Learning, incluindo o `TfidfVectorizer` e o modelo `MultinomialNB`.
- **Jupyter Lab:** Utilizado como ambiente de desenvolvimento para prototipagem e análise interativa.

---

## ⚙️ Como Executar o Projeto

Para executar este projeto em sua máquina local, siga os passos abaixo.

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/Netero-mac/nalise-de-sentimentos-filmes.git](https://github.com/Netero-mac/nalise-de-sentimentos-filmes.git)
    ```

2.  **Navegue até a pasta do projeto:**
    ```bash
    cd nalise-de-sentimentos-filmes
    ```

3.  **Crie e ative o ambiente virtual (para o shell Fish):**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate.fish 
    ```

4.  **Instale as dependências necessárias:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Inicie o Jupyter Lab para explorar o notebook:**
    ```bash
    jupyter lab
    ```

---

## 👤 Autor

**Marco Antonio Cadoso da Cruz Santos**

- **LinkedIn:** `https://linkedin.com/in/[LINKEDIN]` 
- **GitHub:** `https://github.com/Netero-mac`
