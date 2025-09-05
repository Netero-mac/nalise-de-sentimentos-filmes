import pandas as pd
import re
from bs4 import BeautifulSoup
import contractions
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# --- Download de recursos NLTK (executar uma vez) ---
def download_nltk_resources():
    # Adicionada a dependência 'punkt_tab'
    resources = {
        'stopwords': 'corpora/stopwords',
        'punkt': 'tokenizers/punkt',
        'wordnet': 'corpora/wordnet',
        'punkt_tab': 'tokenizers/punkt_tab'
    }
    for resource_name, resource_path in resources.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            print(f"Recurso NLTK '{resource_name}' não encontrado. A fazer download...")
            nltk.download(resource_name)

# --- Funções de Limpeza de Texto ---
def remove_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()

def expand_contractions(text):
    return contractions.fix(text)

def remove_special_characters(text):
    return re.sub(r'[^a-zA-Z\s]', '', text)

def to_lowercase(text):
    return text.lower()

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    return " ".join(filtered_text)

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    word_tokens = word_tokenize(text)
    lemmatized_text = [lemmatizer.lemmatize(word) for word in word_tokens]
    return " ".join(lemmatized_text)

def preprocess_text(text):
    """Pipeline completo de pré-processamento para um único texto."""
    text = remove_html_tags(text)
    text = expand_contractions(text)
    text = remove_special_characters(text)
    text = to_lowercase(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    return text

def preprocess_dataframe(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """Aplica o pipeline de pré-processamento a uma coluna de um DataFrame."""
    print("Iniciando pré-processamento do DataFrame...")
    # Garante que os recursos NLTK estão disponíveis
    download_nltk_resources()
    df[column_name] = df[column_name].apply(preprocess_text)
    print("Pré-processamento concluído.")
    return df
