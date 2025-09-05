import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
import joblib
import numpy as np
from .data_preprocessing import preprocess_dataframe

def train():
    """Função principal para treinar e avaliar o modelo."""
    # 1. Carregar dados
    print("Carregando dataset...")
    df = pd.read_csv('data/IMDB Dataset.csv')

    # Mapear sentimentos para valores numéricos
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

    # 2. Pré-processar os textos
    df = preprocess_dataframe(df, 'review')

    X = df['review']
    y = df['sentiment']

    # 3. Vetorização e Modelação
    print("Iniciando vetorização e treino...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    model = LinearSVC(random_state=42, max_iter=1000)

    # 4. Avaliação Robusta com Validação Cruzada
    print("Executando validação cruzada...")
    X_vectorized = vectorizer.fit_transform(X) 
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_vectorized, y, cv=cv, scoring='accuracy')

    print(f"\nAcurácia da Validação Cruzada (5-folds):")
    print(f"Scores: {scores}")
    print(f"Média: {np.mean(scores):.4f}")
    print(f"Desvio Padrão: {np.std(scores):.4f}\n")

    # 5. Treino Final no Dataset Completo
    print("Treinando modelo final com todos os dados...")
    model.fit(X_vectorized, y)

    # 6. Serialização dos Artefactos
    print("Guardando o modelo e o vetorizador em 'models/'...")
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.joblib')
    joblib.dump(model, 'models/linear_svc_model.joblib')
    print("Processo de treino concluído com sucesso.")

if __name__ == '__main__':
    train()
