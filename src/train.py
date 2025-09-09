import pandas as pd
import numpy as np
import joblib
import time
from .data_preprocessing import preprocess_dataframe

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

def run_benchmarking():
    """
    Executa um benchmarking completo de múltiplos modelos, salva o melhor e
    imprime um relatório comparativo.
    """
    # 1. Carregar e preparar os dados
    print("Carregando dataset...")
    df = pd.read_csv('data/IMDB Dataset.csv')
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    
    # Usar uma amostra para um benchmarking mais rápido. Comente para usar o dataset completo.
    df = df.sample(n=10000, random_state=42)
    
    df = preprocess_dataframe(df, 'review')
    X = df['review']
    y = df['sentiment']

    # 2. Definir os modelos (pipelines) para o benchmarking
    pipelines = {
        "Logistic Regression": Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_df=0.95)),
            ('clf', LogisticRegression(random_state=42, max_iter=1000))
        ]),
        "Linear SVC": Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_df=0.95)),
            ('clf', LinearSVC(C=1.0, random_state=42, max_iter=2000))
        ]),
        "Multinomial NB": Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_df=0.95)),
            ('clf', MultinomialNB())
        ])
    }

    # 3. Executar o benchmarking
    print("\nIniciando benchmarking de modelos...")
    results = []
    
    for name, pipeline in pipelines.items():
        print(f"Avaliando {name}...")
        start_time = time.time()
        
        # Usar validação cruzada para uma avaliação robusta
        scores = cross_val_score(pipeline, X, y, cv=3, scoring='accuracy', n_jobs=-1)
        
        end_time = time.time()
        
        results.append({
            "model": name,
            "mean_accuracy": np.mean(scores),
            "std_accuracy": np.std(scores),
            "training_time": end_time - start_time
        })

    # 4. Apresentar o relatório de resultados
    results_df = pd.DataFrame(results).sort_values(by='mean_accuracy', ascending=False)
    
    print("\n--- Relatório de Benchmarking ---")
    print(results_df.to_string(index=False))
    print("---------------------------------\n")

    # 5. Identificar e treinar o modelo vencedor no dataset completo
    winner_name = results_df.iloc[0]['model']
    winner_pipeline = pipelines[winner_name]
    
    print(f"Modelo vencedor: {winner_name}. Treinando no dataset completo...")
    winner_pipeline.fit(X, y)

    # 6. Salvar o modelo vencedor
    print("Guardando o modelo vencedor em 'models/'...")
    joblib.dump(winner_pipeline, 'models/best_optimized_model.joblib')
    print("Processo de benchmarking e treino concluído com sucesso.")

if __name__ == '__main__':
    run_benchmarking()
