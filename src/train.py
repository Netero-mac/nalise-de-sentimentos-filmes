import pandas as pd
import numpy as np
import joblib
from .data_preprocessing import preprocess_dataframe

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

def train():
    """
    Função principal para treinar e avaliar o modelo usando GridSearchCV
    para otimização de hiperparâmetros.
    """
    # 1. Carregar dados
    print("Carregando dataset...")
    df = pd.read_csv('data/IMDB Dataset.csv')
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

    # 2. Pré-processar os textos
    # NOTA: Para uma busca em grade rápida, vamos usar uma amostra menor do dataset.
    # Para um treino final, comente a linha abaixo para usar todos os 50k de dados.
    df = df.sample(n=10000, random_state=42)
    
    df = preprocess_dataframe(df, 'review')

    X = df['review']
    y = df['sentiment']

    # 3. Definir o Pipeline e a Grade de Hiperparâmetros
    # Um pipeline encadeia o vetorizador e o classificador.
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('svc', LinearSVC(max_iter=2000)),
    ])

    # Parâmetros a serem testados. O GridSearchCV testará todas as combinações.
    # O prefixo (ex: 'tfidf__') refere-se ao nome do passo no pipeline.
    param_grid = {
        'tfidf__ngram_range': [(1, 1), (1, 2)],
        'tfidf__max_df': [0.95, 1.0],
        'svc__C': [0.5, 1.0]
    }

    # 4. Configurar e Executar o GridSearchCV
    # cv=3 -> validação cruzada de 3 folds. n_jobs=-1 -> usa todos os cores de CPU.
    print("Iniciando GridSearchCV para otimização de hiperparâmetros...")
    print("Isto pode demorar vários minutos...")
    
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=1)
    grid_search.fit(X, y)

    # 5. Apresentar os Resultados da Otimização
    print("\nOtimização concluída.")
    print(f"Melhor score (Acurácia Média CV): {grid_search.best_score_:.4f}")
    print("Melhores hiperparâmetros encontrados:")
    print(grid_search.best_params_)

    # 6. Serialização do Melhor Modelo Encontrado
    print("\nGuardando o melhor modelo otimizado em 'models/'...")
    best_model = grid_search.best_estimator_
    joblib.dump(best_model, 'models/best_optimized_model.joblib')
    print("Processo de treino e otimização concluído com sucesso.")


if __name__ == '__main__':
    train()
