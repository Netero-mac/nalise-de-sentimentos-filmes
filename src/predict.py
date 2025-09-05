import joblib
import sys
from .data_preprocessing import preprocess_text

def predict(text: str):
    """Carrega o modelo e o vetorizador para fazer uma previsão."""
    try:
        vectorizer = joblib.load('models/tfidf_vectorizer.joblib')
        model = joblib.load('models/linear_svc_model.joblib')
    except FileNotFoundError:
        print("Erro: Modelo ou vetorizador não encontrado.")
        print("Execute 'python -m src.train' primeiro para treinar o sistema.")
        sys.exit(1)

    # Pré-processar o input do utilizador
    processed_text = preprocess_text(text)
    
    # Vetorizar
    vectorized_text = vectorizer.transform([processed_text])
    
    # Prever
    prediction = model.predict(vectorized_text)
    
    # Interpretar o resultado
    sentiment = "Positivo" if prediction[0] == 1 else "Negativo"
    
    print(f"\nTexto Original: '{text}'")
    print(f"Sentimento Previsto: {sentiment}")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        # Pega todo o texto após o nome do script como um único argumento
        input_text = " ".join(sys.argv[1:])
        predict(input_text)
    else:
        print("Uso: python -m src.predict \"<texto da crítica aqui>\"")
