import joblib
import sys
from .data_preprocessing import preprocess_text

def predict(text: str):
    """
    Carrega o modelo OTIMIZADO e o vetorizador para fazer uma previsão.
    """
    try:
        # Alterado para carregar o novo modelo otimizado
        model = joblib.load('models/best_optimized_model.joblib')
    except FileNotFoundError:
        print("Erro: Modelo otimizado não encontrado.")
        print("Execute 'python -m src.train' primeiro para treinar e otimizar o sistema.")
        sys.exit(1)

    # O pré-processamento e a previsão são feitos pelo pipeline carregado
    prediction = model.predict([text])
    
    # Interpretar o resultado
    sentiment = "Positivo" if prediction[0] == 1 else "Negativo"
    
    print(f"\nTexto Original: '{text}'")
    print(f"Sentimento Previsto (Modelo Otimizado): {sentiment}")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        input_text = " ".join(sys.argv[1:])
        predict(input_text)
    else:
        print("Uso: python -m src.predict \"<texto da crítica aqui>\"")
