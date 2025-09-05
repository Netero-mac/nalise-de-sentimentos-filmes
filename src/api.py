import joblib
from fastapi import FastAPI
from pydantic import BaseModel

# 1. Inicialização da aplicação FastAPI
app = FastAPI(
    title="API de Análise de Sentimentos",
    description="Uma API para classificar o sentimento de críticas de filmes.",
    version="1.0.0"
)

# 2. Definição do modelo de dados para o request
# Pydantic garante que o dado recebido tenha o formato esperado.
class ReviewRequest(BaseModel):
    text: str

# 3. Carregamento do modelo no arranque da API
# O modelo é carregado apenas uma vez, otimizando a performance.
try:
    model = joblib.load('models/best_optimized_model.joblib')
    print("Modelo otimizado carregado com sucesso.")
except FileNotFoundError:
    print("Erro: Ficheiro do modelo não encontrado. Treine o modelo primeiro.")
    model = None

# 4. Definição do endpoint de previsão
@app.post("/predict")
def predict_sentiment(request: ReviewRequest):
    """
    Recebe um texto, classifica o seu sentimento e retorna a previsão.
    """
    if model is None:
        return {"error": "Modelo não está carregado. Verifique os logs do servidor."}

    # O pipeline carregado (modelo) faz todo o trabalho:
    # pré-processamento, vetorização e classificação.
    prediction = model.predict([request.text])
    
    sentiment = "Positivo" if prediction[0] == 1 else "Negativo"
    
    return {
        "text": request.text,
        "sentiment": sentiment
    }

# Endpoint raiz para verificação de estado
@app.get("/")
def read_root():
    return {"status": "API de Análise de Sentimentos está operacional."}
