from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer

app = FastAPI(title="Exara AI API")

# CORS para permitir conexiones desde tu frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cambiar por tu dominio si quieres
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar modelo ligero al iniciar
try:
    model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    raise RuntimeError(f"No se pudo cargar el modelo: {e}")

@app.get("/")
def root():
    return {"message": "Exara AI está corriendo correctamente"}

@app.post("/embed")
def embed(text: str):
    try:
        embedding = model.encode(text).tolist()
        return {"embedding": embedding}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
