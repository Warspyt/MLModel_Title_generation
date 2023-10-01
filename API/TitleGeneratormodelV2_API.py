from fastapi import FastAPI
from simplet5 import SimpleT5

""" Crear aplicacion """
app = FastAPI()

@app.get("/title-generator")

def generate_Title(summarize: str) -> str:
    # Cargar el modelo entrenado
    model = SimpleT5()
    model.load_model("t5","outputs/simplet5-epoch-0-train-loss-1.5979-val-loss-1.2251", use_gpu=False)
    
    newsTitle = model.predict(summarize)
    
    return newsTitle