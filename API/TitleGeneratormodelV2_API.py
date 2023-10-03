from fastapi import FastAPI
from simplet5 import SimpleT5
import random as r

""" Crear aplicacion """
app = FastAPI()

@app.get("/title-generator/{summarize}")

def generate_Title(summarize: str) -> list:
    
    try:
        # Cargar el modelo entrenado
        model = SimpleT5()
        model.load_model("t5","outputs/simplet5-epoch-0-train-loss-1.2504-val-loss-1.0521", use_gpu=False)
        
        options = model.predict(summarize)
        titles = options[0].split(".")
        # Lista de titulos sugeridos
        newsTitle = [i for i in titles if (i != "" and len(i) > 10)]
        
    except:
        newsTitle = ["Error: An exception occurred"]
    
    return newsTitle