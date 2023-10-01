from fastapi import FastAPI
from simplet5 import SimpleT5
import random as r

""" Crear aplicacion """
app = FastAPI()

@app.get("/title-generator/{summarize}")

def generate_Title(summarize: str) -> str:
    
    try:
        # Cargar el modelo entrenado
        model = SimpleT5()
        model.load_model("t5","../outputs/simplet5-epoch-0-train-loss-1.5979-val-loss-1.2251", use_gpu=False)
        
        options = model.predict(summarize)
        titles = options[0].split(".")
        titlesValid = [i for i in titles if i != ""]
        titleSelect = r.randint(0, len(titles) - 1)
        newsTitle = titles[titleSelect]
        
    except:
        newsTitle = "Error: An exception occurred"
    
    return newsTitle