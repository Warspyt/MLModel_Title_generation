import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from simplet5 import SimpleT5

# ! pip install simplet5 -q

""" Importar dataset de entrenamiento"""
# Dataset 1
df = pd.read_csv("Datasets\\news_summary.csv", encoding='latin-1', usecols=['headlines', 'text'])
df.head()

# Dataset 2
# df = pd.read_csv("Datasets\\news_summary_more.csv", encoding='latin-1', usecols=['headlines', 'text'])
# df.head()

# Dataset 3
# df = pd.read_csv("Datasets\arxiv_data.csv", encoding='latin-1', usecols=['headlines', 'text'])
# df.head()

df = df.rename(columns={"headlines":"target_text", "text":"source_text"})
df = df[['source_text', 'target_text']]
df.head()

""" Añadir 'summarize:' a cada resumen para que el modelo reconozca el source text """
df['source_text'] = "summarize: " + df['source_text']
df

""" Dividir la muestra en datos de entrenamiento y datos de prueba """
train_df, test_df = train_test_split(df, test_size=0.3)
train_df.shape, test_df.shape

""" Importar el modelo listo para entrenar """
model = SimpleT5()
model.from_pretrained(model_type="t5", model_name="t5-base")

""" Entrenar el modelo """
model.train(train_df=train_df,
            eval_df=test_df, 
            source_max_token_len=128, 
            target_max_token_len=50, 
            batch_size=8, 
            max_epochs=3, 
            use_gpu=False
           )

# ! ( cd outputs; ls )

""" Pruebas del modelo """
# Cargar el modelo entrenado
model.load_model("t5","outputs/simplet5-epoch-0-train-loss-1.5979-val-loss-1.2251", use_gpu=False)

""" Texto obtenido de una noticia real """
text_summarize="""The planned five-month-long expedition, due to set off in April 2024, aims to voyage the Amazon’s full length, using modern river-mapping satellite technology to scientifically prove once and for all that the Amazon is not just the world’s most voluminous river, but its longest.

Notably, the Amazon is not one singular stretch of water, but rather part of a greater “river system” spanning much of northern South America. Not dissimilar to the branches of a tree, its network includes multiple sources and tributaries.

The length dispute largely stems from the issue of where the Amazon begins. While Britannica and others have traditionally measured the river as starting from the headwaters of the Apurimac River, in southern Peru, American neuroscientist-turned-river expeditionist James “Rocky” Contos, 51, claims to have discovered a more distant river source – the Mantaro River, in northern Peru – while researching whitewater rafting routes in the country.

“I was aware that the most distant source of the Amazon was considered to be the Apurimac, but when I was gathering all information – maps, hydrographs, etc. – in preparation for my trip to Peru, I realized that another river appeared to be longer,” Contos says.

"""

""" Predecir el titulo para la noticia """
model.predict(text_summarize)