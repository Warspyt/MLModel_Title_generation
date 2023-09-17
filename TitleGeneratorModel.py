import tensorflow as tf;
import pandas as pd;
import numpy as np;
import os;
import time;

# Para utilizar Tensor.numpy() más adelante
tf.enable_eager_execution();

# Lectura del dataset para el modelo
data = pd.read_csv('arxiv_data.csv')
data = data.drop(columns=['summaries','terms'])
data = data.sample(frac=1)
data = data[0:500]
data.head()

