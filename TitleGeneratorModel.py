import tensorflow as tf
import pandas as pd
import numpy as np
import os
import time

""" Para utilizar Tensor.numpy() más adelante """
tf.compat.v1.enable_eager_execution()

""" Lectura del dataset para el modelo """
data = pd.read_csv('arxiv_data.csv')
data = data.drop(columns=['summaries','terms'])
data = data.sample(frac=1)
data = data[0:500]
data.head()


""" Vectorizar el texto """
terms = data.titles.tolist()
text = ''
for t in terms:
    text=text+' ' +t
text = text[1::]
# Caracteres unicos
vocab = sorted(set(text))
print('{} unique characters'.format(len(vocab)))
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text])
# Mapeo de los caracteres con indices
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text])
seq_length = 100
examples_per_epoch = len(text)//(seq_length+1)

# Crear objetivos de entrenamiento
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)

BATCH_SIZE = 10

# Buffer para evitar secuencias infinitas
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

dataset