import tensorflow as tf
import pandas as pd
import numpy as np
import os

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

""" Red Neuronal Recurrente (RNN) """
# Longitud del vocabulario basado en los caracteres unicos
vocab_size = len(vocab)

embedding_dim = 100
rnn_units = 100

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        tf.keras.layers.GRU(rnn_units,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model
  
model = build_model(
    vocab_size=len(vocab),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
    batch_size=BATCH_SIZE)

model.summary()

""" Funcion de perdida (Machine Learning) para entrenar la RNN """
def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

model.compile(optimizer='adam', loss=loss,run_eagerly=True)
# Directorio para guardar los checkpoints de entrenamiento
checkpoint_dir = './training_checkpoints'
# Archivos de los checkpoints
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,

    save_weights_only=True)

""" Entrenamiento de la RNN """
EPOCHS = 50
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

""" Modelo con la menor valor de perdida """
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))


""" Generación del título haciendo uso del modelo RNN """
def generate_text(model, start_string,t):

    # Caracteres a generar en el titulo
    num_generate = 40

    # Vectorizacion de la keyword
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []

    # Baja temperatura para un título más preciso
    # Alta temperatura para un título menos predecible
    temperature = t

    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)

        # Distribucion categorica para predecir el caracter devuelto por el modelo
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

        # El caracter predecido es la nueva entrada del modelo
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])

    return (start_string + ''.join(text_generated))


""" Generación de títulos de prueba """

titulo_1 = (generate_text(model, start_string=u"Labor Restructuring",t = 0.1))
print(titulo_1)

titulo_2 = (generate_text(model, start_string=u"Concert",t = 0.3))
print(titulo_2)

titulo_3 = (generate_text(model, start_string=u"Earthquake",t = 0.5))
print(titulo_3)

titulo_4 = (generate_text(model, start_string=u"Crisis",t = 0.8))
print(titulo_4)

titulo_5 = (generate_text(model, start_string=u"Virus",t = 1))
print(titulo_5)