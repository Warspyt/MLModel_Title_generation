# Librerias a utilizar
import pandas as pd;
import string;
import numpy as np;
import json;
from keras.preprocessing.sequence import pad_sequences;
from keras.layers import Embedding, LSTM, Dense, Dropout;
from keras.preprocessing.text import Tokenizer;
from keras.callbacks import EarlyStopping;
from keras.models import Sequential;
import keras.utils as ku;
import tensorflow as tf;
tf.random.set_seed(2);
from numpy.random import seed;
seed(1);

