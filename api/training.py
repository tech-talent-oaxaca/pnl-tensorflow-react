# Modulos que necesito
import json
import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
import tensorflow as tf
import random
import pickle

stemmer = LancasterStemmer()
nltk.download('punkt')

# Abrimos nuestros intents
with open('intents.json') as data:
    intents = json.load(data)

words = []
classes = []
documents = []
ignore_words = ['?']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Arroja los tokens de cada palabra en la frase
        w = nltk.word_tokenize(pattern)

        # Añadimos las palabras a nuestra lista
        words.extend(w)

        # Añadimos los documentos a nuestro corpus
        documents.append((w, intent['tag']))

        # Añadimos a nuestra lista de clases
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# Removemos duplicados
classes = sorted(list(set(classes)))

# print(len(documents), 'documents')
# print(len(classes), 'classes', classes)
# print(len(words), "stems", words)

# Entrenando nuestros datos
training = []
output = []

output_empty = [0] * len(classes)
print(output_empty)

# Bag of words
for doc in documents:
    # Inicializamos nuestra 'bag of words'
    bag = []
    # Lista de tokens
    pattern_words = doc[0]
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]

    for word in words:
        bag.append(1) if word in pattern_words else bag.append(0)

    # la salida es 0 si no está la etiqueta y 1 si si está
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

# creamos nuestras listas de entrenamiento
train_x = list(training[:, 0])
train_y = list(training[:, 1])

# Construimos nuestras redes neuronales
tf.reset_default_graph()

net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Definimos nuestro modelo
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)
model.save('model.tflearn')

# Guardamos todos nuestros datos
pickle.dump( {'words':words, 'classes':classes, 'train_x':train_x, 'train_y':train_y}, open( "training_data", "wb" ) )
