# Modulos que necesito
import json
import nltk
from nltk.stem.lancaster import LancasterStemmer

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

print(len(documents), 'documents')
print(len(classes), 'classes', classes)
print(len(words), "stems", words)
