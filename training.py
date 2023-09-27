import random
import json
import pickle
import numpy as np
import spacy
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD

# Descarga los recursos necesarios de NLTK
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Carga el modelo de lenguaje de spaCy
nlp = spacy.load('es_core_news_sm')

# Función para lematizar texto
def lemmatize_text(text):
    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc]
    return " ".join(lemmas)

# Función para obtener sinónimos de una palabra
def get_synonyms(word):
    synonyms = set()
    for syn in word.synsets:
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)

# Inicializa el lematizador de palabras
lemmatizer = WordNetLemmatizer()

# Carga los intents desde el archivo JSON
with open('intents.json') as file:
    intents = json.load(file)

# Inicializa listas para palabras, clases y documentos
words = []
classes = []
documents = []
ignore_letters = ['?', '!', '¿', '.', ',']

# Clasifica los patrones y las categorías
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent["tag"]))
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

# Lematiza y normaliza las palabras
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(set(words))

# Guarda las palabras y las clases en archivos pickle
with open('words.pkl', 'wb') as words_file:
    pickle.dump(words, words_file)

with open('classes.pkl', 'wb') as classes_file:
    pickle.dump(classes, classes_file)

# Crea los datos de entrenamiento
training = []
output_empty = [0] * len(classes)
for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

# Mezcla y convierte los datos de entrenamiento a formato numpy
random.shuffle(training)
training = np.array(training)

# Divide los datos en características (features) y etiquetas (labels)
train_x = list(training[:, 0])
train_y = list(training[:, 1])

# Crea el modelo de red neuronal
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Configura el optimizador y compila el modelo
sgd = SGD(learning_rate=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Entrena el modelo
train_process = model.fit(np.array(train_x), np.array(train_y), epochs=100, batch_size=5, verbose=1)

# Guarda el modelo entrenado
model.save("chatbot_model.h5", train_process)

