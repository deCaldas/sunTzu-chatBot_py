# Importación de las bibliotecas necesarias
import random
import json
import pickle
import numpy as np
import nltk
import spacy
import warnings
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

# Descargar datos necesarios para NLTK
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Definición de la clase Chatbot
class Chatbot:
    def __init__(self):
        # Inicialización de recursos necesarios
        self.lemmatizer = WordNetLemmatizer()
        self.nlp = spacy.load('es_core_news_sm')
        self.intents, self.words, self.classes, self.model = self.load_data()

    def load_data(self):
        # Cargar datos esenciales para el chatbot
        with open('intents.json') as file:
            intents = json.load(file)

        with open('words.pkl', 'rb') as file:
            words = pickle.load(file)

        with open('classes.pkl', 'rb') as file:
            classes = pickle.load(file)

        model = load_model('chatbot_model.h5')

        return intents, words, classes, model

    def clean_up_sentence(self, sentence):
        # Tokenizar y lematizar las palabras de la oración
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [self.lemmatizer.lemmatize(word) for word in sentence_words]
        return sentence_words

    def bag_of_words(self, sentence):
        # Crear una bolsa de palabras para la oración
        sentence_words = self.clean_up_sentence(sentence)
        bag = [0] * len(self.words)
        for i, word in enumerate(self.words):
            if word in sentence_words:
                bag[i] = 1
        return np.array(bag)

    def predict_class(self, sentence):
        # Predecir la intención de la oración
        bow = self.bag_of_words(sentence)
        res = self.model.predict(np.array([bow]))[0]
        max_index = np.argmax(res)
        category = self.classes[max_index]
        return category

    def get_response(self, tag):
        # Obtener una respuesta basada en la intención
        list_of_intents = self.intents['intents']
        for intent in list_of_intents:
            if intent['tag'] == tag:
                responses = intent['responses']
                result = random.choice(responses)
                break
        return result

    def run(self):
        # Ejecutar el chatbot en un bucle infinito
        while True:
            message = input("Ingrese un mensaje: ")
            intent = self.predict_class(message)
            response = self.get_response(intent)
            print(response)

if __name__ == '__main__':
    # Inicializar y ejecutar el chatbot
    chatbot = Chatbot()
    chatbot.run()
