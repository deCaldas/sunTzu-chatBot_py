import random
import json
import pickle
import numpy as np
import nltk
import spacy
import warnings
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

# Silenciar advertencias de TensorFlow
warnings.filterwarnings("ignore", category=FutureWarning)

# Descargar datos necesarios para NLTK
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

class Chatbot:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.nlp = spacy.load('es_core_news_sm')
        self.intents, self.words, self.classes, self.model = self.load_data()

    def load_data(self):
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
        sentence_words = self.clean_up_sentence(sentence)
        bag = [0] * len(self.words)
        for i, word in enumerate(self.words):
            if word in sentence_words:
                bag[i] = 1
        return np.array(bag)

    def predict_class(self, sentence):
        bow = self.bag_of_words(sentence)
        res = self.model.predict(np.array([bow]))[0]
        max_index = np.argmax(res)
        category = self.classes[max_index]
        return category

    def get_response(self, tag):
        list_of_intents = self.intents['intents']
        for intent in list_of_intents:
            if intent['tag'] == tag:
                responses = intent['responses']
                result = random.choice(responses)
                break
        return result

    def run(self):
        while True:
            message = input("Ingrese un mensaje: ")
            intent = self.predict_class(message)
            response = self.get_response(intent)
            print(response)

if __name__ == '__main__':
    chatbot = Chatbot()
    chatbot.run()

