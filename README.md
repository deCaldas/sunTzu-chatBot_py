# Chatbot en Python

Este es un proyecto de chatbot en Python que utiliza procesamiento de lenguaje natural (NLP) y aprendizaje profundo para responder a preguntas e interactuar con los usuarios.

## Archivo chatbot.py

El archivo `chatbot.py` contiene la implementación del chatbot en sí. Aquí hay una breve descripción de las principales funciones y características:

- **Clase Chatbot**: Define la clase principal del chatbot.
- **Inicialización**: Carga los datos esenciales para el chatbot, como intenciones, palabras y clases, y carga el modelo de chatbot previamente entrenado.
- **clean_up_sentence**: Tokeniza y lematiza una oración de entrada.
- **bag_of_words**: Crea una bolsa de palabras para una oración.
- **predict_class**: Predice la intención de una oración utilizando el modelo.
- **get_response**: Obtiene una respuesta basada en la intención predecida.
- **run**: Inicia el chatbot y permite al usuario interactuar con él en un bucle infinito.

Para ejecutar el chatbot, simplemente ejecute el archivo `chatbot.py`. El chatbot escuchará sus preguntas y proporcionará respuestas basadas en las intenciones definidas en el archivo `intents.json`.

## Archivo training.py

El archivo `training.py` contiene el código para entrenar el modelo de chatbot. Aquí hay una breve descripción de lo que hace:

- **Descarga de recursos NLTK y carga de spaCy**: Descarga los recursos necesarios para NLTK y carga el modelo de lenguaje de spaCy en español.
- **Funciones de lematización y sinónimos**: Define funciones para lematizar texto y obtener sinónimos de una palabra.
- **Carga de intenciones**: Carga las intenciones desde el archivo `intents.json`.
- **Preparación de datos y entrenamiento**: Tokeniza, lematiza y prepara los datos de entrenamiento, y luego entrena un modelo de red neuronal utilizando Keras.

Ejecuta el archivo `training.py` para entrenar o actualizar el modelo del chatbot.

## Cómo usar

1. Ejecuta `training.py` para entrenar el modelo (o cargar uno preexistente).
2. Ejecuta `chatbot.py` para interactuar con el chatbot.

¡Diviértete interactuando con tu chatbot!

---

Este proyecto es un ejemplo básico de un chatbot en Python. Puedes personalizar las intenciones y las respuestas en el archivo `intents.json` para adaptarlo a tus necesidades específicas.

Si deseas aprender más sobre cómo mejorar y personalizar este chatbot, consulta la documentación de las bibliotecas utilizadas y explora otras técnicas de procesamiento de lenguaje natural y aprendizaje profundo.
