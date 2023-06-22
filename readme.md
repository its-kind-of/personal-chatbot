# Chatbot Python Files

This repository contains the following Python files:

1. `utils.py`: Contains utility functions for tokenizing and preprocessing text.
2. `model.py`: Defines the NeuralNet class for the chatbot model.
3. `train.py`: Trains the chatbot model using intents data and saves the trained model to a file.
4. `app.py`: Implements a Streamlit web application for interacting with the chatbot.

## File Descriptions

### utils.py

This file contains the following utility functions:

- `tokenize(sentence)`: Splits a sentence into an array of words/tokens.
- `stem(word)`: Stems a word to its root form.
- `bag_of_words(tokenized_sentence, words)`: Creates a bag of words representation for a tokenized sentence.

### model.py

This file defines the `NeuralNet` class, which represents the chatbot model. The model architecture consists of three linear layers with ReLU activation functions.

### train.py

This file trains the chatbot model using the intents data from the `intent.json` file. It preprocesses the data, creates a bag of words representation, and uses the `NeuralNet` model for training. The trained model is saved to a file named `data.pth`.

### app.py

This file implements a Streamlit web application for interacting with the chatbot. It loads the trained model from the `data.pth` file and allows users to input queries. The chatbot responds with appropriate messages based on the trained model's predictions.

Please refer to each file for more details on their implementations.
you can check the app on this website _https://nikhil-chatbot.onrender.com/_
