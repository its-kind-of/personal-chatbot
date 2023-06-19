import streamlit as st
from streamlit_chat import message
import random
import json
import torch
from model import NeuralNet
from utils import bag_of_words, tokenize

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

st.title("Tensorflow Chatbot")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intent.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Nick"
st.write("Let's chat! (type 'quit' to exit)")

    # sentence = "do you use credit cards?"
if __name__ == "__main__":
    user_query = st.text_input(label="Write the query")
    sentence = tokenize(user_query)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                st.write(f"{bot_name}: {random.choice(intent['responses'])}")

    else:
	    st.write(f"{bot_name}: I do not understand...")