import random
import json
import torch
from model import ChatBotNeuralNet
from utils import *

with open('intents.json','r') as f:
    intents = json.load(f)

FILE = "model.pth"
data = torch.load(FILE)

input_size = data['input_size']
hidden_size = data['hidden_size']
output_size = data['output_size']
all_words = data['all_words']
tags = data['tags']
model_state = data['model_state']

bot_name = "Magi"

model = ChatBotNeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()


while True:
    sentence=input()

    sentence = tokenize(sentence)
    X = bagOfWords(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    proba, predicted = torch.max(output, dim=1)
    # print(output)
    # print(predicted)
    tag = tags[predicted.item()]
    print(tag)

    if proba.item() > 0.9:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(f"{bot_name}: {intent['response']} - {proba.item()}")
    else:
        print(f"{bot_name}: I do not understand... - {proba.item()}")


