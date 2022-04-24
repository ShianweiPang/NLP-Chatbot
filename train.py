import json
from utils import *
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import ChatBotNeuralNet
from matplotlib import pyplot as plt

ignore_words = ['?','!',',','.','(',')']
all_words =[]
tags =[]
# represent the training data (X_train, Y_train)
xy = []

with open('intents.json','r') as f:
    intents = json.load(f)

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for question in intent['questions']:
        # tokenize return an array of string
        words = tokenize(question)
        all_words.extend(words)
        xy.append((words,tag))

all_words = [stem(word) for word in all_words if word not in ignore_words]
all_words = sorted(set(all_words))

tags = sorted(set(tags))

X_train = []
y_train = []

for (question, tag) in xy:
    bag = bagOfWords(question, all_words)

    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label) #Cross Entropy Loss will be used, it uses normal classes but not one-hot


X_train = np.array(X_train)
y_train = np.array(y_train)

class ChatDataset():
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # to access dataset with index
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

# hyperparameters
batch_size = 8
input_size = len(all_words)
hidden_size = 8
output_size = len(tags)
epochs= 1000
learning_rate = 0.002


dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # this is for running with GPU
model = ChatBotNeuralNet(input_size=input_size, hidden_size=hidden_size, num_classes=output_size) #.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#optimizer, optimization strategy-to escape the local minima and to converge quickly
#rule of thumb for optimizer
#1. if you want to keep things simple. use ADAM
#2. if you have time, then use SGD, and tune the learning rate/parameters 
#3. if you are implementing a paper, use the same strategy as what the authors are using 

epoch_loss_hist_m1 =[]
batch_loss_hist_m1 = []
for epoch in range(epochs):
    # model.train() means that we are currently training the model, so some layers will behave accordingly
    model.train()
    loss = 0 

    for input, target in train_loader:
        target = target.to(dtype=torch.long)
        # manually zero the gradients after updating
        optimizer.zero_grad()

        #forward 
        predict_batch = model(input) # forward propagate
        loss_batch = loss_fn(predict_batch, target) # calculate the loss
        
        # backward
        loss_batch.backward() # backward propagate (accumulating the gradients)
        # optimize (i.e. update the weights)
        optimizer.step()

        batch_loss_hist_m1.append(loss_batch.item())
        loss +=loss_batch.item()
    
    epoch_loss_hist_m1.append(loss)
    if (epoch+1)%100==0: 
        print(f'Epoch: {epoch+1}/{epochs}  loss: {loss}')

data = {
    "model_state":model.state_dict(),
    "input_size":input_size,
    "output_size":output_size,
    "hidden_size":hidden_size,
    "all_words":all_words,
    "tags":tags,
}

FILE = "model.pth"
torch.save(data, FILE)


plt.plot(range(epochs), epoch_loss_hist_m1, label="ChatBotNeuralNet 2 Hidden Layers - Epochs")
plt.xlabel("Epoch")
plt.ylabel("Cross Entropy")
plt.legend()
plt.savefig("Epoch results.png")
plt.clf() 

print(f'Training complete, file saved to {FILE}')