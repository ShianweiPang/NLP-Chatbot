import torch 
import torch.nn as nn
import torch.nn.functional as F

class ChatBotNeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(ChatBotNeuralNet, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size,hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size,num_classes),
        )
        

    def forward(self,x):
        output = self.layers(x)

        # self.training 
        # model.train() tells your model that you are training the model. 
        # So effectively layers like dropout, batchnorm etc. which behave 
        # different on the train and test procedures know what is going on 
        # and hence can behave accordingly.
        if not self.training:
            output = F.softmax(output, dim=1)

        return output