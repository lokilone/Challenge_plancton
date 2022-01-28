import timm
import torch.nn as nn

# Neural Network blocks and models

class Linear_model(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.linear1 = nn.Linear(input_size, 1650)
        # hidden layers
        self.linear2 = nn.Linear(1650, 512)
        self.linear3 = nn.Linear(512, 138)
        # output layer
        self.linear4 = nn.Linear(138, num_classes)
        
    def forward(self, xb):
        # Flatten images into vectors
        out = xb.view(xb.size(0), -1)
        # Apply layers & activation functions
        out = self.linear1(out)
        # Apply activation function
        out = nn.functional.relu(out)
        # Get intermediate outputs using hidden layer 2
        out = self.linear2(out)
        # Apply activation function
        out = nn.functional.relu(out)
        # Get predictions using output layer
        out = self.linear3(out)
        # Apply activation function
        out = nn.functional.relu(out)
        # Get predictions using output layer
        out = self.linear4(out)
        # Apply activation function
        out = nn.functional.relu(out)
        return out

class Pretrained_CNN(nn.Module):
    def __init__(self,model_choice, num_classes):
        super(Pretrained_CNN, self).__init__()
        self.net =  model_choice
        self.l1 = nn.Linear(1000 , 256)
        self.dropout = nn.Dropout(0.75)
        self.l2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.net(input)
        x = x.view(x.size(0),-1)
        x = self.dropout(self.relu(self.l1(x)))
        x = self.l2(x)
        return x
