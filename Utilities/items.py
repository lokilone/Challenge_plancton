import torch
import torch.nn as nn
import torch.nn.functional

import numpy as np

####################################
##### Model construction items #####
####################################

# A convolutional base block
def conv_relu_maxpool(cin, cout, csize, cstride, cpad, msize, mstride, mpad):
    return [nn.Conv2d(cin, cout, csize, cstride, cpad),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(msize, mstride, mpad)]

# Linear base blocks
def linear_relu(dim_in, dim_out, p_dropout):
    return [nn.Dropout(p_dropout),
            nn.Linear(dim_in, dim_out),
            nn.ReLU(inplace=True)]


def linear_softmax(dim_in, dim_out):
    return [nn.Linear(dim_in, dim_out),
            nn.Softmax()]

# Dropout layerv (#To add, currently exists within linear)


# Compute convolutional layers output size
def out_size(conv_model, dummy_input):
    dummy_output = conv_model(dummy_input)
    return np.prod(dummy_output.shape[1:])


####################################
## Earlystopping Checkpoint class ##
####################################

class ModelCheckpoint:

    def __init__(self, filepath, model):
        self.min_loss = None
        self.filepath = filepath
        self.model = model

    def update(self, loss):
        if (self.min_loss is None) or (loss < self.min_loss):
            print("Saving a better model")
            torch.save(self.model.state_dict(), self.filepath)
            self.min_loss = loss

####################################
########## Loss functions ##########
####################################

def get_loss(loss_arg):
    if loss_arg == 'f1':
        return F1_Loss()
    elif loss_arg == 'cross_entropy':
        return torch.nn.CrossEntropyLoss()

# F1 loss definition
class F1_Loss(nn.Module):

    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true,):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1

        y_true = nn.functional.one_hot(
            y_true, 86).to(torch.float32)
        y_pred = nn.functional.softmax(y_pred, dim=1)

        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision*recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1-self.epsilon)  # clamp ?
        return 1 - f1.mean()

####################################
############ Optimizers ############
####################################

def get_optimizer(optimizer, model):
    if optimizer == 'adam':
        return torch.optim.Adam(model.parameters())