import torch
import torchvision
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

# Dropout layer


# Compute convolutional layers output size
def out_size(conv_model, dummy_input):
    dummy_output = conv_model(dummy_input)
    return np.prod(dummy_output.shape[1:])


############################
##### Building a Model #####
############################
# Convolutional model class
class convClassifier(nn.Module):
    def __init__(self, model_name, num_classes):

        if model_name == 'custom1':
            super(convClassifier, self).__init__()
            self.conv_model = nn.Sequential(*conv_relu_maxpool(cin=1, cout=8,
                                                               csize=5, cstride=1, cpad=2,
                                                               msize=2, mstride=2, mpad=0),
                                            *conv_relu_maxpool(cin=8, cout=16,
                                                               csize=5, cstride=1, cpad=2,
                                                               msize=2, mstride=2, mpad=0),
                                            *conv_relu_maxpool(cin=16, cout=32,
                                                               csize=5, cstride=1, cpad=2,
                                                               msize=2, mstride=2, mpad=0),
                                            *conv_relu_maxpool(cin=32, cout=64,
                                                               csize=5, cstride=1, cpad=2,
                                                               msize=2, mstride=2, mpad=0))
            print("initiated conv model")
            dummy_input = torch.zeros(1, 1, 300, 300)
            output_size = out_size(self.conv_model, dummy_input)
            print(output_size)
            self.fc_model = nn.Sequential(*linear_relu(output_size, 128, 0.2),
                                          *linear_relu(128, 256, 0.2),
                                          *linear_softmax(256, num_classes))
            print("initiated linear model")

    def forward(self, x):
        x = self.conv_model(x).view(x.size(dim=0), -1)
        y = self.fc_model(x)
        return y
