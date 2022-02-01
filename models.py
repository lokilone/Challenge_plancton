#import timm
import torch.nn as nn
import torch

# Neural Network blocks and models
## Custom CNN 
def dropout_linear_relu(dim_in, dim_out, p_drop):
    return [nn.Dropout(p_drop),
            nn.Linear(dim_in, dim_out),
            nn.ReLU(inplace=True)]


def conv_relu_maxp(in_channels, out_channels, ks):
    return [nn.Conv2d(in_channels, out_channels,
                      kernel_size=ks,
                      stride=1,
                      padding=int((ks-1)/2), bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)]


class Custom_CNN(nn.Module):
    def __init__(self, num_classes):
        super(Custom_CNN, self).__init__()

        # By default, Linear layers and Conv layers use Kaiming He initialization

        self.features = nn.Sequential(
            *conv_relu_maxp(1, 16, 5),
            *conv_relu_maxp(16, 32, 5),
            *conv_relu_maxp(32, 64, 5)
        )
        # You must compute the number of features manualy to instantiate the
        # next FC layer
        # self.num_features = 64*3*3

        # Or you create a dummy tensor for probing the size of the feature maps
        probe_tensor = torch.zeros((1,1,300,300))
        out_features = self.features(probe_tensor).view(-1)

        self.classifier = nn.Sequential(
            *dropout_linear_relu(out_features.shape[0], 128, 0.5),
            *dropout_linear_relu(128, 256, 0.5),
            nn.Linear(256,num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size()[0], -1)  #  OR  x = x.view(-1, self.num_features)
        y = self.classifier(x)
        return y
## Custom CNN over

def build_model(model_name, img_size, num_classes):
    model = None
    if(model_name == "custom"):
        model = Custom_CNN(num_classes)
    else:
        raise NotImplementedError("Unknown model {}".format(model_name))
    return model
