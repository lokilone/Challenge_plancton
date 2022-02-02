import torch
import argparse
import logging
import pathlib
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch.nn as nn
import torch.nn.functional
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import pandas as pd
from tqdm import tqdm

import os.path
import sys
import time

###############################
##### Defining transforms #####
###############################

# Images are B/W but in 3 channels, we only need one
greyscale = torchvision.transforms.Grayscale(num_output_channels=1)
# Negative
invert = torchvision.transforms.functional.invert
# Resize
resize = torchvision.transforms.Resize((300,300))
# Normalization
normalization = torchvision.transforms.Normalize(mean=[0.0988],std=[0.1444])


### Data Augmentation
rotate = torchvision.transforms.RandomRotation((0, 360))
flip = torchvision.transforms.RandomHorizontalFlip(p=0.5)
flou = torchvision.transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 1))


# Compose transforms
test_composed_transforms = torchvision.transforms.Compose([greyscale, invert, resize, torchvision.transforms.ToTensor(), normalization])

train_composed_transforms = torchvision.transforms.Compose([greyscale, invert, flip, rotate, resize, torchvision.transforms.ToTensor(), normalization])


########################
##### Loading Data #####
########################
print(os.path.exists("/opt/ChallengeDeep/test/"))
test_path = "/opt/ChallengeDeep/test/"
# Little sample to try
#train_path = "/usr/users/gpusdi1/gpusdi1_49/Bureau/sample_train"

valid_ratio = 0.2

# Load testing data
print('Data loading started')
#train_path = "/opt/ChallengeDeep/train/"
dataset = datasets.ImageFolder(test_path, test_composed_transforms)
print('Dataset Loaded')

##### Generating Loaders #####
num_workers = 4
batch_size = 256

# testing loader
test_loader = torch.utils.data.DataLoader(dataset=dataset,
                                           batch_size=batch_size,
                                           num_workers=num_workers,
                                           shuffle=False)

# Data Inspect
print("The test set contains {} images, in {} batches".format(
    len(test_loader.dataset), len(test_loader)))

####################################
##### Model construction items #####
####################################
# A convolutional base block
def conv_relu_maxpool(cin, cout, csize, cstride, cpad, msize, mstride, mpad):
    return [nn.Conv2d(cin, cout, csize, cstride, cpad),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(msize, mstride, mpad)]

# A linear base block


def linear_relu(dim_in, dim_out):
    return [nn.Linear(dim_in, dim_out),
            nn.ReLU(inplace=True)]

# Compute convolution output
def out_size(conv_model):
    dummy_input = torch.zeros(1, 1, 300, 300)
    dummy_output = conv_model(dummy_input)
    return np.prod(dummy_output.shape[1:])

# Loss function (Adapté de Git)


class F1_Loss(nn.Module):
    '''Calculate F1 score. Can work with gpu tensors

    The original implmentation is written by Michal Haltuf on Kaggle.

    Returns
    -------
    torch.Tensor
        `ndim` == 1. epsilon <= val <= 1
    '''

    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true,):
        assert y_pred.ndim == 2
        assert y_true.ndim == 1


        ##### Warning remplacer 2 par 86
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


# Init loss object
f_loss = F1_Loss()

############################
##### Building a Model #####
############################

class convClassifier(nn.Module):
    def __init__(self, num_classes):
        super(convClassifier, self).__init__()
        self.conv_model = nn.Sequential(*conv_relu_maxpool(cin=1, cout=4,
                                                           csize=3, cstride=1, cpad=1,
                                                           msize=2, mstride=2, mpad=0),
                                        *conv_relu_maxpool(cin=4, cout=8,
                                                           csize=3, cstride=1, cpad=1,
                                                           msize=2, mstride=2, mpad=0),
                                        *conv_relu_maxpool(cin=8, cout=16,
                                                           csize=3, cstride=1, cpad=1,
                                                           msize=2, mstride=2, mpad=0),
                                        *conv_relu_maxpool(cin=16, cout=16,
                                                           csize=3, cstride=1, cpad=1,
                                                           msize=2, mstride=2, mpad=0))

        print("initiated conv model")

        output_size = out_size(self.conv_model)
        print(output_size)

        self.fc_model = nn.Sequential(*linear_relu(output_size, 256),
                                      nn.Linear(256, 86))
        print("initiated linear model")

    def forward(self, x):
        x = x.view(x.size(dim=0), 1, 300, 300)
        x = self.conv_model(x).view(x.size(dim=0), -1)
        y = self.fc_model(x)
        return y


model = convClassifier(num_classes=len(dataset.classes))

use_gpu = torch.cuda.is_available()
if use_gpu:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print("saving model to device")
model.to(device)
print("model saved to device")

# Optimizer
optimizer = torch.optim.Adam(model.parameters())

################################
##### testing #####
################################

# Test
def test(model, loader, device):
    print('entered test')

    # We disable gradient computation
    with torch.no_grad():

        # We enter evaluation mode
        model.eval()

        # Empty sub 
        preds = []

        for i, inputs in enumerate(tqdm(loader)):

            inputs = inputs[0].to(device)

            # Compute the forward pass, i.e. the scores for each input image
            outputs = model(inputs).max(1).indices

            # Accumulate the predictions
            preds += outputs

        return pd.DataFrame(np.int_(preds), columns = ['label'])


###############################
##### Load the best model #####
###############################
runName = "Fast_3"

def best_run_model_logpath(logdir, raw_run_name, runName):
    run_name = raw_run_name + "_" + str(runName) 
    log_path = os.path.join(logdir, run_name)
    return log_path

# 1- create the logs directory if it does not exist
top_logdir = "./logs"
if not os.path.exists(top_logdir):
    os.mkdir(top_logdir)

logpath = best_run_model_logpath(top_logdir, "run", runName)
print("Loading from {}".format(logpath))

# Load trained model
model.load_state_dict(torch.load(os.path.join(logpath,"best_model.pt")))
print("loaded mode state dict")

###############################
##### Main test run   #####
###############################
if __name__ == '__main__':
    """
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--n_epochs',
        type=int,
        required=True
    )

    print(sys.argv)
    args = parser.parse_args()
    eval(f"{args.command}(args)")
    """
    ##### testing #####

    start_time = time.time()

    predictions = test(model, test_loader, device)
    print("Test time : {}".format(time.time() - start_time))

    # load img names and compose with predictions 
    names = pd.read_csv('./all_detritus_test.csv', usecols = ['imgname'])

    ##### Save predictions csv #####

    print("saving predictions")
    pd.concat([names,predictions], axis=1).to_csv(os.path.join(logpath,"sub.csv"), header = True, index = None)
    print("saved predictions")
