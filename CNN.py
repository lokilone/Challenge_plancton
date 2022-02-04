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
# Normalization with computed values
normalization = torchvision.transforms.Normalize(mean=[0.0988],std=[0.1444])


### Data Augmentation
rotate = torchvision.transforms.RandomRotation((0, 360))
flip = torchvision.transforms.RandomHorizontalFlip(p=0.5)
flou = torchvision.transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 1))


# Compose transforms
test_composed_transforms = torchvision.transforms.Compose([greyscale, invert, resize, torchvision.transforms.ToTensor(), normalization])

train_composed_transforms = torchvision.transforms.Compose([greyscale, invert, flip, rotate, resize, torchvision.transforms.ToTensor(), normalization])


# Create transformer class (currently unused, we transform on data loading)


class DatasetTransformer(torch.utils.data.Dataset):

    def __init__(self, base_dataset, transforms):
        self.base_dataset = base_dataset
        self.transform = transforms

    def __getitem__(self, index):
        img, target = self.base_dataset[index]
        return self.transform(img), target

    def __len__(self):
        return len(self.base_dataset)

########################
##### Loading Data #####
########################
print(os.path.exists("/opt/ChallengeDeep/train/"))
train_path = "/opt/ChallengeDeep/train/"
# Little sample to try
#train_path = "/usr/users/gpusdi1/gpusdi1_49/Bureau/sample_train"

valid_ratio = 0.2

# Load learning data
print('Data loading started')
#train_path = "/opt/ChallengeDeep/train/"
dataset = datasets.ImageFolder(train_path, train_composed_transforms)
print('Dataset Loaded')

# Train test split
nb_train = int((1.0 - valid_ratio) * len(dataset))
nb_valid = len(dataset)-nb_train
train_dataset, valid_dataset = torch.utils.data.dataset.random_split(dataset, [
                                                                     nb_train, nb_valid])


# Random sampler

def sampler_(dataset,train_counts):
    start_time = time.time()
    num_samples = len(dataset)
    labels = [dataset[item][1] for item in tqdm(range(len(dataset)))]
    label_end_time = time.time()
    print('got labels in {} s'.format(label_end_time-start_time))

    class_weights = torch.from_numpy(1./ np.array(train_counts))
    classw_end_time = time.time()
    print('got class weights in {} s'.format(classw_end_time-label_end_time))
    weights = [class_weights[labels[i]] for i in tqdm(range(num_samples))]
    weight_end_time = time.time()
    print('got final weights in {} s'.format(weight_end_time-classw_end_time))
    sampler = torch.utils.data.sampler.WeightedRandomSampler(torch.DoubleTensor(weights), num_samples)
    return sampler

##### Generating Loaders #####
num_workers = 4
batch_size = 256

# Random sampler
train_set_counts = [1983, 243570, 214, 638, 1326, 1328, 23797, 5781, 289, 18457, 536, 185, 1045, 1210, 
686, 5570, 8402, 3060, 168, 953, 4764, 2825, 3242, 78, 37, 3916, 98, 8200, 576, 19225, 686, 4213, 336, 188, 
1459, 1869, 180000, 3538, 1091, 6056, 142, 33147, 2085, 170, 308, 14799, 4609, 156, 3900, 3983, 3111, 1988, 
5079, 244, 6368, 757, 1289, 12636, 42096, 10008, 3465, 269, 457, 10038, 8213, 372, 2314, 234, 590, 15431, 12954, 
4391, 1285, 5604, 6996, 53387, 235, 632, 11490, 88, 2589, 2517, 388, 2086, 172, 727]

print("creating sampler")
sampler = sampler_(train_dataset,train_set_counts)
print("sampler created")

# training loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           num_workers=num_workers,
                                           shuffle=False, sampler=sampler)

# validation loader
valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                           batch_size=batch_size,
                                           num_workers=num_workers,
                                           shuffle=True)

# Data Inspect
print("The train set contains {} images, in {} batches".format(
    len(train_loader.dataset), len(train_loader)))
print("The validation set contains {} images, in {} batches".format(
    len(valid_loader.dataset), len(valid_loader)))

####################################
##### Model construction items #####
####################################
# A convolutional base block
def conv_relu_maxpool(cin, cout, csize, cstride, cpad, msize, mstride, mpad):
    return [nn.Conv2d(cin, cout, csize, cstride, cpad),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(msize, mstride, mpad)]

# A linear base blocks

def linear_relu(dim_in, dim_out, p_dropout):
    return [nn.Dropout(p_dropout),
            nn.Linear(dim_in, dim_out),
            nn.ReLU(inplace=True)]

def linear_softmax(dim_in, dim_out):
    return [nn.Linear(dim_in, dim_out),
            nn.Softmax()]

# Dropout layer


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

        output_size = out_size(self.conv_model)
        print(output_size)

        self.fc_model = nn.Sequential(*linear_relu(output_size, 128, 0.2),
                                      *linear_relu(128, 256, 0.2),
                                      *linear_softmax(256, num_classes))
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

model.to(device)

# Optimizer
optimizer = torch.optim.Adam(model.parameters())


################################
##### training and testing #####
################################

# One train step
def train(model, loader, f_loss, optimizer, device):
    print('entered train')

    # enter train mode
    model.train()

    N = 0
    tot_loss, correct = 0.0, 0.0

    
    class_targets = {}
    for i in range(86):
        class_targets[i]=[0,0]

    for i, (inputs, targets) in enumerate(tqdm(loader)):
        
        inputs, targets = inputs.to(device), targets.to(device)

        # Compute the forward pass through the network up to the loss
        outputs = model(inputs)
        loss = f_loss(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate the exact number of processed samples
        N += inputs.shape[0]

        # Accumulate the loss considering
        tot_loss += inputs.shape[0] * f_loss(outputs, targets).item()

        # For the accuracy, we compute the labels for each input image
        predicted_targets = outputs.argmax(dim=1)
        correct += (predicted_targets == targets).sum().item()


        if inputs.shape[0]==batch_size :
            for i in range(86):
                number_of_target = (targets == i).sum().item()
                number_of_good_pred = ((predicted_targets == targets)*(targets == i)).sum().item()
                class_targets[i][0] += number_of_good_pred
                class_targets[i][1] += number_of_target

    acc_targets = {}
    for i in range(86):
        acc_targets[i] = str(class_targets[i][0]) + '/' + str(class_targets[i][1])

    print(" Training : Loss : {:.4f}, Acc : {:.4f}, Acc_targets :{}".format(
    tot_loss/N, correct/N, acc_targets))
        
    return tot_loss/N, correct/N

# Test


def test(model, loader, f_loss, device):
    print('entered test')

    # We disable gradient computation
    with torch.no_grad():

        # We enter evaluation mode
        model.eval()

        N = 0
        tot_loss, correct = 0.0, 0.0

        class_targets = {}
        for i in range(86):
            class_targets[i]=[0,0]

        for i, (inputs, targets) in enumerate(tqdm(loader)):
            
            inputs, targets = inputs.to(device), targets.to(device)

            # Compute the forward pass, i.e. the scores for each input image
            outputs = model(inputs)

            # Accumulate the exact number of processed samples
            N += inputs.shape[0]

            # Accumulate the loss considering
            tot_loss += inputs.shape[0] * f_loss(outputs, targets).item()

            # For the accuracy, we compute the labels for each input image
            predicted_targets = outputs.argmax(dim=1)
            correct += (predicted_targets == targets).sum().item()

            if inputs.shape[0]==batch_size :
                for i in range(86):
                    
                    number_of_target = (targets == i).sum().item()
                    number_of_good_pred = ((predicted_targets == targets)*(targets == i)).sum().item()

                    class_targets[i][0] += number_of_good_pred
                    class_targets[i][1] += number_of_target

        acc_targets = {}
        for i in range(86):
            acc_targets[i] = str(class_targets[i][0]) + '/' + str(class_targets[i][1])
        
        print(acc_targets)

        return tot_loss/N, correct/N, acc_targets


###############################
##### Save the best model #####
###############################

def generate_unique_logpath(logdir, raw_run_name):
    i = "Night_CNN_2"
    while(True):
        run_name = raw_run_name + "_" + str(i)
        log_path = os.path.join(logdir, run_name)
        if os.path.isdir(log_path):
            return log_path


# 1- create the logs directory if it does not exist
top_logdir = "./logs"
if not os.path.exists(top_logdir):
    os.mkdir(top_logdir)

logdir = generate_unique_logpath(top_logdir, "run")
print("Logging to {}".format(logdir))


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


# Define the callback object
model_checkpoint = ModelCheckpoint(logdir + "/best_model.pt", model)

# Monitoring obejct
tensorboard_writer = SummaryWriter(log_dir=logdir)


###############################
##### Main learning run   #####
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
    ##### learning loop #####
    epochs = 20

    for t in range(epochs):
        start_time = time.time()

        train_loss, train_acc = train(
            model, train_loader, f_loss, optimizer, device)

        val_loss, val_acc, acc_targets = test(model, valid_loader, f_loss, device)
        print(" Validation : Loss : {:.4f}, Acc : {:.4f}".format(
            val_loss, val_acc))
        model_checkpoint.update(val_loss)

        # Monitoring
        tensorboard_writer.add_scalar('metrics/train_loss', train_loss, t)
        tensorboard_writer.add_scalar('metrics/train_acc',  train_acc, t)
        tensorboard_writer.add_scalar('metrics/val_loss', val_loss, t)
        tensorboard_writer.add_scalar('metrics/val_acc',  val_acc, t)

        print("Epoch {} time : {}".format(t, time.time() - start_time))

    print('learned')
    #tensorboard_writer.flush()

    ##### Save a summary of the run #####

    summary_file = open(logdir + "/summary.txt", 'w')
    summary_text = """

    Executed command
    ================
    {}

    Dataset
    =======
    FashionMNIST

    Model summary
    =============
    {}

    {} trainable parameters

    Optimizer
    ========
    {}

    """.format(" ".join(sys.argv), model, sum(p.numel() for p in model.parameters() if p.requires_grad), optimizer)
    summary_file.write(summary_text)
    summary_file.close()
