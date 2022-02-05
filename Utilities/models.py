import torch
import torchvision
import torch.nn as nn
import torch.nn.functional

import pandas as pd
import numpy as np
from tqdm import tqdm

import items

import os
import time

####################################
####### Model handler class ########
####################################

class ModelHandler():
    '''A wrapper around pytorch models and associated items for less bulky usage 
    Currently supported operations : 
        - Creating a model (from the models.py zoo)
        - Setting a loss and optimizer
        - Training a model, saving model summary, best model earlystopping
        - Making test predictions and saving to the Kaggle challenge format
    '''
    def __init__(self,model_name, f_loss, optimizer, run_name, batch_size):
        self.model= convClassifier(model_name = model_name)
        self.f_loss= items.get_loss(f_loss)
        self.optimizer= items.get_optimizer(optimizer, self.model)
        self.batch_size= batch_size
        # Set log directory 
        run_dir = "run_" + run_name
        top_logdir = "./logs"
        if not os.path.exists(top_logdir):
            os.mkdir(top_logdir)
        self.log_path = os.path.join(top_logdir, run_dir)
        # Set Model checkpoint 
        self.checkpoint = items.ModelCheckpoint(self.log_path + "/best_model.pt", self.model)
        # Storage items
        self.summary = None 
        self.predictions = None
    
    def toDevice(self):
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.model.to(self.device)
        print('loaded model on {}'.format(self.device))
    
    def save_summary(self, args, preprocessing, augmentation):
        summary_file = open(self.log_path + "/summary.txt", 'w')
        summary_text = """
        Executed command
        ================
        {}
        Preprocessing
        =======
        {}
        Augmentation
        =======
        {}
        Model summary
        =============
        {}
        {} trainable parameters
        Optimizer
        ========
        {}
        Loss
        ========
        {}
        """.format(args, preprocessing, augmentation, self.model, sum(p.numel() for p in self.model.parameters() if p.requires_grad), self.optimizer, self.f_loss)
        summary_file.write(summary_text)
        summary_file.close()
        print('Saved a summary of this run')
    
    def train(self, loader):
        print('entered train')
        self.model.train()

        N = 0
        tot_loss, correct = 0.0, 0.0
        class_targets = {}
        for i in range(86):
            class_targets[i] = [0, 0]

        for i, (inputs, targets) in enumerate(tqdm(loader)):

            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Forward pass
            outputs = self.model(inputs)
            loss = self.f_loss(outputs, targets)

            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Compute and accumulate metrics
            N += inputs.shape[0]
            tot_loss += inputs.shape[0] * self.f_loss(outputs, targets).item()
            predicted_targets = outputs.argmax(dim=1)
            correct += (predicted_targets == targets).sum().item()

            # Acc targets stuff
            if inputs.shape[0] == self.batch_size:
                for i in range(86):
                    number_of_target = (targets == i).sum().item()
                    number_of_good_pred = (
                        (predicted_targets == targets)*(targets == i)).sum().item()
                    class_targets[i][0] += number_of_good_pred
                    class_targets[i][1] += number_of_target
        acc_targets = {}
        for i in range(86):
            acc_targets[i] = str(class_targets[i][0]) + \
                '/' + str(class_targets[i][1])
        print("Acc_targets :{}".format(acc_targets))
        print(" Training : Loss : {:.4f}, Acc : {:.4f}".format(tot_loss/N, correct/N))

    def valid(self, loader):
        print('entered validation')
        with torch.no_grad():
            self.model.eval()

            N = 0
            tot_loss, correct = 0.0, 0.0
            class_targets = {}
            for i in range(86):
                class_targets[i] = [0, 0]

            for i, (inputs, targets) in enumerate(tqdm(loader)):

                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # Forward pass
                outputs = self.model(inputs)

                # Compute and accumulate
                N += inputs.shape[0]
                tot_loss += inputs.shape[0] * self.f_loss(outputs, targets).item()
                predicted_targets = outputs.argmax(dim=1)
                correct += (predicted_targets == targets).sum().item()

                # Acc target stuff
                if inputs.shape[0] == self.batch_size:
                    for i in range(86):
                        number_of_target = (targets == i).sum().item()
                        number_of_good_pred = (
                            (predicted_targets == targets)*(targets == i)).sum().item()
                        class_targets[i][0] += number_of_good_pred
                        class_targets[i][1] += number_of_target
            acc_targets = {}
            for i in range(86):
                acc_targets[i] = str(class_targets[i][0]) + \
                    '/' + str(class_targets[i][1])
            print("Acc_targets :{}".format(acc_targets))
            print(" Training : Loss : {:.4f}, Acc : {:.4f}".format(tot_loss/N, correct/N))

            self.model_checkpoint.update(tot_loss/N)
    
    def predict(self, loader):
        start_time = time.time()
        with torch.no_grad():
            self.model.eval()
            preds = []
            for i, inputs in enumerate(tqdm(loader)):
                inputs = inputs[0].to(self.device)
                outputs = self.model(inputs).max(1).indices
                preds += outputs
            print("Test time : {}".format(time.time() - start_time))
            return preds
    
    def label(self, loader): 
        '''Takes an unlabeled data loader and stores the image names and predicted labels'''
        # Get image names
        dataset = loader.dataset
        self.filenames = [dataset.samples[i][0].split('/')[-1] for i in range(len(dataset))]
        # Get predictions 
        self.predictios = self.predict(loader)
        print("labeled images")
    
    def save_predictions(self):
        '''Saves predicted labels on images in kaggle challenge format'''
        df = pd.DataFrame()
        df['imgname'] = self.filenames
        df['label'] = np.int_(self.predictions)
        df.to_csv(os.path.join(self.log_path,"sub.csv"), header = True, index = None)
        print("saved predictions")

#####################
##### Model zoo #####
#####################

class convClassifier(nn.Module):
    def __init__(self, model_name, num_classes=86):

        if model_name == 'custom1':
            super(convClassifier, self).__init__()
            self.conv_model = nn.Sequential(*items.conv_relu_maxpool(cin=1, cout=8,
                                                               csize=5, cstride=1, cpad=2,
                                                               msize=2, mstride=2, mpad=0),
                                            *items.conv_relu_maxpool(cin=8, cout=16,
                                                               csize=5, cstride=1, cpad=2,
                                                               msize=2, mstride=2, mpad=0),
                                            *items.conv_relu_maxpool(cin=16, cout=32,
                                                               csize=5, cstride=1, cpad=2,
                                                               msize=2, mstride=2, mpad=0),
                                            *items.conv_relu_maxpool(cin=32, cout=64,
                                                               csize=5, cstride=1, cpad=2,
                                                               msize=2, mstride=2, mpad=0))
            print("initiated conv model")
            dummy_input = torch.zeros(1, 1, 300, 300)
            output_size = items.out_size(self.conv_model, dummy_input)
            print(output_size)
            self.fc_model = nn.Sequential(*items.linear_relu(output_size, 128, 0.2),
                                          *items.linear_relu(128, 256, 0.2),
                                          *items.linear_softmax(256, num_classes))
            print("initiated linear model")

        elif model_name == 'custom2':
            super(convClassifier, self).__init__()
            self.conv_model = nn.Sequential(*items.conv_relu_maxpool(cin=1, cout=8,
                                                               csize=5, cstride=1, cpad=2,
                                                               msize=2, mstride=2, mpad=0),
                                            *items.conv_relu_maxpool(cin=8, cout=16,
                                                               csize=5, cstride=1, cpad=2,
                                                               msize=2, mstride=2, mpad=0),
                                            *items.conv_relu_maxpool(cin=16, cout=32,
                                                               csize=5, cstride=1, cpad=2,
                                                               msize=2, mstride=2, mpad=0),
                                            *items.conv_relu_maxpool(cin=32, cout=64,
                                                               csize=5, cstride=1, cpad=2,
                                                               msize=2, mstride=2, mpad=0))
            print("initiated conv model")
            dummy_input = torch.zeros(1, 1, 300, 300)
            output_size = items.out_size(self.conv_model, dummy_input)
            print(output_size)
            self.fc_model = nn.Sequential(*items.linear_relu(output_size, 128, 0.2),
                                          *items.linear_relu(128, 256, 0.2),
                                          *nn.Linear(256, num_classes))
            print("initiated linear model")
        
    def forward(self, x):
        x = self.conv_model(x).view(x.size(dim=0), -1)
        y = self.fc_model(x)
        return y
