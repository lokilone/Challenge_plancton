import numpy as np
import torch
import csv
from torch.nn.modules.module import _addindent
from sklearn import metrics

def train(model, loader, f_loss, optimizer, device, type):
    """
        Train a model for one epoch, iterating over the loader
        using the f_loss to compute the loss and the optimizer
        to update the parameters of the model.
        Arguments :
        model     -- A torch.nn.Module object
        loader    -- A torch.utils.data.DataLoader
        f_loss -- The loss function, i.e. a loss Module
        optimizer -- A torch.optim.Optimzer object
        use_gpu  -- Boolean, whether to use GPU
        Returns :
    """

    # We enter train mode. This is useless for the linear model
    # but is important for layers such as dropout, batchnorm, ...
    model.train()
    N = 0
    tot_loss, correct = 0.0, 0
    for i, (inputs, targets) in enumerate(loader):

        inputs, targets = inputs.to(device), targets.to(device)

        # Compute the forward propagation
        if type == "deit_tiny" or type == "deit_base" :
            outputs,x = model(inputs)
        else :
            outputs = model(inputs)

        loss = f_loss(outputs, targets)

        # Accumulate the number of processed samples
        N += inputs.shape[0]

        # For the total loss
        tot_loss += inputs.shape[0] * loss.item()

        # For the total accuracy
        predicted_targets = outputs.argmax(dim=1)
        correct += (predicted_targets == targets).sum().item()

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Display status
        #progress_bar(i, len(loader), msg = "Loss : {:.4f}, Acc : {:.4f}".format(tot_loss/N, correct/N))
    return tot_loss/N, correct/N

def validation(model, loader, f_loss, device):
    """
    Test a model by iterating over the loader

    Arguments :

        model     -- A torch.nn.Module object
        loader    -- A torch.utils.data.DataLoader
        f_loss    -- The loss function, i.e. a loss Module
        device    -- The device to use for computation 

    Returns :

        A tuple with the mean loss and mean accuracy

    """
    # We disable gradient computation which speeds up the computation
    # and reduces the memory usage
    with torch.no_grad():
        # We enter evaluation mode. This is useless for the linear model
        # but is important with layers such as dropout, batchnorm, ..
        model.eval()
        N = 0
        tot_loss, correct = 0.0, 0.0
        for i, (inputs, targets) in enumerate(loader):

            # We got a minibatch from the loader within inputs and targets
            # With a mini batch size of 128, we have the following shapes
            #    inputs is of shape (128, 1, 28, 28)
            #    targets is of shape (128)

            # We need to copy the data on the GPU if we use one
            inputs, targets = inputs.to(device), targets.to(device)

            # Compute the forward pass, i.e. the scores for each input image
            outputs= model(inputs)

            # We accumulate the exact number of processed samples
            N += inputs.shape[0]

            # We accumulate the loss considering
            # The multipliation by inputs.shape[0] is due to the fact
            # that our loss criterion is averaging over its samples
            tot_loss += inputs.shape[0] * f_loss(outputs, targets).item()

            # For the accuracy, we compute the labels for each input image
            # Be carefull, the model is outputing scores and not the probabilities
            # But given the softmax is not altering the rank of its input scores
            # we can compute the label by argmaxing directly the scores
            predicted_targets = outputs.argmax(dim=1)
            correct += (predicted_targets == targets).sum().item()
        return tot_loss/N, correct/N

def test(model, loader, device, num_classes):
    """
    Test a model by iterating over the loader
    Arguments :
        model     -- A torch.nn.Module object
        loader    -- A torch.utils.data.DataLoader
        f_loss    -- The loss function, i.e. a loss Module
        device    -- a torch.device object
    Returns :
        A tuple with the mean loss and mean accuracy
    """
    # Generate a list of integers representing the 86 different classes.
    classes = []
    for k in range(num_classes):
        classes.append(k)
    # We disable gradient computation which speeds up the computation
    # and reduces the memory usage
    with torch.no_grad():
        # We enter evaluation mode. This is useless for the linear model
        # but is important with layers such as dropout, batchnorm, ..
        model.eval()
        csv_file = open("submission.csv", 'w')
        csv_writer = csv.writer(csv_file, delimiter=",")
        csv_writer.writerow(["imgname", "label"])
        for i, (inputs, targets) in enumerate(loader):

            # We got a minibatch from the loader within inputs and targets
            # With a mini batch size of 128, we have the following shapes
            #    inputs is of shape (128, 1, 28, 28)
            #    targets is of shape (128)
            # We need to copy the data on the GPU if we use one
            inputs, targets = inputs.to(device), targets.to(device)
            # Compute the forward pass, i.e. the scores for each input image
            outputs = model(inputs)
            sample_fname, _ = loader.dataset.samples[i]
            name = sample_fname.split("/")[-1]
            print(name)
            # For the accuracy, we compute the labels for each input image
            # Be carefull, the model is outputing scores and not the probabilities
            # But given the softmax is not altering the rank of its input scores
            # we can compute the label by argmaxing directly the scores
            predicted_targets = outputs.argmax(dim=1)
            #correct += (predicted_targets == targets).sum().item()
            csv_writer.writerow([name, classes[predicted_targets]])
