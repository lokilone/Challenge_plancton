import torch

from tqdm import tqdm

##################### One train step ###########################


def train(model, loader, f_loss, optimizer, device, batch_size):
    print('entered train')
    # enter train mode
    model.train()

    N = 0
    tot_loss, correct = 0.0, 0.0
    class_targets = {}
    for i in range(86):
        class_targets[i] = [0, 0]

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

        if inputs.shape[0] == batch_size:
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

    return tot_loss/N, correct/N

##################### test on given data ###########################


def test(model, loader, f_loss, device, batch_size):
    print('entered test')

    # We disable gradient computation
    with torch.no_grad():

        # We enter evaluation mode
        model.eval()

        N = 0
        tot_loss, correct = 0.0, 0.0

        class_targets = {}
        for i in range(86):
            class_targets[i] = [0, 0]

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

            if inputs.shape[0] == batch_size:
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

        return tot_loss/N, correct/N
