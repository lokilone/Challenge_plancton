import torch
import argparse
import logging
import pathlib
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms

import models
import items
import scripts
import preprocessing

import pandas as pd

import os.path
import sys
import time

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'test']
        required=True
    )

    parser.add_argument(
        '--model',
        type=str,
        choices=['custom', 'resnet', 'vgg'],
        default='custom'
    )

    parser.add_argument(
        '--loss',
        type=str,
        choices=['f1', 'cross_entropy']
        default='f1'
    )

    parser.add_argument(
        '--n_epochs',
        type=int,
        default=10
    )

    parser.add_argument(
        '--valid',
        type=int,
        default=0.2
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=64
    )

    parser.add_argument(
        '--num_workers',
        type=int,
        default=4
    )

    parser.add_argument(
        '--sampler',
        type=bool,
        default=False,
        required=True
    )

    parser.add_argument(
        '--run_name',
        type=str,
        required=True
    )

    print(sys.argv)
    args = parser.parse_args()
    eval(f"{args.command}(args)")


##### Locating data and log directories #####
run_dir = "run_" + args.run_name
top_logdir = "./logs"
if not os.path.exists(top_logdir):
    os.mkdir(top_logdir)
log_path = os.path.join(top_logdir, run_dir)
if os.path.exists(log_path):
    print("Logging to {}".format(log_path))

data_path = os.path.join("/opt/ChallengeDeep/", args.mode)
if os.path.exists(data_path):
    print("Found data path")


##### Load and preprocess data #####
print('Data loading started')
if args.mode == 'train':
    dataset = datasets.ImageFolder(
        data_path, preprocessing.transforms("train"))
    # Train test split
    train_dataset, valid_dataset = torch.utils.data.dataset.random_split(dataset,
                                                                         [int((1.0 - args.valid) * len(dataset)),  # nb_train
                                                                          len(dataset)-int((1.0 - args.valid) * len(dataset))])  # nb_valid = data-nb_train
elif args.mode == 'test':
    test_dataset = datasets.ImageFolder(
        data_path, preprocessing.transforms("test"))
print('Dataset Loaded')


##### Generating Loaders #####
if args.mode == "train":
    # train loader
    if args.sampler:
        # Random sampler
        print("creating sampler")
        sampler = preprocessing.sampler(train_dataset)
        print("sampler created")
        # training loader
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=args.batch_size,
                                                   num_workers=args.num_workers,
                                                   shuffle=False, sampler=sampler)
    else:
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=args.batch_size,
                                                   num_workers=args.num_workers,
                                                   shuffle=True)
    # validation loader
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.num_workers,
                                               shuffle=False)
    # Data Inspect
    print("The train set contains {} images, in {} batches".format(
        len(train_loader.dataset), len(train_loader)))
    print("The validation set contains {} images, in {} batches".format(
        len(valid_loader.dataset), len(valid_loader)))

elif args.mode == "test":
    # test loader
    test_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=args.batch_size,
                                              num_workers=args.num_workers,
                                              shuffle=False)
    # Data Inspect
    print("The test set contains {} images, in {} batches".format(
        len(test_loader.dataset), len(test_loader)))


##### Getting model #####
model = models.convClassifier(args.model, num_classes=86)

use_gpu = torch.cuda.is_available()
if use_gpu:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model.to(device)
print("generated model")

if args.mode == "test":
    model.load_state_dict(torch.load(os.path.join(log_path, "best_model.pt")))
    print("loaded trained parameters")


##### Main script #####
if args.mode == "train":
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters())
    # Loss
    f_loss = items.loss(args.f_loss)
    # Callback object
    model_checkpoint = items.ModelCheckpoint(
        log_path + "/best_model.pt", model)
    # Learning loop
    for t in range(args.n_epochs):
        start_time = time.time()
        # train step
        train_loss, train_acc = scripts.train(
            model, train_loader, f_loss, optimizer, device, args.batch_size)
        print(" Training : Loss : {:.4f}, Acc : {:.4f}".format(
            train_loss, train_acc))
        # validation step
        val_loss, val_acc = scripts.test(
            model, valid_loader, f_loss, device, args.batch_size)
        print(" Validation : Loss : {:.4f}, Acc : {:.4f}".format(
            val_loss, val_acc))
        # update best model
        model_checkpoint.update(val_loss)

        print("Epoch {} time : {}".format(t, time.time() - start_time))

    print('Finished training on {} epochs'.format(args.n_epochs))
    # Save a summary of the run
    summary_file = open(log_path + "/summary.txt", 'w')
    summary_text = """
    Executed command
    ================
    {}
    Preprocessing
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
    """.format(" ".join(sys.argv), preprocessing.transforms("train"), model, sum(p.numel() for p in model.parameters() if p.requires_grad), optimizer, f_loss)
    summary_file.write(summary_text)
    summary_file.close()

elif args.mode == "test":
    subs = pd.Dataframe()
    # Get image names :
    filenames = []
    subs['imgname'] = filenames
    # Make predictions :
    start_time = time.time()
    predictions = scripts.test(model, test_loader, device)
    subs['label'] = predictions
    print("Test time : {}".format(time.time() - start_time))
    # Save submission as CSV
    print("saving predictions")
    subs.to_csv(os.path.join(log_path, "sub.csv"), header=True, index=None)
    print("saved predictions")
