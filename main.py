import torch
import argparse
import logging
import pathlib
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms

from Utilities import models
from Utilities import Preprocessing

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
        choices=['train', 'test'],
        required=True
    )

    parser.add_argument(
        '--model',
        type=str,
        choices=['custom1', 'custom2', 'resnet', 'vgg'],
        default='custom'
    )

    parser.add_argument(
        '--loss',
        type=str,
        choices=['f1', 'cross_entropy'],
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
        default=False
    )

    parser.add_argument(
        '--run_name',
        type=str,
        required=True
    )

    parser.add_argument(
        '--preprocessing',
        nargs='*',
        choices=['centercrop', 'resize', 'totensor', 'invert', 'normalization', 'greyscale', 'greyscale3'],
        default=['greyscale', 'invert', 'centercrop', 'totensor']
    )

    parser.add_argument(
        '--preprocessing',
        nargs='*',
        choices=['flip', 'rotate', 'blur'],
        default = []
    )

    print(sys.argv)
    args = parser.parse_args()
    eval(f"{args.command}(args)")

# Load and preprocess data 
transforms = Preprocessing.ComposedTransforms()
dataloader = Preprocessing.DataLoader(valid_ratio=args.valid, num_workers=args.num_workers, batch_size=args.batch_size)
train_transforms = dataloader.train_transforms(preprocessing_seq=args.preprocessing, augmentation_seq=args.augmentation)
test_transforms = dataloader.train_transforms(preprocessing_seq=args.preprocessing)

if args.mode == 'train':
    dataloader.Load_Train_Valid(train_transforms=train_transforms, valid_transforms=test_transforms)
    train_loader = dataloader.train_loader
    valid_loader = dataloader.valid_loader

elif args.mode == 'test':
    dataloader.Load_Test(test_composed_transforms=test_transforms)
    test_loader = dataloader.test_loader

# Getting model
handler = models.ModelHandler(model_name = args.model, f_loss=args.f_loss, optimizer=args.optimizer, run_name=args.run_name, batch_size=args.batch_size)
handler.toDevice()

# Main script 
if args.mode == "train":
    handler.save_summary(" ".join(sys.argv), args.preprocessing, args.augmentation)
    # Learning loop
    for t in range(args.n_epochs):
        start_time = time.time()
        handler.train(train_loader)
        handler.valid(valid_loader)
        print("Epoch {} time : {}".format(t, time.time() - start_time))

    print('Finished training on {} epochs'.format(args.n_epochs))

elif args.mode == "test":
    handler.label(test_loader)
    handler.save_predictions()
