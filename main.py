import argparse
import logging

import models
import Preprocessing

import sys
import time

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'mode',
        type=str,
        choices=['train', 'test']
    )

    parser.add_argument(
        'path',
        type=str
    )

    parser.add_argument(
        '--debug',
        type=bool,
        choices=[True, False],
        required=False,
        default = False
    )

    parser.add_argument(
        '--model',
        type=str,
        choices=['custom1', 'custom2', 'minimal', 'minimal_softmax', 'minimal_dropout', 'resnet', 'resnet152', 'vgg', 'vgg19'],
        default='minimal'
    )

    parser.add_argument(
        '--loss',
        type=str,
        choices=['f1', 'cross_entropy'],
        default='cross_entropy'
    )

    parser.add_argument(
        '--optimizer',
        type=str,
        choices=['adam'],
        default='adam'
    )

    parser.add_argument(
        '--n_epochs',
        type=int,
        default=20
    )

    parser.add_argument(
        '--valid',
        type=int,
        default=0.2
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=256
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
        default='final'
    )

    parser.add_argument(
        '--preprocessing',
        nargs='*',
        choices=['centercrop', 'resize', 'totensor', 'invert', 'normalization', 'greyscale', 'greyscale3'],
        default=['greyscale', 'invert', 'centercrop', 'totensor', 'normalization']
    )

    parser.add_argument(
        '--augmentation',
        nargs='*',
        choices=['flip', 'rotate', 'blur'],
        default = ['flip', 'rotate']
    )

    print(sys.argv)
    args = parser.parse_args()

# Load and preprocess data 
transforms = Preprocessing.ComposedTransforms()
train_transforms = transforms.train_transforms(preprocessing_seq=args.preprocessing, augmentation_seq=args.augmentation)
test_transforms = transforms.test_transforms(preprocessing_seq=args.preprocessing)

if args.debug:
    data_path = "/usr/users/gpusdi1/gpusdi1_49/Bureau/sample_train" # A small dataset used for debugging 
else:
    data_path = args.path

if args.mode == 'train':
    dataloader = Preprocessing.DataLoader(train_valid_path = data_path, valid_ratio=args.valid, num_workers=args.num_workers, batch_size=args.batch_size)
    dataloader.Load_Train_Valid(train_composed_transforms=train_transforms, valid_composed_transforms=test_transforms)
    train_loader = dataloader.train_loader
    valid_loader = dataloader.valid_loader
elif args.mode == 'test':
    dataloader = Preprocessing.DataLoader(test_path = data_path, num_workers=args.num_workers, batch_size=args.batch_size)
    dataloader.Load_Test(test_composed_transforms=test_transforms)

# Getting model
handler = models.ModelHandler(model_name = args.model, f_loss=args.loss, optimizer=args.optimizer, run_name=args.run_name, batch_size=args.batch_size)
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
    handler.load_best()
    handler.label(dataloader)
    handler.save_predictions()

