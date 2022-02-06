import argparse
import logging

import sys
import os

sys.path.append(os.path.abspath('../'))
import models
import Preprocessing

if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'run',
        type=str
    )

    parser.add_argument(
        '--debug',
        type=bool,
        choices=[True, False],
        default = False
    )

    parser.add_argument(
        '--model',
        type=str,
        choices=['custom1', 'custom2', 'minimal', 'minimal_softmax', 'minimal_dropout', 'resnet', 'resnet152', 'vgg', 'vgg19'],
        required = True
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
        '--preprocessing',
        nargs='*',
        choices=['centercrop', 'resize', 'totensor', 'invert', 'normalization', 'greyscale', 'greyscale3'],
        required = True
    )

    print(sys.argv)
    args = parser.parse_args()

# Load and preprocess data 
transforms = Preprocessing.ComposedTransforms()
test_transforms = transforms.test_transforms(preprocessing_seq=args.preprocessing)

if args.debug:
    dataloader = Preprocessing.DataLoader(test_path = "/usr/users/gpusdi1/gpusdi1_49/Bureau/sample_train", num_workers=args.num_workers, batch_size=args.batch_size)
else:
    dataloader = Preprocessing.DataLoader(num_workers=args.num_workers, batch_size=args.batch_size)

dataloader.Load_Test(test_composed_transforms=test_transforms)

# Getting model
handler = models.ModelHandler(model_name = args.model, run_name= args.run, batch_size=args.batch_size)
handler.toDevice()

# Main script 
handler.load_best(args.run)
handler.score(dataloader.test_loader)
