import numpy as np
import torch
import torchvision.transforms.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import WeightedRandomSampler

import os
import sys
import argparse

import models
import utils


img_size = 300
num_classes = 86
epochs= 50
valid_ratio=0.2

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--action',
        type=str,
        help='Choose whether to train or test a model',
        choices=['train', 'test'],
        required=True
    )

    parser.add_argument(
        '--dataset_dir',
        type=str,
        help='Where to load the dataset',
        default=None
    )

    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='The number of CPU threads used'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='The batch size used to train the model'
    )

    parser.add_argument(
        '--optimizer',
        type=str,
        choices=['adam','sgd'],
        default="adam",
        help='The optimizer used for the training.'
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Choose the learning rate for the tranning process.'
    )

    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0,
        help='Weight decay'
    )

    parser.add_argument(
        '--weighted_sampler',
        type=bool,
        default=False,
        choices=[True,False],
        help='Whether or not use a weighted random sampler for the training dataset.'
    )

    parser.add_argument(
        '--logdir',
        type=str,
        default="./logs",
        help='The directory in which to store the logs'
    )

    parser.add_argument(
        '--model',
        choices=['linear', 'coat_mini', 'coat_small','cait', 'deit_tiny','deit_base', 'efficientnetv2','efficientnetb8','fnet4','fnet6', 'resnet50', 'swin','swin_in22k', 'tnt','vit', 'custom'],
        action='store',
        help='Choose one of the proposed model',
        required=True
    )

    parser.add_argument(
    '--load_model',
    action='store',
    default=None,
    help='The directory to load a trained model'
    )


    args = parser.parse_args()

print("Start")
# Class to add blank pixels around image for a resizing without changing the image shape
class SquarePad:
    def __call__(self, image):
        max_wh = max(image.size)
        p_left, p_top = [(max_wh - s) // 2 for s in image.size]
        p_right, p_bottom = [max_wh - (s+pad) for s, pad in zip(image.size, [p_left, p_top])]
        padding = (p_left, p_top, p_right, p_bottom)
        return F.pad(image, padding, 255, 'constant')

class Preprocessing:
    def __call__(self, image):
        return None
        
#Possible d'ajouter des transformation sous forme de classe comme SquarePad 
grayscale = transforms.Grayscale(num_output_channels=1)
invert = F.invert
resize = transforms.Resize((img_size, img_size),  transforms.InterpolationMode.BICUBIC)

rotate = transforms.RandomRotation((0, 360))
flip = transforms.RandomHorizontalFlip(p=0.5)
flou = transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 1))

#rotation
# 
# Data transformation and augmentation
transform = {
        'train': transforms.Compose([grayscale,
                invert,
                resize,
                flip,
                rotate,
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.0988],
                                  std=[0.1444])]),

        'test': transforms.Compose([grayscale,
                invert,
                resize,
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.0988],
                                  std=[0.1444])])
        }

# Function to apply the required data transformation and to create the datasets
def load_data(data_folder, batch_size, mode, valid_ratio=valid_ratio, num_workers=args.num_workers, **kwargs):
    if mode == 'test':
        print("Ligne 141")
        test_dataset = datasets.ImageFolder(root = data_folder,
                                    transform=transform['test'])
        test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

        return test_data_loader
    else:
        print("Ligne 147")
        train_valid_dataset = datasets.ImageFolder(root = data_folder, transform=transform['train'])

        nb_train = int((1.0 - valid_ratio) * len(train_valid_dataset)) + 1
        nb_valid =  len(train_valid_dataset) - nb_train
        print("Nb Train : ", nb_train)
        print("Nb Valid : ", nb_valid)
        print("Len input : ", len(train_valid_dataset))
        train_dataset, valid_dataset = torch.utils.data.dataset.random_split(train_valid_dataset, [nb_train, nb_valid])
        if args.weighted_sampler :
            targets = [train_dataset[i][1] for i in range(len(train_dataset))]
            class_sample_count = np.unique(targets, return_counts=True)[1]
            weight = 1. / class_sample_count
            samples_weight = weight[targets]
            samples_weight = torch.from_numpy(samples_weight)

            sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
            train_data_loader = torch.utils.data.DataLoader(train_dataset, 
                                                  batch_size=batch_size, 
                                                  sampler=sampler,
                                                  num_workers=num_workers,
                                                  **kwargs, drop_last = False)
            train_count = [0] * 86
            classes = []
            for k in range(86):
                classes.append(k)

            for _, targets in train_data_loader:
                for k in targets :
                    train_count[k] +=1
            print("train :",train_count)
        else :
            train_data_loader = torch.utils.data.DataLoader(train_dataset, 
                                                  batch_size=batch_size, 
                                                  shuffle=True,
                                                  num_workers=num_workers,
                                                  **kwargs, drop_last = False)
                                                  
        val_data_loader = torch.utils.data.DataLoader(valid_dataset, 
                                                  batch_size=batch_size, 
                                                  shuffle=True,
                                                  num_workers=num_workers,
                                                  **kwargs, drop_last = False)
        return train_data_loader, val_data_loader

# Load the datasets for the training or test phase. 
if args.action == "train" :
    print("Ligne 191")
    train_loader, valid_loader = load_data(data_folder=args.dataset_dir, batch_size=args.batch_size, mode=args.action)
    print("The train set contains {} images, in {} batches, model = {}".format(len(train_loader.dataset), len(train_loader), args.model))
    print("The validation set contains {} images, in {} batches".format(len(valid_loader.dataset), len(valid_loader)))
elif args.action == "test" :
    print("Ligne 199")
    test_loader = load_data(data_folder=args.dataset_dir, batch_size=1, mode=args.action)
    print("The test set contains {} images, in {} batches, model = {}".format(len(test_loader.dataset), len(test_loader), args.model))

# Check if GPU is available
use_gpu = torch.cuda.is_available()
print("Gpu : ", use_gpu)
if use_gpu:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Init model, loss, optimizer
# Load existing model if argument exists

model = models.build_model(args.model, img_size, num_classes)
if args.load_model != None :
    model.load_state_dict(torch.load(args.load_model))
model = model.to(device)
type = args.model

#### Weights and loss
def class_weights(path):
    train_path = path

    categories = np.sort(os.listdir(train_path))

    category_paths = []

    for category in categories :
        category_paths.append(os.path.join(train_path, category))

    number_of_images = []
    image_paths = {}
    image_names = {}

    for category_path, category in zip(category_paths, categories):
        number_of_images.append(len(os.listdir(category_path)))
        image_names[category] = np.sort(os.listdir(category_path))
        image_paths[category] = []
        for image in image_names[category]:
            image_paths[category].append(os.path.join(category_path, image))

    return number_of_images


weights = class_weights("/opt/ChallengeDeep/train/")
M = max(weights)
weights = torch.cuda.FloatTensor([ M/weight for weight in weights])

print("Len weights : ", len(weights))
print(weights)
f_loss = torch.nn.CrossEntropyLoss()

if args.optimizer == "adam" :    
    optimizer = torch.optim.Adam(params = model.parameters(), lr = args.lr, weight_decay=args.weight_decay)
if args.optimizer == "sgd" :
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True)

if args.action == "train":
    # Where to store the logs
    logdir = utils.generate_unique_logpath(args.logdir, args.model)
    print("Logging to {}".format(logdir))
    if not os.path.exists(args.logdir):
        os.mkdir(args.logdir)
    if not os.path.exists(logdir):
        os.mkdir(logdir)

    ###################################################
    # Where to save the logs of the metrics
    history_file = open(logdir + '/history', 'w', 1)
    history_file.write("Epoch\tTrain loss\tTrain acc\tVal loss\tVal acc\n")

    # Generate and dump the summary of the model
    #model_summary = utils.torch_summarize(model)
    #print("Summary:\n {}".format(model_summary))

    summary_file = open(logdir + "/summary.txt", 'w')
    summary_text = """
    Executed command
    ===============
    {}
    Dataset
    =======
    Train transform : {}
    Model summary
    =============
    {}
    {} trainable parameters
    Loss function
    ========
    {}
    Optimizer
    ========
    {}
        """.format(" ".join(sys.argv),
                   transform['train'],
                   str(model).replace('\n','\n\t'),
                   sum(p.numel() for p in model.parameters() if p.requires_grad),
                   str(f_loss).replace('\n', '\n\t'),
                   str(optimizer).replace('\n', '\n\t'))
    summary_file.write(summary_text)
    summary_file.close()

    tensorboard_writer   = SummaryWriter(log_dir = logdir)
    tensorboard_writer.add_text("Experiment summary", summary_text)
    model_checkpoint = utils.ModelCheckpoint(logdir + "/best_model.pt", model)

    # Add the graph of the model to the tensorboard
    inputs, _ = next(iter(train_loader))
    inputs = inputs.to(device)
    tensorboard_writer.add_graph(model, inputs)

    ###################################################################################### Main Loop
    print('Training is starting')
    for t in range(epochs):
        print("Epoch {}".format(t))
        train_loss, train_acc = utils.train(model, train_loader, f_loss, optimizer, device, type)
        val_loss, val_acc = utils.validation(model, valid_loader, f_loss, device)
        print(" Train : Loss : {:.4f}, Acc : {:.4f}, Validation : Loss : {:.4f}, Acc : {:.4f}".format(train_loss, train_acc,val_loss, val_acc))

        history_file.write("{}\t{}\t{}\t{}\t{}\n".format(t,
                                                                 train_loss, train_acc,
                                                                 val_loss, val_acc))
        model_checkpoint.update(val_loss)
        tensorboard_writer.add_scalar('metrics/train_loss', train_loss, t)
        tensorboard_writer.add_scalar('metrics/train_acc',  train_acc, t)
        tensorboard_writer.add_scalar('metrics/val_loss', val_loss, t)
        tensorboard_writer.add_scalar('metrics/val_acc',  val_acc, t)

if args.action == "test" :
    utils.test(model, test_loader, device, num_classes)

    print("Test completed.")
