import torchvision.transforms 
import torch.utils.data
import os.path
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
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


class composed_transforms(torchvision.transforms.transforms.Compose):
    '''Contain the list and the compose transforms for the validation and the train set.
    The list of possible transformations for validation are :
        - greyscale
        - invert
        - resize
        - normalization 
    For Train you can add Data Augmentation :
        - rotate
        - flip
        - blur'''
    def __init__(self, greyscale=True, invert=True, resize=True, normalization=True, rotate=True, flip=True, blur=False):
        
        valid_list_transforms = []
        if greyscale:
            valid_list_transforms+=[torchvision.transforms.Grayscale(num_output_channels=1)]
        if invert:
            valid_list_transforms+=[torchvision.transforms.functional.invert]
        if resize:
            valid_list_transforms+=[torchvision.transforms.Resize((300,300))]
        if normalization:
            valid_list_transforms+=[torchvision.transforms.Normalize(mean=[0.0988],std=[0.1444])]
        self.valid_transforms = torchvision.transforms.Compose(valid_list_transforms)
        self.valid_transforms_list = valid_list_transforms

        train_list_transforms = valid_list_transforms
        if rotate:
            train_list_transforms+=[torchvision.transforms.RandomRotation((0, 360))]
        if flip:
            train_list_transforms+=[torchvision.transforms.RandomHorizontalFlip(p=0.5)]
        if blur:
            train_list_transforms+=[torchvision.transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 1))]
        self.train_transforms = torchvision.transforms.Compose(train_list_transforms)
        self.train_transforms_list = train_list_transforms




########################
##### Loading Data #####
########################
valid_ratio = 0.2

##### Load learning data

# Whole dataset
train_path = "/opt/ChallengeDeep/train/"

# Little sample to try
#train_path = "/usr/users/gpusdi1/gpusdi1_49/Bureau/sample_train"

print('data train loading...')
dataset = torchvision.datasets.ImageFolder(train_path, train_composed_transforms)
print('data train loaded')

##### Load test data
test_path = "/opt/ChallengeDeep/test/"

#print('data test loading...')
#dataset = torchvision.datasets.ImageFolder(train_path, valid_composed_transforms)
#print('data test loaded')

# Train test split
nb_train = int((1.0 - valid_ratio) * len(dataset))
nb_valid = len(dataset)-nb_train
train_dataset, valid_dataset = torch.utils.data.dataset.random_split(dataset, [nb_train, nb_valid])
print('data split')

##### Generating Loaders #####
num_workers = 4
batch_size = 64

# training loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           num_workers=num_workers,
                                           shuffle=True)

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

############################
### Compute mean and std ###
############################

def compute_global_mean_std(loader):
    print("Computing train data mean")
    # Compute the mean over minibatches
    mean_img = None
    i = 0
    num_img = 0
    for imgs, _ in loader:
        i += 1
        if mean_img is None:
            print("step1")
            mean_img = torch.zeros_like(imgs[0])
            print("step2")
        print("{}/{}".format(i, len(loader)))
        mean_img += imgs.sum(dim=0)
        num_img += len(imgs)
        print((mean_img/num_img).view(1,-1).mean())
    mean_img /= len(loader.dataset)
    real_mean = mean_img.view(1,-1).mean()

    print("Computing train data std")
    # Compute the std over minibatches
    std_img = torch.zeros_like(mean_img)
    for imgs, _ in loader:
        std_img += ((imgs - mean_img)**2).sum(dim=0)
    std_img /= len(loader.dataset)
    std_img = torch.sqrt(std_img)

    # Set the variance of pixels with no variance to 1
    # Because there is no variance
    # these pixels will anyway have no impact on the final decision
    std_img[std_img == 0] = 1
    real_std = std_img.view(1,-1).mean()

    return real_mean, real_std


#############################
##### Display Some data #####
#############################

def display_data(n_samples):
    class_names = dataset.classes
    imgs, labels = next(iter(train_loader))

    fig = plt.figure(figsize=(30, 30), facecolor='w')

    for col in range(n_samples):
        ax = plt.subplot(2, n_samples, col+1)
        plt.imshow(imgs[col, 0, :, :], vmin=0, vmax=1.0, cmap=cm.gray)
        ax.set_title("{}".format(class_names[labels[0]]), fontsize=15)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.savefig('plancton_alban.png', bbox_inches='tight')
    print('saved_images')
    plt.show()

#############################
#####   Sampling Data   #####
#############################

def sampler_(dataset,train_counts):
    start_time = time.time()
    num_samples = len(dataset)
    labels = [dataset[item][1] for item in tqdm(range(len(dataset)))]
    label_end_time = time.time()
    print('got labels in {} s'.format(label_end_time-start_time))

    class_weights = 1./ np.array(train_counts)
    classw_end_time = time.time()
    print('got class weights in {} s'.format(classw_end_time-label_end_time))
    weights = class_weights[labels]
    weights = torch.from_numpy(weights)
    weight_end_time = time.time()
    print('got final weights in {} s'.format(weight_end_time-classw_end_time))
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, num_samples)
    return sampler