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

class ComposedTransforms():
    '''Contain the list and the compose transforms for the validation and the train set.
    The list of possible transformations for validation are :
        - greyscale
        - invert
        - resize
        - normalization \n
    For Train you can add Data Augmentation :
        - rotate
        - flip
        - blur'''
    def __init__(self, mean = 0.0988, std=0.1444):
        self.mean = mean
        self.std = std
        self.transform_compendium = {
            'greyscale' : torchvision.transforms.Grayscale(num_output_channels=1),
            'greyscale3' : torchvision.transforms.Grayscale(num_output_channels=3),
            'invert' : torchvision.transforms.functional.invert,
            'resize' : torchvision.transforms.Resize((300,300)),
            'centercrop' : torchvision.transforms.CenterCrop(300),
            'totensor' : torchvision.transforms.ToTensor(),
            'normalization' : torchvision.transforms.Normalize(mean=[self.mean],std=[self.std]),
            'rotate' : torchvision.transforms.RandomRotation((0, 360)),
            'flip' : torchvision.transforms.RandomHorizontalFlip(p=0.5),
            'blur' : torchvision.transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 1))
            }

    def test_transforms(self, preprocessing_seq = None):
        test_list_transforms = []
        for op in preprocessing_seq :
            test_list_transforms.append(self.transform_compendium[op])
        return torchvision.transforms.Compose(test_list_transforms)

    def train_transforms(self, preprocessing_seq = None, augmentation_seq = None):
        train_list_transforms = []
        for op in preprocessing_seq :
            train_list_transforms.append(self.transform_compendium[op])
        for op in augmentation_seq : 
            train_list_transforms.append(self.transform_compendium[op])
        return torchvision.transforms.Compose(train_list_transforms)


class DatasetTransformer(torch.utils.data.Dataset):

    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.base_dataset[index]
        return self.transform(img), target

    def __len__(self):
        return len(self.base_dataset)


#############################
#####   Sampling Data   #####
#############################

def sampler_(dataset,train_counts):
    start_time = time.time()
    num_samples = len(dataset)
    labels = [dataset[item][1] for item in range(len(dataset))]
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


########################
##### Loading Data #####
########################

class DataLoader(torch.utils.data.dataloader.DataLoader):

    def __init__(self, train_valid_path="/opt/ChallengeDeep/train/", valid_ratio=0.2, test_path = "/opt/ChallengeDeep/test/", num_workers = 4, batch_size = 256):
        self.train_valid_path = train_valid_path
        self.test_path = test_path
        self.valid_ratio = valid_ratio
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None

    def Load_Train_Valid(self, train_composed_transforms=ComposedTransforms().train_transforms(), valid_composed_transforms=ComposedTransforms().valid_transforms(), sampling=False):
        # Load Train_Validation set
        print('Loading Train_Validation set')
        dataset = torchvision.datasets.ImageFolder(self.train_valid_path)

        # Train test split
        print('Splitting Data')
        nb_train = round((1.0 - self.valid_ratio) * len(dataset))
        nb_valid = len(dataset)-nb_train
        train_subset, valid_subset = torch.utils.data.dataset.random_split(dataset, [nb_train, nb_valid])

        # Transform Train
        print("Transforming Train set")
        self.train_dataset = DatasetTransformer(train_subset, train_composed_transforms)
        if sampling:
            print('Creating sampler')
            train_counts = [1983, 243570, 214, 638, 1326, 1328, 23797, 5781, 289, 18457, 536, 185, 1045, 1210, 
            686, 5570, 8402, 3060, 168, 953, 4764, 2825, 3242, 78, 37, 3916, 98, 8200, 576, 19225, 686, 4213, 336, 188, 
            1459, 1869, 180000, 3538, 1091, 6056, 142, 33147, 2085, 170, 308, 14799, 4609, 156, 3900, 3983, 3111, 1988, 
            5079, 244, 6368, 757, 1289, 12636, 42096, 10008, 3465, 269, 457, 10038, 8213, 372, 2314, 234, 590, 15431, 12954, 
            4391, 1285, 5604, 6996, 53387, 235, 632, 11490, 88, 2589, 2517, 388, 2086, 172, 727]
            sampler = sampler_(self.train_dataset,train_counts)
            print('Sampler created')
            self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                            batch_size=self.batch_size,
                                            num_workers=self.num_workers,
                                            sampler=sampler)
        else:
            self.train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                            batch_size=self.batch_size,
                                            num_workers=self.num_workers,
                                            shuffle=True)

        # Transform Validation
        print("Transforming Validation set")
        self.valid_dataset = DatasetTransformer(valid_subset, valid_composed_transforms)
        self.valid_loader = torch.utils.data.DataLoader(dataset=self.valid_dataset,
                                            batch_size=self.batch_size,
                                            num_workers=self.num_workers,
                                            shuffle=False)

        # Data Inspect
        print("The train set contains {} images, in {} batches".format(len(self.train_loader.dataset), len(self.train_loader)))
        print("The validation set contains {} images, in {} batches".format(len(self.valid_loader.dataset), len(self.valid_loader)))



    def Load_Test(self, test_composed_transforms=ComposedTransforms().valid_transforms()):
        # Load Testing set
        print('Loading Test set')
        test_dataset = torchvision.datasets.ImageFolder(self.test_path)

        # Transform Test
        print("Transforming Test set")
        self.test_dataset  = DatasetTransformer(test_dataset, test_composed_transforms)
        self.test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                            batch_size=self.batch_size,
                                            num_workers=self.num_workers,
                                            shuffle=False)

        # Data Inspect
        print("The train set contains {} images, in {} batches".format(len(self.test_loader.dataset), len(self.test_loader)))


# Little sample to try
# train_path = "/usr/users/gpusdi1/gpusdi1_49/Bureau/sample_train"


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

def display_data(n_samples, loader, labels = True, class_names=np.arange(86)):
    if labels :
        imgs, labels = next(iter(loader))

        fig = plt.figure(figsize=(30, 30), facecolor='w')

        for col in range(n_samples):
            ax = plt.subplot(2, n_samples, col+1)
            plt.imshow(imgs[col, 0, :, :], vmin=0, vmax=1.0, cmap=cm.gray)
            ax.set_title("{}".format(class_names[labels[0]]), fontsize=15)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.savefig('img_train.png', bbox_inches='tight')
        print('Saved Train Images')
        plt.show()
    else:
        imgs = next(iter(loader))

        fig = plt.figure(figsize=(30, 30), facecolor='w')

        for col in range(n_samples):
            ax = plt.subplot(2, n_samples, col+1)
            plt.imshow(imgs[col][0, 0, :, :], vmin=0, vmax=1.0, cmap=cm.gray)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.savefig('img_test.png', bbox_inches='tight')
        print('Saved Test Images')


### Define what transformations you want to make
# On train and test
mean = 0.0988
std = 0.1444
resize = True
normalization = True

# Only on train
rotate = False
flip = False
blur = False

Transform = ComposedTransforms(mean = mean, std=std)
test_composed_transforms = ComposedTransforms().valid_transforms(resize=resize, normalization=normalization)
train_composed_transforms = ComposedTransforms().train_transforms(resize=resize, normalization=normalization, rotate=rotate, flip=flip, blur=blur)

### Load data
batch_size = 256

Data_Loader = DataLoader(batch_size = batch_size)
Data_Loader.Load_Train_Valid(train_composed_transforms=train_composed_transforms, valid_composed_transforms=test_composed_transforms)
Data_Loader.Load_Test(test_composed_transforms=test_composed_transforms)

train_loader = Data_Loader.train_loader
test_loader = Data_Loader.test_loader

### Display data to check that everything is fine
display_data(10, train_loader, labels=True)
#display_data(10, test_loader, labels=False)