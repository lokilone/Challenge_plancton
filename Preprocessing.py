import torchvision.transforms 
import torch.utils.data
import os.path
import matplotlib.pyplot as plt
import matplotlib.cm as cm

###############################
##### Defining transforms #####
###############################


params = {"padding": (300, 300)}

# pad smaller images to desired size
pad = torchvision.transforms.Pad(padding=params["padding"], fill=0)
centercrop = torchvision.transforms.CenterCrop(300)  # Crop images to be of same 300x300
# Images are B/W but in 3 channels, we only need one
greyscale = torchvision.transforms.Grayscale(num_output_channels=1)
# Negative
invert = torchvision.transforms.functional.invert
# Resize
resize = torchvision.transforms.Resize((300,300))


### Data Augmentation
rotate = torchvision.transforms.RandomRotation((0, 360))


# Compose transforms
#composed_transforms = torchvision.transforms.Compose([greyscale, invert, resize, torchvision.transforms.ToTensor(), rotate])
composed_transforms = torch.nn.Sequential(greyscale, invert, resize, torchvision.transforms.ToTensor(), rotate)


# Create transformer class
class DatasetTransformer(torch.utils.data.Dataset):

    def __init__(self, base_dataset, transforms):
        self.base_dataset = base_dataset
        self.transform = transforms

    def __getitem__(self, index):
        img, target = self.base_dataset[index]
        return self.transform(img), target

    def __len__(self):
        return len(self.base_dataset)


########################
##### Loading Data #####
########################
print(os.path.exists("/mounts/Datasets1/ChallengeDeep/train/"))
valid_ratio = 0.2

# Load learning data
#train_path = "/mounts/Datasets1/ChallengeDeep/train/"
train_path = "/usr/users/gpusdi1/gpusdi1_49/Bureau/sample_train"

print('data loading...')
dataset = torchvision.datasets.ImageFolder(train_path, composed_transforms)
print('data loaded')

# Train test split
nb_train = int((1.0 - valid_ratio) * len(dataset))
nb_valid = len(dataset)-nb_train
train_dataset, valid_dataset = torch.utils.data.dataset.random_split(dataset, [
                                                                     nb_train, nb_valid])
print('data split')


### Augmentation
augmentation = torchvision.transforms.AutoAugment

train_dataset = augmentation(train_dataset)

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


#############################
##### Display Some data #####
#############################
n_samples = 30

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
print('saved_iamges')
plt.show()
