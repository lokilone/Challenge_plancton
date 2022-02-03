import torch
import torchvision
import torchvision.datasets as datasets                                                     
import torchvision.transforms 
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import os.path


params = {"padding" : (300,300)}

##### Defining transforms #####
pad = torchvision.transforms.Pad(padding = params["padding"], fill = 0) # pad smaller images to desired size
centercrop = torchvision.transforms.CenterCrop(300) # Crop images to be of same 300x300

#Compose transforms
composed_transforms = torchvision.transforms.Compose([centercrop,
                torchvision.transforms.ToTensor()])

##### Loading Data #####
train_path = "/usr/users/gpusdi1/gpusdi1_49/Bureau/train/"
test_path = "/usr/users/gpusdi1/gpusdi1_49/Bureau/test"
valid_ratio = 0.2

train_valid_dataset = datasets.ImageFolder(train_path, composed_transforms)

# Train valid split 
nb_train = round((1.0 - valid_ratio) * len(train_valid_dataset))
nb_valid =  len(train_valid_dataset)-nb_train
train_dataset, valid_dataset = torch.utils.data.dataset.random_split(train_valid_dataset, [nb_train, nb_valid])

# Test set
test_dataset = datasets.ImageFolder(test_path, composed_transforms)

print(len(train_dataset))
print(len(valid_dataset))
print(len(test_dataset))


# #Create transformer class
# class DatasetTransformer(torch.utils.data.Dataset):
    
#     def __init__(self, base_dataset, transforms):
#         self.base_dataset = base_dataset
#         self.transform = transforms

#     def __getitem__(self, index):
#         img, target = self.base_dataset[index]
#         return self.transform(img), target

#     def __len__(self):
#         return len(self.base_dataset)

# ##### Generating Loaders #####
# num_workers = 4
# batch_size = 64

# # training loader
# train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
#                                             batch_size = batch_size,
#                                             num_workers = num_workers,
#                                             shuffle = True)

# # validation loader
# valid_loader = torch.utils.data.DataLoader(dataset = valid_dataset,
#                                             batch_size = batch_size,
#                                             num_workers= num_workers,
#                                             shuffle = True)

# # Data Inspect
# print("The train set contains {} images, in {} batches".format(len(train_loader.dataset), len(train_loader)))
# print("The validation set contains {} images, in {} batches".format(len(valid_loader.dataset), len(valid_loader)))

# ##### Display Some data #####
# n_samples = 10

# class_names = dataset.classes
# imgs, labels = next(iter(train_loader))

# fig=plt.figure(figsize=(20,5),facecolor='w')
# for i in range(n_samples) : 
#     ax = plt.subplot(1,n_samples, i+1)
#     plt.imshow(imgs[i, 0, :, :], vmin=0, vmax=1.0, cmap=cm.gray)
#     ax.set_title("{}".format(class_names[labels[0]]), fontsize=15)
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)

# plt.savefig('plancton.png', bbox_inches='tight')
# plt.show()