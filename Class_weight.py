import os
import numpy as np 
import matplotlib.pyplot as plt

print(os.path.exists("/opt/ChallengeDeep/train/"))

train_path = "/opt/ChallengeDeep/train/"

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

print(number_of_images)
