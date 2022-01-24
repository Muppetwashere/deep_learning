from matplotlib.colors import Colormap
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from datasets import * 
from utils import *
from logs import * 

import os

# Path to train and test data
DATA_TRAIN_PATH = '/mounts/Datasets1/ChallengeDeep/train'
DATA_TEST_PATH = '/mounts/Datasets1/ChallengeDeep/test/imgs'

print(os.path.exists(DATA_TRAIN_PATH))

# Hyperparams
num_epochs = 10
num_workers = 4
batch_size = 32
img_size = 224
num_class = 86
lr = 1e-3
top_logdir = "./logs"

# ratio dataset split
train_ratio = 0.8
eval_ratio = 0.15
test_ratio = 0.05

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load dataset
plankton_dataset = PlanktonDataset(DATA_TRAIN_PATH, final_size = img_size)


# Split dataset
nb_train = int(train_ratio * len(plankton_dataset))+1
nb_eval = int(eval_ratio * len(plankton_dataset))
nb_test = int(test_ratio * len(plankton_dataset))
train_dataset, eval_dataset, test_dataset = torch.utils.data.dataset.random_split(plankton_dataset, (nb_train, nb_eval, nb_test))

# DataLoaders
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
test_dataloader = DataLoader(dataset=test_dataset, pin_memory=True, batch_size=batch_size, shuffle=True, num_workers=num_workers)

print("The train set contains {} images, in {} batches".format(len(train_dataloader.dataset), len(train_dataloader)))
print("The evaluation set contains {} images, in {} batches".format(len(eval_dataloader.dataset), len(eval_dataloader)))
print("The test set contains {} images, in {} batches".format(len(test_dataloader.dataset), len(test_dataloader)))

# example
example = iter(train_dataloader)
img,label = example.next()


# classes names
classes = []
subfolders = glob.glob(DATA_TRAIN_PATH + '/*')
subfolders.sort()
for subfolder in subfolders:
    classes.append(subfolder.split('/')[-1])

nsamples=10
fig=plt.figure(figsize=(20,5),facecolor='w')
for i in range(nsamples):
    ax = plt.subplot(1,nsamples, i+1)
    plt.imshow(img[i, 0, :, :], vmin=0, vmax=1.0)
    ax.set_title("{}".format(classes[label[i]]), fontsize=15)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

#plt.savefig('fashionMNIST_samples.png', bbox_inches='tight')
#plt.show()