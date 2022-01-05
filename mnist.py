import torch
import torchvision                                                       
import torchvision.transforms as transforms
import data
import models
import utils

import os.path
import torch.nn as nn


 ############################################################################################ Datasets

dataset_dir = os.path.join(os.path.expanduser("~"), 'Datasets', 'FashionMNIST')
valid_ratio = 0.2  # Going to use 80%/20% split for train/valid

############################################################################################ Dataloaders
num_threads = 4     # Loading the dataset is using 4 CPU threads
batch_size  = 128   # Using minibatches of 128 samples

train_loader, valid_loader, test_loader = data.load_fashion_mnist(valid_ratio, batch_size, num_threads)

print("The train set contains {} images, in {} batches".format(len(train_loader.dataset), len(train_loader)))
print("The validation set contains {} images, in {} batches".format(len(valid_loader.dataset), len(valid_loader)))
print("The test set contains {} images, in {} batches".format(len(test_loader.dataset), len(test_loader)))


model = models.LinearNet(1*28*28, 10)

use_gpu = torch.cuda.is_available()
if use_gpu:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


model.to(device)

f_loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

epochs = 10

for t in range(epochs):
    print("Epoch {}".format(t))
    utils.train(model, train_loader, f_loss, optimizer, device)

    val_loss, val_acc = utils.test(model, valid_loader, f_loss, device)
    print(" Validation : Loss : {:.4f}, Acc : {:.4f}".format(val_loss, val_acc))