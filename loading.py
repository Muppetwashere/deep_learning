import torch
import os

import models
import utils
import data


## example of loading a model 
model_path = "./logs/linear_0/best_model.pt"
model = models.LinearNet(1*28*28, 10)

# Datasets
dataset_dir = os.path.join(os.path.expanduser("~"), 'Datasets', 'FashionMNIST')
valid_ratio = 0.2  # Going to use 80%/20% split for train/valid

# Dataloaders
num_threads = 4     # Loading the dataset is using 4 CPU threads
batch_size  = 128   # Using minibatches of 128 samples

train_loader, valid_loader, test_loader = data.load_fashion_mnist(valid_ratio, batch_size, num_threads)

print("The train set contains {} images, in {} batches".format(len(train_loader.dataset), len(train_loader)))
print("The validation set contains {} images, in {} batches".format(len(valid_loader.dataset), len(valid_loader)))
print("The test set contains {} images, in {} batches".format(len(test_loader.dataset), len(test_loader)))

# GPU settings
use_gpu = torch.cuda.is_available()

if use_gpu:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model = model.to(device)

model.load_state_dict(torch.load(model_path))

# Loss  function
f_loss = torch.nn.CrossEntropyLoss()

# Switch to eval mode 
model.eval()

test_loss, test_acc = utils.test(model, test_loader, f_loss, device)
print(" Test       : Loss : {:.4f}, Acc : {:.4f}".format(test_loss, test_acc))