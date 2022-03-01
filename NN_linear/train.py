import src.rollwear_lib_object.data.datasets as datasets
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import os
from sklearn.preprocessing import MinMaxScaler
import numpy as np

import utils
import pytorch_model

# Info for monitoring (to be open in another terminal at the root)
print("tensorboard --logdir ./logs")

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
input_size = 36*306 # [16,306,36] i.e [batch_num, nb of strips, nb of parameters]
#hidden_size = 500
num_epochs = 100
batch_size = 16
learning_rate = 0.001

# Loading legacy dataset
dataset = datasets.StripsWearCenter(f6=True, top=False)
x_train, x_valid, x_test, y_train, y_valid, y_test = dataset.get_train_var()

print("train {}, valid {}, test {}, sum {}".format(len(x_train),len(x_test),len(x_valid),(len(x_train)+len(x_test)+len(x_valid))))

# Convert to pytorch tensors 

y_train = torch.tensor(y_train.to_numpy()) #seems to be a panda dataset 
x_train = torch.tensor(x_train) #isn't a panda dataset
y_test = torch.tensor(y_test.to_numpy()) #seems to be a panda dataset 
x_test = torch.tensor(x_test) #isn't a panda dataset
y_valid = torch.tensor(y_valid.to_numpy()) #seems to be a panda dataset 
x_valid = torch.tensor(x_valid) #isn't a panda dataset

# Create concatenated dataset with (features,target)

x_y_train = torch.utils.data.TensorDataset(x_train,y_train)
x_y_test = torch.utils.data.TensorDataset(x_test,y_test)
x_y_valid = torch.utils.data.TensorDataset(x_valid,y_valid)

# Transform the dataset

class DatasetTransformer(torch.utils.data.Dataset):

    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __getitem__(self, index):
        strip_data, target = self.base_dataset[index]
        return self.transform(strip_data), self.transform(target)

    def __len__(self):
        return len(self.base_dataset)


x_y_train = DatasetTransformer(x_y_train,transforms.Lambda(lambda x: (x)))
x_y_test = DatasetTransformer(x_y_test,transforms.Lambda(lambda x: (x)))
x_y_valid  = DatasetTransformer(x_y_valid,transforms.Lambda(lambda x: (x)))

# Data loaders

x_y_train_loader = torch.utils.data.DataLoader(dataset=x_y_train, 
                                           batch_size=batch_size, 
                                           shuffle=True)

x_y_test_loader = torch.utils.data.DataLoader(dataset=x_y_test, 
                                           batch_size=batch_size, 
                                           shuffle=True)

x_y_valid_loader = torch.utils.data.DataLoader(dataset=x_y_valid, 
                                           batch_size=batch_size, 
                                           shuffle=True)

# Example to play with

examples = iter(x_y_train_loader)
data,target = examples.next()
print(target)


# Model
model = pytorch_model.TinyModel().to(device)

# Loss and optimizer
f_loss = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

# Where to save
top_logdir = "./logs"

if not os.path.exists(top_logdir):
    os.mkdir(top_logdir)

logdir = utils.generate_unique_logpath(top_logdir, "linear")

print("Logging to {}".format(logdir))

if not os.path.exists(logdir):
    os.mkdir(logdir)

# Monitoring 
tensorboard_writer   = SummaryWriter(log_dir = logdir) # start tensorboard --logdir ./logs in the terminal

# Define the callback object for saving
model_checkpoint = utils.ModelCheckpoint(logdir + "/best_model.pt", model)

# Train the model

for epoch in range(num_epochs):

    print("Epoch {}".format(epoch))
    utils.train(model,x_y_train_loader,f_loss,optimizer,device)
    train_loss = utils.test(model, x_y_train_loader, f_loss, device)
    #print(" Train       Loss : {:.4f}".format(train_loss))

    test_loss = utils.test(model, x_y_test_loader, f_loss, device)
    print(" Test       : Loss : {:.4f}".format(test_loss))

    val_loss = utils.test(model, x_y_valid_loader, f_loss, device)
    print(" Validation : Loss : {:.4f}".format(val_loss))
    model_checkpoint.update(val_loss)

    tensorboard_writer.add_scalar('metrics/val_loss', val_loss, epoch)
    tensorboard_writer.add_scalar('metrics/test_loss', test_loss, epoch)

