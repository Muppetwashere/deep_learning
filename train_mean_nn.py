from matplotlib import markers
import src.rollwear_lib_object.data.datasets as datasets
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import os
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import utils
import pytorch_model

# Info for monitoring (to be open in another terminal at the root)
print("tensorboard --logdir ./logs")

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
input_size = 36*306 # [16,8] i.e [batch_num, nb of parameters]
#hidden_size = 500
num_epochs = 250
batch_size = 32
learning_rate = 0.001
name_for_saving = "mean_SELU_ReLU_sigmoid"

# Loading legacy dataset
dataset = datasets.MeanWearCenter(f6=True, top=False, mode="selected")
x_train, x_valid, _, y_train, y_valid, _ = dataset.get_train_var()

#print(y_train)

# Denormalisation of output
#y_train = dataset.denormalize(y = y_train)
#print(y_train)
#y_valid = dataset.denormalize(y = y_valid)

print("train {}, valid {}, sum {}".format(len(x_train),len(x_valid),(len(x_train)+len(x_valid))))

print("number of parameters mesured on the rolls : ", len(dataset.input.mode))

# Dataset description

if (False):
    print("\nTarget dataset description : \n")
    print(y_train.describe())

    df = pd.DataFrame(x_train,columns=dataset.get_x_dataframe().columns) #il manque une colonne dans dataset.input.column_names

    print("\n Input dataset description : \n")
    print("\n This is the original dataset : \n")
    print(dataset.get_x_dataframe())
    print("\n This is the normalized dataset : \n")
    print(df)

# Convert to pytorch tensors

y_train = torch.tensor(y_train.to_numpy()) #is a panda dataset 
x_train = torch.tensor(x_train) #isn't a panda dataset
y_valid = torch.tensor(y_valid.to_numpy()) #is a panda dataset 
x_valid = torch.tensor(x_valid) #isn't a panda dataset

# Create concatenated dataset with (features,target)

x_y_train = torch.utils.data.TensorDataset(x_train,y_train)
x_y_valid = torch.utils.data.TensorDataset(x_valid,y_valid)

# Transform the dataset

class DatasetTransformer(torch.utils.data.Dataset):

    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __getitem__(self, index):
        strip_data, target = self.base_dataset[index]
        return self.transform(strip_data), target

    def __len__(self):
        return len(self.base_dataset)

x_y_train = DatasetTransformer(x_y_train,transforms.Lambda(lambda x: (x)))
x_y_valid  = DatasetTransformer(x_y_valid,transforms.Lambda(lambda x: (x)))

# Data loaders

x_y_train_loader = torch.utils.data.DataLoader(dataset=x_y_train, 
                                           batch_size=batch_size, 
                                           shuffle=False)

x_y_valid_loader = torch.utils.data.DataLoader(dataset=x_y_valid, 
                                           batch_size=batch_size, 
                                           shuffle=False)

# Example to play with

examples = iter(x_y_train_loader)
data,target = examples.next()
print(data)


# Model
model = pytorch_model.LinearModel_mean().to(device)

# Loss and optimizer
f_loss = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

# Where to save
top_logdir = "./logs"

if not os.path.exists(top_logdir):
    os.mkdir(top_logdir)

logdir = utils.generate_unique_logpath(top_logdir, name_for_saving)

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
    print(" Train       Loss : {:.4f}".format(train_loss))

    val_loss = utils.test(model, x_y_valid_loader, f_loss, device)
    print(" Validation : Loss : {:.4f}".format(val_loss))
    model_checkpoint.update(val_loss)

    tensorboard_writer.add_scalar('metrics/val_loss', val_loss, epoch)
    tensorboard_writer.add_scalar('metrics/train_loss', train_loss, epoch)

if (True): # loading best model 

    model_path = logdir + "/best_model.pt"

    model.load_state_dict(torch.load(model_path))

    # Switch to eval mode 
    model.eval()

    print("The log dir is {}".format(logdir + "/best_model.pt"))
    val_loss, ground_truth_valid, prediction_valid =  utils.Load_best_prediction(model,logdir + "/best_model.pt",x_y_valid_loader,f_loss,device)
    train_loss, ground_truth_train, prediction_train =  utils.Load_best_prediction(model,logdir + "/best_model.pt",x_y_train_loader,f_loss,device)

    print(" BEST_MODEL Validation : MEAN Loss : {:.4f}".format(val_loss))
    print(" BEST_MODEL training :   MEAN Loss : {:.4f}".format(val_loss))

    # Display 

    print("LENGTH OF VAL : {}".format(len(prediction_valid)))
    

    # De-normalisation and setting to µm
    denorm = dataset.denormalize
    y_train, y_val, pred_train, pred_val = denorm(y_train), denorm(y_valid), denorm(prediction_train), denorm(prediction_valid)
    y_train, y_val, pred_train, pred_val = 1000 * y_train, 1000 * y_val, 1000 * pred_train, 1000 * pred_val
    
    # Plot

    fig, ax = plt.subplots()
    ax.scatter(y_val,pred_val,s=2,c="brown")
    ax.scatter(y_train,pred_train,s=2)
    ax.set_xlim([0, 600])
    ax.set_ylim([0, 600])
    ax.set_title("prediction / ground truth (in mm)")
    plt.show()