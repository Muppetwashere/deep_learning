import src.rollwear_lib_object.data.datasets as datasets
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import os
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt

import utils
import pytorch_model

# Info for monitoring (to be open in another terminal at the root)
print("tensorboard --logdir ./logs")

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
input_size = 36*306 # [16,306,36] i.e [batch_num, nb of strips, nb of parameters]
#hidden_size = 500
num_epochs = 250
batch_size = 16
learning_rate = 0.001
name_for_saving = "ExpertModel"

# Loading legacy dataset
dataset = datasets.StripsWearCenter(f6=True, top=False)
x_train, x_valid, x_test, y_train, y_valid, y_test = dataset.get_train_var()

print("train {}, valid {}, test {}, sum {}".format(len(x_train),len(x_test),len(x_valid),(len(x_train)+len(x_test)+len(x_valid))))

print("number of parameters mesured on the rolls : ", len(dataset.input.columns_name))

#print("name of the parameters mesured on the rolls : ", dataset.input.columns_name)

# Convert to pytorch tensors

y_train = torch.tensor(y_train.to_numpy()) #seems to be a panda dataset 
x_train = torch.tensor(x_train) #isn't a panda dataset
y_test = torch.tensor(y_test.to_numpy()) #seems to be a panda dataset 
x_test = torch.tensor(x_test) #isn't a panda dataset
y_valid = torch.tensor(y_valid.to_numpy()) #seems to be a panda dataset 
x_valid = torch.tensor(x_valid) #isn't a panda dataset

print("SHAPE OF X_TRAIN{}".format(x_train.shape))
print([x_train[0][0][i].item() for i in range(len(x_train[0][0]-1))])

# Normalisation checker si c'est déjà normaliser 
y_train = dataset.denormalize(y = y_train)
y_valid = dataset.denormalize(y = y_valid)
y_test = dataset.denormalize(y = y_test)

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

mean_train_strips,mean_train_target, std_train_strips, std_train_target = utils.get_mean_std(x_y_train)

x_y_train = DatasetTransformer(x_y_train,transforms.Lambda(lambda x: (x)))
x_y_test = DatasetTransformer(x_y_test,transforms.Lambda(lambda x: (x)))
x_y_valid  = DatasetTransformer(x_y_valid,transforms.Lambda(lambda x: (x)))

# Data loaders

x_y_train_loader = torch.utils.data.DataLoader(dataset=x_y_train, 
                                           batch_size=batch_size, 
                                           shuffle=False) ##True

x_y_test_loader = torch.utils.data.DataLoader(dataset=x_y_test, 
                                           batch_size=batch_size, 
                                           shuffle=True)

x_y_valid_loader = torch.utils.data.DataLoader(dataset=x_y_valid, 
                                           batch_size=batch_size, 
                                           shuffle=True)

# Example to play with
if (False):
    examples = iter(x_y_train_loader)
    data,target = examples.next()   
    print(target)

    print(data.shape)
    fwl_mult_factors = data.permute(2,0,1) # reshape to be able to select fwl shape is [36,16,306]
    fwl_mult_factors = fwl_mult_factors [-1] # select fwl (the 36th parameter), shape is [16,306]
    fwl_mult_factors = fwl_mult_factors.unsqueeze(2) # add a 1 dim as dimension 2, shape here is [16,306,1]
    print(fwl_mult_factors[0,1])
    print(data[0,1])


    models=torch.nn.Sequential(
                nn.Linear(36,16),
                nn.Linear(16,4),
                nn.ReLU(),
                nn.Linear(4,1),
                nn.Sigmoid(),
    )


    print(models(data.float()).shape)
    print("mult{}".format(data[0,1,-1]*models(data.float())[0,1]))
    k_coefficient = models(data.float()) # shape here is [16,36,1]
    print("coef{}".format(k_coefficient))
    print("multfator{}".format(fwl_mult_factors[0,1]))
    x = fwl_mult_factors * k_coefficient # this multiplies term-to-term, shape here is [16,36,1] -> [16,36,1]
    print("same{}".format(x[0,1]))
    x = torch.sum(x,dim=1)
    print(x)

    sm = [fwl_mult_factors.select(1,i) * k_coefficient.select(1,i) for i in range(0,fwl_mult_factors.shape[1]-1)] # this multiplies term-to-term, shape here is [16,306,1] -> [16,306,1]
    x = sum(sm) #shape here is [16,1]
    print(x)

# Model
model = pytorch_model.ExpertModel().to(device)

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

# For display 
ground_truth_display = []
prediction_display = []

# Train the model
for epoch in range(num_epochs):

    print("Epoch {}".format(epoch))
    utils.train(model,x_y_train_loader,f_loss,optimizer,device)

    train_loss, ground_truth_train, prediction_train = utils.test(model, x_y_train_loader, f_loss, device)
    #print(" Train       Loss : {:.4f}".format(train_loss))

    test_loss, ground_truth_test, prediction_test = utils.test(model, x_y_test_loader, f_loss, device)
    print(" Test       : Loss : {:.4f}".format(test_loss))

    val_loss, ground_truth_val, prediction_val = utils.test(model, x_y_valid_loader, f_loss, device)
    print(" Validation : Loss : {:.4f}".format(val_loss))

    ground_truth_display += ground_truth_val
    prediction_display += prediction_val

    model_checkpoint.update(val_loss)

    tensorboard_writer.add_scalar('metrics/val_loss', val_loss, epoch)
    tensorboard_writer.add_scalar('metrics/test_loss', test_loss, epoch)

# Display 
fig, ax = plt.subplots()
ax.scatter(ground_truth_display,prediction_display)
ax.set_aspect(aspect='equal')
ax.set_title("prediction / ground truth (in mm)")
plt.show()

if (False): # loading best model 

    model_path = logdir + "/best_model.pt"

    model.load_state_dict(torch.load(model_path))

    # Switch to eval mode 
    model.eval()

    val_loss = utils.test(model, x_y_valid_loader, f_loss, device)

    print("same val {:.4f}".format(val_loss))


    print("The log dir is {}".format(logdir + "/best_model.pt"))
    val_loss, ground_truth, prediction =  utils.Load_best_prediction(model,logdir + "/best_model.pt",x_y_valid_loader,f_loss,device)

    print(" BEST_MODEL Validation : MEAN Loss : {:.4f}".format(val_loss))

