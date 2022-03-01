import torch
import numpy as np
import os

def train(model, loader, f_loss, optimizer, device):
    """
    Train a model for one epoch, iterating over the loader
    using the f_loss to compute the loss and the optimizer
    to update the parameters of the model.

    Arguments :

        model     -- A torch.nn.Module object
        loader    -- A torch.utils.data.DataLoader
        f_loss    -- The loss function, i.e. a loss Module
        optimizer -- A torch.optim.Optimzer object
        device    -- a torch.device class specifying the device
                    used for computation

    Returns :
    """

    # We enter train mode. This is useless for the linear model
    # but is important for layers such as dropout, batchnorm, ...
    model.train()
    for i, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Compute the forward pass through the network up to the loss
        outputs = model(inputs.float()) #if no float: error
        loss = f_loss(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test(model, loader, f_loss, device):
    """
    Test a model by iterating over the loader

    Arguments :

        model     -- A torch.nn.Module object
        loader    -- A torch.utils.data.DataLoader
        f_loss    -- The loss function, i.e. a loss Module
        device    -- The device to use for computation 

    Returns :

        A tuple with the mean loss, target and prediction (for display purposes)

    """
    # We disable gradient computation which speeds up the computation
    # and reduces the memory usage
    with torch.no_grad():
        # We enter evaluation mode. This is useless for the linear model
        # but is important with layers such as dropout, batchnorm, ..
        model.eval()

        #  List used to compute the average loss on the dataset
        out = [] 

        for i, (inputs, targets) in enumerate(loader):

            # We got a minibatch from the loader within inputs and targets

            # We need to copy the data on the GPU if we use one
            inputs, targets = inputs.to(device), targets.to(device)

            # Compute the forward pass, i.e. the scores for each input image
            outputs = model(inputs.float()) #if no float: error

            # We compute the loss
            out.append(f_loss(outputs, targets))

        return np.mean(out,dtype=np.float64)

def Load_best_prediction(model, model_path, loader, f_loss, device):
    """
    Test a model by iterating over the loader

    Arguments :

        model     -- A torch.nn.Module object
        loader    -- A torch.utils.data.DataLoader
        f_loss    -- The loss function, i.e. a loss Module
        device    -- The device to use for computation 

    Returns :

        A tuple with the mean loss, the target list and  prediction list (for diplay purposes)

    """
    # We disable gradient computation which speeds up the computation
    # and reduces the memory usage
    with torch.no_grad():
        # We enter evaluation mode. This is useless for the linear model
        # but is important with layers such as dropout, batchnorm, ..

        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint)
        model.eval()

        #  List used to compute the average loss on the dataset
        out = [] 
        ground_truth  = []
        prediction = []
        
        for i, (inputs, targets) in enumerate(loader):

            # We got a minibatch from the loader within inputs and targets

            # We need to copy the data on the GPU if we use one
            inputs, targets = inputs.to(device), targets.to(device)

            # Compute the forward pass, i.e. the scores for each input image
            outputs = model(inputs.float()) #if no float: error

            # We compute the loss
            out.append(f_loss(outputs, targets))

            # For display purposes

            ground_truth += targets.tolist()
            prediction += outputs.tolist()

        return np.mean(out,dtype=np.float64), ground_truth, prediction


def generate_unique_logpath(logdir, raw_run_name):
    i = 0
    while(True):
        run_name = raw_run_name + "_" + str(i)
        log_path = os.path.join(logdir, run_name)
        if not os.path.isdir(log_path):
            return log_path
        i = i + 1

class ModelCheckpoint:

    def __init__(self, filepath, model):
        self.min_loss = None
        self.filepath = filepath
        self.model = model

    def update(self, loss):
        if (self.min_loss is None) or (loss < self.min_loss):
            print("Saving a better model")
            print("Value of the loss saved : {:.4f}".format(loss))
            torch.save(self.model.state_dict(), self.filepath)
            #torch.save(self.model, self.filepath)
            self.min_loss = loss




def get_mean_std(x_y_train):
    '''get mean and std from an inputs and targets tensor'''

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data loader
    batch_size = 16
    normalizing_loader = torch.utils.data.DataLoader(dataset=x_y_train,
                                            batch_size=batch_size,
                                            shuffle=True)

    # Example to play with

    examples = iter(normalizing_loader)
    data,target = examples.next()

    # Compute the mean over minibatches
    mean_strips = torch.zeros(data.size()[2])
    mean_target = torch.zeros(1)
    nb_strips = 0

    for strips_data, target in normalizing_loader:
        mean_strips = mean_strips + strips_data.sum(dim=(0,1)) # taille [16, 306, 36] on somme sur la dim 16 et 306
        mean_target = mean_target + target.sum(0) # taille [16]
        nb_strips = nb_strips + torch.count_nonzero(strips_data,dim = (0,1)) # pour trouver le nombre total de strips

    #print(nb_strips)
    nb_strips = 31895
    print("nb of strips for the mean: {}".format(nb_strips))
    mean_strips /= nb_strips
    mean_target /= nb_strips

    #print(mean_strips)



    std_strips = torch.zeros_like(mean_strips)
    std_target = torch.zeros_like(mean_target)

    for strips_data, target in normalizing_loader:
        std_strips = std_strips + ((strips_data-mean_strips)**2).sum(dim=(0,1))
        std_target = std_target + ((target-mean_target)**2).sum(0)

    std_strips /= nb_strips
    std_strips = torch.sqrt(std_strips)

    std_target /= nb_strips
    std_target = torch.sqrt(std_target)

    return mean_strips,mean_target, std_strips, std_target