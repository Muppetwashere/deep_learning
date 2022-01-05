import torch
import torchvision                                                       
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import data
import models
import utils

import os.path
import os
import torch.nn as nn

if __name__ == '__main__':
    print("tensorboard --logdir ./logs for monitoring")
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

    # model
    model = models.LinearNet(1*28*28, 10)

    # GPU settings
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    #loss, optim
    model.to(device)
    
    f_loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

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
    # Define the callback object
    model_checkpoint = utils.ModelCheckpoint(logdir + "/best_model.pt", model)

    # Main loop
    epochs = 10

    for t in range(epochs):
        print("Epoch {}".format(t))
        utils.train(model, train_loader, f_loss, optimizer, device)
        val_loss, val_acc = utils.test(model, valid_loader, f_loss, device)
        model_checkpoint.update(val_loss)
        print(" Validation : Loss : {:.4f}, Acc : {:.4f}".format(val_loss, val_acc))

        train_loss, train_acc = utils.test(model, train_loader, f_loss, device)
        
        test_loss, test_acc = utils.test(model, test_loader, f_loss, device)
        print(" Test       : Loss : {:.4f}, Acc : {:.4f}".format(test_loss, test_acc))

        tensorboard_writer.add_scalar('metrics/train_loss', train_loss, t)
        tensorboard_writer.add_scalar('metrics/train_acc',  train_acc, t)
        tensorboard_writer.add_scalar('metrics/val_loss', val_loss, t)
        tensorboard_writer.add_scalar('metrics/val_acc',  val_acc, t)

