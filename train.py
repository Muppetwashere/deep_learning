import torch
import os
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import models
import data
import utils

if __name__ == '__main__':
    #chose the model
    model = models.LinearNet(1*28*28, 10)

    #GPU settings
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # load the dataset
    num_threads = 4 # Loading the dataset is using 4 CPU threads
    valid_ratio = 0.2 # Going to use 80%/20% split for train/valid
    batch_size = 128 # Using minibatches of 128 samples

    train_loader, valid_loader, test_loader = data.load_fashion_mnist(valid_ratio, batch_size, num_threads)

    # loss, optim
    model.to(device)
    loss = nn.CrossEntropyLoss()  # This computes softmax internally
    optimizer = torch.optim.Adam(model.parameters())


    # Main Loop
    epochs = 10

    for t in range(epochs):
        print("Epoch {}".format(t))
        train_loss, train_acc = utils.train(model, train_loader, loss, optimizer, device)

        val_loss, val_acc = utils.test(model, valid_loader, loss, device)
        print(" Validation : Loss : {:.4f}, Acc : {:.4f}".format(val_loss, val_acc))

        test_loss, test_acc = utils.test(model, test_loader, loss, device)
        print(" Test       : Loss : {:.4f}, Acc : {:.4f}".format(test_loss, test_acc))
