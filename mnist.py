import torch
import torchvision                                                       
import torchvision.transforms as transforms
import data
import models
import utils

import os.path
import os
import torch.nn as nn

if __name__ == '__main__':

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

    # Saving the model everytime a script is executed 
    top_logdir = "./logs"
    logdir = utils.generate_unique_logpath(top_logdir, "linear")

    if not os.path.exists(top_logdir):
        os.mkdir(top_logdir)

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

