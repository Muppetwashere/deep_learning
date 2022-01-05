import os
import torchvision
import torch
import torchvision.transforms as transforms


def load_fashion_mnist(valid_ratio, batch_size, num_threads):

    dataset_dir = os.path.join(os.path.expanduser("~"), 'Datasets', 'FashionMNIST')

    # Load the dataset for the training/validation sets
    train_valid_dataset = torchvision.datasets.FashionMNIST(root=dataset_dir,
                                            train=True,
                                            transform= None, #transforms.ToTensor(),
                                            download=True)

    # Split it into training and validation sets
    nb_train = int((1.0 - valid_ratio) * len(train_valid_dataset))
    nb_valid =  int(valid_ratio * len(train_valid_dataset))
    train_dataset, valid_dataset = torch.utils.data.dataset.random_split(train_valid_dataset, [nb_train, nb_valid])


    # Load the test set
    test_dataset = torchvision.datasets.FashionMNIST(root=dataset_dir,
                                                    transform= None, #transforms.ToTensor(),
                                                    train=False)

    class DatasetTransformer(torch.utils.data.Dataset):

        def __init__(self, base_dataset, transform):
            self.base_dataset = base_dataset
            self.transform = transform

        def __getitem__(self, index):
            img, target = self.base_dataset[index]
            return self.transform(img), target

        def __len__(self):
            return len(self.base_dataset)


    train_dataset = DatasetTransformer(train_dataset, transforms.ToTensor())
    valid_dataset = DatasetTransformer(valid_dataset, transforms.ToTensor())
    test_dataset  = DatasetTransformer(test_dataset , transforms.ToTensor())

    ######## Dataloaders

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True,                # <-- this reshuffles the data at every epoch
                                            num_workers=num_threads)

    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                            batch_size=batch_size, 
                                            shuffle=False,
                                            num_workers=num_threads)


    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=num_threads)


    return train_loader, valid_loader, test_loader


if __name__ == '__main__':
    num_threads = 4 # Loading the dataset is using 4 CPU threads
    valid_ratio = 0.2 # Going to use 80%/20% split for train/valid
    batch_size = 128 # Using minibatches of 128 samples

    train_loader, valid_loader, test_loader = load_fashion_mnist(valid_ratio, batch_size, num_threads)

    print("The train set contains {} images, in {} batches".format(len(train_loader.dataset), len(train_loader)))
    print("The validation set contains {} images, in {} batches".format(len(valid_loader.dataset), len(valid_loader)))
    print("The test set contains {} images, in {} batches".format(len(test_loader.dataset), len(test_loader)))