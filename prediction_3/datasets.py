import numpy as np
import glob
from PIL import Image
from collections import Counter

import torch
import torchvision
from torchvision import transforms, io
from torch.utils.data import Dataset
import os

class PlanktonDataset(Dataset):
    """Plankton Dataset"""
    def __init__(self, root_path, final_size=64):
        self.root_path = root_path
        self.images, self.labels, self.classes = self.load_frames()
        self.final_size = final_size
        self.class_count = Counter(self.labels)

    def load_frames(self):
        images, labels, classes = [], [], []
        subfolders = glob.glob(self.root_path + '/*')
        subfolders.sort()
        for subfolder in subfolders:
            classes.append(subfolder.split('/')[-1])
            for file in glob.glob(subfolder + '/*.jpg'):
                images.append(file)
                labels.append(int(file.split('/')[-2].split('_')[0]))
        return images, labels, classes

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        label = torch.tensor(self.labels[idx])
        image=transforms.Compose([
            transforms.Resize((self.final_size,self.final_size),interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])(image)
        return image, label


class PlanktonTestDataset(Dataset):
    def __init__(self,test_path, final_size=64):
        self.test_images_path =  glob.glob(os.path.join(test_path, '*.jpg'))
        self.test_images_path.sort()
        self.final_size = final_size

    def __len__(self):
        return len(self.test_images_path)

    def __getitem__(self,index):
        im_name = self.test_images_path[index].split('/')[-1]
        image = Image.open(self.test_images_path[index]).convert('RGB')
        image=transforms.Compose([
                transforms.Resize((self.final_size,self.final_size),interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])(image)
        return image,im_name