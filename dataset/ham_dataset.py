import os
import os.path
import logging
import numpy as np
import torch.utils.data as data
from PIL import Image
from torchvision.datasets import CIFAR10, CIFAR100, FashionMNIST, ImageFolder, DatasetFolder
import torch
import torchvision.transforms as transforms



def init_with_obj(dict):
    transform_test = transforms.Compose([
        transforms.Resize((56, 56)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # transforms.Resize((ImageSize, ImageSize)),
        # transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_folder = ImageFolder_HAM10000(dict, transform=transform_test)
    return test_folder


def init_ham_train(client_info):
    train_data = client_info["train_set"]
    test_data = client_info["test_set"]
    ImageSize = 224
    use_mode = {
        "ColorJitter": [0.2, 0.2, 0.2, 0.2],
        "Gray": 0.5,
    }
    # if self.
    colorJitter = use_mode["ColorJitter"]
    gray = use_mode["Gray"]

    transform_train = transforms.Compose([
        transforms.Resize((56, 56)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # transforms.Resize((ImageSize, ImageSize)),
        # transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation((-180, 180)),
        # transforms.ColorJitter(colorJitter[0], colorJitter[1], colorJitter[2], colorJitter[3]),
        # transforms.RandomGrayscale(p=gray),
        # transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

    ])
    transform_test = transforms.Compose([
        transforms.Resize((56, 56)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # transforms.Resize((ImageSize, ImageSize)),
        # transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_folder = ImageFolder_HAM10000(train_data,transform=transform_train)
    test_folder = ImageFolder_HAM10000(test_data,transform=transform_test)
    return train_folder, test_folder

class ImageFolder_HAM10000(torch.utils.data.Dataset):
    def __init__(self, datalist, transform=None):
        self.sampleList = datalist
        self.transform = transform
        self.class_counter = {}
        self.targets = []

        for v in self.sampleList:
            # cl = v['class']
            cl_index = v['class_no']
            self.targets.append(cl_index)
            if cl_index not in self.class_counter:
                self.class_counter[cl_index] = 1
            else:
                self.class_counter[cl_index] += 1



    def __getitem__(self, index):
        row = self.sampleList[index]
        img = Image.open(row['path']).convert('RGB')
        img = self.transform(img)
        y = row['class_no']
        return img, y


    def __len__(self):
        return len(self.sampleList)