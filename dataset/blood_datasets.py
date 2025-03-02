import os
import torch
from PIL import Image, ImageFile
from torchvision import transforms
import random
import json
ImageSize = 224
SPLIT_TRAIN_PERCENTAGE = 0.8
SPLIT_VAL_PERCENTAGE = 0.2


# 初始化下标
ClassIndex = {}
startIndex = 0
orderKeys = []


def init_with_obj(dict,augment=False):
    return BloodCellImageFolder(dict, augment=augment)


class BloodCellImageFolder(torch.utils.data.Dataset):

    def __init__(self, sample_list, augment, env=0, env_name='',transform = None):
        self.augment_mode = None
        self.augment = augment
        if transform is None:
            self.transform = self.get_transform()
        else:
            self.transform = transform
        self.sampleList = sample_list
        self.env = env
        self.env_name = env_name
        self.class_counter = {}
        self.targets = []

        for v in sample_list:
            # cl = v['class']
            cl_index = v['class_no']
            self.targets.append(cl_index)
            if cl_index not in self.class_counter:
                self.class_counter[cl_index] = 1
            else:
                self.class_counter[cl_index] += 1

        self.affect = {}
        # for i,v in enumerate(self.class_counter):
        #     self.
    def __len__(self):
        return len(self.sampleList)

    def __getitem__(self, index):
        row = self.sampleList[index]
        img = Image.open(row['path']).convert('RGB')
        img = self.transform(img)
        y = row['class_no']
        return img, y

    def get_transform(self):
        color_jitter = [0.2, 0.2, 0.2, 0.2]
        gray = 0.5
        if self.augment:
            transform = transforms.Compose([
                transforms.Resize((ImageSize, ImageSize)),
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation((-180,180)),
                transforms.ColorJitter(color_jitter[0], color_jitter[1], color_jitter[2], color_jitter[3]),
                transforms.RandomGrayscale(p=gray),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((ImageSize, ImageSize)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        return transform


