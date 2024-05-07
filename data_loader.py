import cv2
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import os
from torchvision import transforms


class train_data(data.Dataset):

    def __init__(self,train_path,label_path):
        self.train_path = train_path
        self.label_path = label_path
        self.data = []
        file_list = os.listdir(self.label_path)
        for i in file_list:
            label_name = os.path.join(self.label_path,i)
            data_name = os.path.join(self.train_path,i)
            self.data.append([data_name, label_name])

        self.trans = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):

        data_path, label_path = self.data[index]

        data = Image.open(data_path)
        label = Image.open(label_path)

        data = self.trans(data)
        label = self.trans(label)


        return data,label

    def __len__(self):
        return len(os.listdir(self.train_path))




class test_data(data.Dataset):

    def __init__(self, test_path, label_path):
        self.test_path = test_path
        self.label_path = label_path
        self.data = []
        test_file_list = os.listdir(self.test_path)
        label_path_list = os.listdir(self.label_path)
        for i in range(len(label_path_list)):
            data_name = os.path.join(self.test_path, test_file_list[i])
            label_name = os.path.join(self.label_path, label_path_list[i])
            self.data.append([data_name, label_name])

        self.trans = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        data_path, label_path = self.data[index]

        img = Image.open(data_path)
        data = Image.open(data_path)
        label = Image.open(label_path)

        img = self.trans(img)
        data = self.trans(data)
        label = self.trans(label)
        name = data_path

        return img,data,label,name

    def __len__(self):
        return len(os.listdir(self.test_path))