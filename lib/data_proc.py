#!/usr/bin/python
# -*- coding: utf-8 -*-
###########################################################################
# O'Reilly Generating Deeplearning the 2nd ED.
# Working Directory : ..\work_2025
# 2025 02 09 by Jinwuk Seok
###########################################################################
_description = '''\
================================================================
data_proc.py : data_proc.py for Generating Deeplearning the 2nd ED 
                    Written by Jinwuk @ 2025-02-09
================================================================
Example :  There is no Operation instruction. 
'''
g_line      = "----------------------------------------------------------------"

import numpy as np

from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
import torchvision.transforms as Transforms

import os
import time
from PIL import Image

import interface_function as IF
import my_debug as DBG

class Fashion_MNIST:
    def __init__(self, conf_data):
        self.work_path      = conf_data.data_path
        self.num_workers    = conf_data.num_workers
        self.batch_size     = conf_data.batch_size
        self.data_padding_size = conf_data.data_padding_size
        self.data_shape     = None
        self.c_conf         = conf_data
    # first define a transform function, to turn images into tersors
    def set_data_transform(self):
        _transform = Transforms.Compose([
                        Transforms.ToTensor(),
                        Transforms.Pad(self.data_padding_size)])
        return _transform

    def get_dataloaders(self):
        print("Load Fashion MNIST Data\n" + g_line)
        # Set data Transform
        transform = self.set_data_transform()
        # load MNIST dataset
        try:
            train_ds    = torchvision.datasets.FashionMNIST(root=self.work_path, train=True, download=True, transform=transform)
            test_ds     = torchvision.datasets.FashionMNIST(root=self.work_path, train=False, download=True, transform=transform)
            train_loader= DataLoader(dataset=train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
            test_loader = DataLoader(dataset=test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        except Exception as e:
            DBG.dbg("Fail of Fashion MINIST Data Download !!! Please Check Network \nError : ", e)
            exit(0)
        # Check Data Shape
        self.data_shape = next(iter(train_loader))[0].shape
        # Set data classes
        self.c_conf.set_data_label(l_label=train_ds.classes)

        return train_loader, test_loader

class CelebA(Dataset):
    def __init__(self, conf_data):
        super().__init__()
        #----------------------------------------------------
        # For structure of CelebA data directory
        # ----------------------------------------------------
        try:
            self.dir    = os.path.join(conf_data.data_path, conf_data.data_dir)
            self.imgs   = os.listdir(self.dir)
            self.length = len(self.imgs)
            self.image_size = conf_data.image_size

            self.batch_size     = conf_data.batch_size
            self.data_padding_size = conf_data.data_padding_size
            self.data_shape     = None
            self.c_conf         = conf_data

            self.set_data_transform()
        except Exception as e:
            DBG.dbg("Configuration is not compatible to the CelebA data. Please check the configuration YAML file")
            DBG.dbg(e)
            exit(0)
    def __len__(self):
        return self.length
    def __getitem__(self, i):
        output_img = self.transform(Image.open(os.path.join(self.dir, self.imgs[i])))
        return output_img
    def set_data_transform(self):
        self.transform = Transforms.Compose([
                 Transforms.Resize((self.image_size, self.image_size)),
                 Transforms.ToTensor(),
        ])

    def get_dataloaders(self):
        print("Load CelebA Data\n")
        _train_set_ratio = float(self.c_conf.train_set_ratio)
        _test_set_ratio  = 1.0 - _train_set_ratio
        train_ds, test_ds = random_split(self, [_train_set_ratio, _test_set_ratio])
        print('Train data size: {}'.format(len(train_ds)))
        print('Test data size : {}'.format(len(test_ds)))
        print(g_line)

        train_loader = DataLoader(dataset=train_ds, batch_size=self.c_conf.batch_size,
                                  shuffle=True, num_workers=self.c_conf.num_workers, pin_memory=True)
        test_loader = DataLoader(dataset=test_ds, batch_size=self.c_conf.batch_size,
                                 shuffle=False, num_workers=self.c_conf.num_workers, pin_memory=True)
        # Check Data Shape
        self.data_shape = next(iter(train_loader))[0].shape
        # Set data classes
        #self.c_conf.set_data_label(l_label=train_ds.classes)

        return train_loader, test_loader

# =================================================================
# Main Routine
# =================================================================
if __name__ == "__main__":


    print("===================================================")
    print("Process Finished ")
    print("===================================================")

