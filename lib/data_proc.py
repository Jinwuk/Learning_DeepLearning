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

from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as Transforms
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

# =================================================================
# Main Routine
# =================================================================
if __name__ == "__main__":


    print("===================================================")
    print("Process Finished ")
    print("===================================================")

