#!/usr/bin/python
# -*- coding: utf-8 -*-
###########################################################################
# O'Reilly Generating Deeplearning the 2nd ED.
# Working Directory : ..\work_2025
# 2025 02 09 by Jinwuk Seok
###########################################################################
_description = '''\
====================================================
data_proc.py : configuration.py for Generating Deeplearning the 2nd ED 
                    Written by Jinwuk @ 2025-02-09
====================================================
Example :  There is no Operation instruction. 
'''

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as Transforms
from torchsummary import summary
from matplotlib import pyplot as plt

import os
import interface_function as IF
class configuration:
    def __init__(self, L_param, _intro_msg=_description, bUseParam=False):
        self.args = IF.ArgumentParse(L_Param=L_param, _prog=__file__, _intro_msg=_intro_msg, bUseParam=bUseParam)
        # ----------------------------------------------------------------
        # Path and File
        #----------------------------------------------------------------
        self.work_path  = os.getcwd()
        #self.work_fullpath = os.path.join(self.work_path, self.work_file)
        # ----------------------------------------------------------------
        # Fundamental Configure
        #----------------------------------------------------------------
        self.fundamental_config = IF.read_yaml(self.args.fundamental_configure_file)
        self.image_size = self.fundamental_config['IMAGE_SIZE']
        self.channels   = self.fundamental_config['CHANNELS']
        self.batch_size = self.fundamental_config['BATCH_SIZE']
        self.buffer_size= self.fundamental_config['BUFFER_SIZE']
        self.validation_split   = self.fundamental_config['VALIDATION_SPLIT']
        self.embedding_dim      = self.fundamental_config['EMBEDDING_DIM']
        self.epoch      = self.fundamental_config['EPOCHS']

        # For CUDA setting
        self.fundamental_config['DEVICE'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device     = self.fundamental_config['DEVICE']
        # ----------------------------------------------------------------
        # Miscellaneous Setting
        #----------------------------------------------------------------
        self.data_padding_size  = self.args.data_padding_size
        self.num_workers        = self.args.number_of_workers
        self.quite_mode         = self.args.quite_mode

        # ----------------------------------------------------------------
        # Data Setting
        #----------------------------------------------------------------
        #self.transform  = self.set_data_transform()

        # ----------------------------------------------------------------
        # Optimizer Setting
        #----------------------------------------------------------------
        self.learning_rate = self.fundamental_config['LEARNING_RATE']

# =================================================================
# Test Routine
# =================================================================
if __name__ == "__main__":
    c_conf  = configuration(L_param=[])

    print("===================================================")
    print("Process Finished ")
    print("===================================================")


