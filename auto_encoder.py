#!/usr/bin/python
# -*- coding: utf-8 -*-
###########################################################################
# O'Reilly Generating Deeplearning the 2nd ED.
# Working Directory : ..\work_2025
# 2025 02 09 by Jinwuk Seok
###########################################################################
_description = '''\
====================================================
auto_encoder : VAE for Generating Deeplearning the 2nd ED 
                    Written by Jinwuk @ 2025-02-09
====================================================
Example :  There is no Operation instruction. 
'''
g_line      = "----------------------------------------------------"

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as Transforms
from torchsummary import summary
from matplotlib import pyplot as plt
import my_debug as DBG
# Encoder

# ----------------------------------------------------------------
# Service Function
# ----------------------------------------------------------------
def create_multiplier(_param=2):
    value = 1
    def multiplier(_param):
        nonlocal value
        value *= _param
        return value
    return multiplier

# ----------------------------------------------------------------
# Main Classes
# ----------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, latents, config_params):
        super().__init__()
        #----------------------------------------------------------------
        # Fundamental Spec
        #----------------------------------------------------------------
        self.data_ch = config_params['CHANNELS']
        self.fund_ch = config_params['IMAGE_SIZE']
        self.kernel_size = config_params['KERNEL']
        self.stride      = config_params['STRIDE']
        self.features    = config_params['FEATURES']
        self.num_layers  = config_params['LAYERS']
        if (self.kernel_size%2) == 1:
            self.padding     = int(self.kernel_size/2)
        else:
            DBG.dbg("We strongly recommend an odd size of kernel")
            DBG.dbg("This kernel : %d" %self.kernel_size)
            DBG.dbg("Process Terminated!!")
            exit(0)
        # ----------------------------------------------------------------
        # Internel param
        # ----------------------------------------------------------------
        _param  = create_multiplier()
        self.l_channels = []
        self.l_channels.append(self.fund_ch)
        for _k in range(self.num_layers-1):
            self.l_channels.append(self.fund_ch * _param(self.stride))
        #----------------------------------------------------------------
        # Model Description
        #----------------------------------------------------------------
        self.latents = latents
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=self.data_ch, out_channels=self.l_channels[0],
                      kernel_size=self.kernel_size, stride=self.stride, padding=self.padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.l_channels[0], out_channels=self.l_channels[1],
                      kernel_size=self.kernel_size, stride=self.stride, padding=self.padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.l_channels[1], out_channels=self.l_channels[2],
                      kernel_size=self.kernel_size, stride=self.stride, padding=self.padding),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=self.features, out_features=self.latents)
        )


    def forward(self, x):
        return self.model(x)


# encoder = Encoder(EMBEDDING_DIM).to(DEVICE)
# summary(encoder, (1, 32, 32))

# Decoder
class Decoder(nn.Module):
    def __init__(self, latents):
        super().__init__()
        self.latents = latents
        self.fc = nn.Linear(self.latents, 2048)

        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.reshape(x.shape[0], 128, 4, 4)
        x = self.model(x)
        return x

# decoder = Decoder(EMBEDDING_DIM).to(DEVICE)

# =================================================================
# Main Routine
# =================================================================
if __name__ == "__main__":
    c_encoder = Encoder(latents=2).to('cpu')
    summary(c_encoder, (1, 32, 32))

    print("===================================================")
    print("Process Finished ")
    print("===================================================")