#!/usr/bin/python
# -*- coding: utf-8 -*-
###########################################################################
# O'Reilly Generating Deeplearning the 2nd ED.
# Working Directory : ..\work_2025
# 2025 02 09 by Jinwuk Seok
###########################################################################
_description = '''\
================================================================
auto_encoder : VAE for Generating Deeplearning the 2nd ED 
                    Written by Jinwuk @ 2025-02-09
================================================================
Example :  There is no Operation instruction. 
'''
g_line      = "----------------------------------------------------------------"

import torch
from torch import nn
from torch.nn import functional as F
from torchsummary import summary
import lib.my_debug as DBG

# ----------------------------------------------------------------
# Service Function
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# Main Classes : Encoder
# ----------------------------------------------------------------
# ===============================================================
# encoder = Encoder(c_config=c_conf).to(DEVICE)
# summary(encoder, (c_conf.channels, c_conf.image_size, c_conf.image_size)))  ## (1, 32, 32)
# ===============================================================
class Encoder(nn.Module):
    def __init__(self, c_config):
        super().__init__()
        #----------------------------------------------------------------
        # Fundamental Spec
        #----------------------------------------------------------------
        _data_ch     = c_config.fundamental_config['DATASPEC']['CHANNELS']
        _kernel_size = c_config.fundamental_config['NETWORK_PARAMS']['KERNEL']
        _stride      = c_config.fundamental_config['NETWORK_PARAMS']['STRIDE']
        _features    = c_config.fundamental_config['NETWORK_PARAMS']['FEATURES']

        if (_kernel_size%2) == 1:
            _padding     = int(_kernel_size/2)
        else:
            DBG.dbg("We strongly recommend an odd size of kernel")
            DBG.dbg("This kernel : %d" %_kernel_size)
            DBG.dbg("Process Terminated!!")
            exit(0)
        #----------------------------------------------------------------
        # Model Description
        #----------------------------------------------------------------
        self.latents = c_config.embedding_dim

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=_data_ch, out_channels=32, kernel_size=_kernel_size, stride=_stride, padding=_padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,       out_channels=64, kernel_size=_kernel_size, stride=_stride, padding=_padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,       out_channels=128, kernel_size=_kernel_size, stride=_stride, padding=_padding),
            nn.ReLU(),
            nn.Flatten(),
            # nn.Linear(in_features=2048, out_features=self.latents)
        )

        self.mean   = nn.Linear(in_features=_features, out_features=self.latents)
        self.logvar = nn.Linear(in_features=_features, out_features=self.latents)

    def forward(self, x):
        x           = self.model(x)
        mean_x      = self.mean(x)
        logvar_x    = self.logvar(x)
        return mean_x, logvar_x

    # Service function
    def print_summary(self, _shape, _quite=True):
        if _quite == False:
            summary(self, _shape)
        else: pass

# ----------------------------------------------------------------
# Main Classes : Decoder
# ----------------------------------------------------------------
# ===============================================================
# c_decoder = Decoder(c_config=c_conf).to(c_conf.device)
# summary(c_decoder, (c_conf.embedding_dim, ))              # 쉼표가 중요. Decoder의 Dimension 문제 떄문
# ===============================================================
class Decoder(nn.Module):
    def __init__(self, c_config):
        super().__init__()
        #----------------------------------------------------------------
        # Fundamental Spec
        #----------------------------------------------------------------
        _data_ch     = c_config.fundamental_config['DATASPEC']['CHANNELS']
        _kernel_size = c_config.fundamental_config['NETWORK_PARAMS']['KERNEL']
        _stride      = c_config.fundamental_config['NETWORK_PARAMS']['STRIDE']
        _features    = c_config.fundamental_config['NETWORK_PARAMS']['FEATURES']
        if (_kernel_size%2) == 1:
            _padding     = int(_kernel_size/2)
        else:
            DBG.dbg("We strongly recommend an odd size of kernel")
            DBG.dbg("This kernel : %d" %_kernel_size)
            DBG.dbg("Process Terminated!!")
            exit(0)
        #----------------------------------------------------------------
        # Model Description
        #----------------------------------------------------------------
        self.latents = c_config.embedding_dim
        self.fc = nn.Linear(self.latents, _features)

        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=_kernel_size, stride=_stride, padding=_padding, output_padding=_padding),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=64,  kernel_size=_kernel_size, stride=_stride, padding=_padding, output_padding=_padding),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32,   kernel_size=_kernel_size, stride=_stride, padding=_padding, output_padding=_padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=_data_ch, kernel_size=_kernel_size, stride=1, padding=_padding)
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.reshape(x.shape[0], 128, 4, 4)
        x = self.model(x)
        return x

    # Service function
    def print_summary(self, _shape, _quite=True):
        if _quite == False:
            summary(self, _shape)
        else: pass
# ----------------------------------------------------------------
# Main Classes : Simple VAE
# ----------------------------------------------------------------
class VAE(nn.Module):
    def __init__(self, c_config):
        super().__init__()
        self.c_config= c_config
        self.encoder = Encoder(c_config=c_config)
        self.decoder = Decoder(c_config=c_config)

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.reparameterize(mean, logvar)
        recon_x = self.decoder(z)
        return recon_x, mean, logvar

    def generate(self, z):
        return F.sigmoid(self.decoder(z))

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar).to(self.c_config.device)
        eps = torch.randn(size=logvar.shape).to(self.c_config.device)
        return mean + std * eps

    # Service function
    def print_summary(self, _shape, _quite=True):
        if _quite == False:
            summary(self, _shape)
        else:
            pass
