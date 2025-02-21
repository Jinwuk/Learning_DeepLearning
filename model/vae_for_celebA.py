#!/usr/bin/python
# -*- coding: utf-8 -*-
###########################################################################
# O'Reilly Generating Deeplearning the 2nd ED.
# Working Directory : ..\work_2025
# 2025 02 20 by Jinwuk Seok
###########################################################################
_description = '''\
================================================================
vae_for_celebA : VAE for CelebA in Generating Deeplearning the 2nd ED 
                    Written by Jinwuk @ 2025-02-20
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
# summary(encoder, (c_conf.channels, c_conf.image_size, c_conf.image_size)))  ## (3, 32, 32)
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
        _num_blocks  = c_config.fundamental_config['NETWORK_PARAMS']['BLOCKS']

        if (_kernel_size%2) == 1:
            _padding     = int(_kernel_size/2)
        else:
            DBG.dbg("We strongly recommend an odd size of kernel")
            DBG.dbg("This kernel : %d" %_kernel_size)
            DBG.dbg("Process Terminated!!")
            exit(0)
        # To calculate the z_dimension which is the final dimension before mean and logvar : 64x(2x2) _z_dim=2
        _z_dim      = c_config.image_size/pow(2, _num_blocks)
        #----------------------------------------------------------------
        # Model Description
        #----------------------------------------------------------------
        self.latents = c_config.embedding_dim
        self.z_dimension = int(_features * (_z_dim * _z_dim))
        self.conv_module = nn.ModuleList()

        # Adding convolutional blocks to the module list
        for i in range(_num_blocks):
            in_chan = _data_ch if i == 0 else _features
            out_chan = _features
            conv_block = nn.Sequential(
                nn.Conv2d(in_channels=in_chan,
                          out_channels=out_chan,
                          kernel_size=_kernel_size,
                          stride=_stride,
                          padding=_padding),
                nn.BatchNorm2d(num_features=_features),
                nn.LeakyReLU()
            )
            self.conv_module.append(conv_block)

        # Mapping features to mean and logvar
        # in dimension : 64ch x (2x2) = 256, out dimension = 200, latents ir embedding_dim
        self.mean = nn.Linear(in_features=self.z_dimension, out_features=self.latents)
        self.logvar = nn.Linear(in_features=self.z_dimension, out_features=self.latents)

    def forward(self, x):
        for module in self.conv_module:
            x = module(x)
        #----------------------------------------------------------------
        # x의 Dimension은 batch size 제외하고 64x(2x2),
        # reshape에서 -1 이면 flat하게 펼치는 것이므로 256이 된다.
        # ----------------------------------------------------------------
        x = x.reshape(x.shape[0], -1)
        # ----------------------------------------------------------------
        # 256x1 vector가 mean, logvar에 들어가서 200 dim latents vector가 된다
        # ----------------------------------------------------------------
        mean_x   = self.mean(x)
        logvar_x = self.logvar(x)
        return mean_x, logvar_x


# encoder = Encoder(latents=EMBEDDING_DIM).to(DEVICE)
# summary(encoder, (3, 64, 64))
# ----------------------------------------------------------------
# Main Classes : Decoder
# ----------------------------------------------------------------
# ===============================================================
# c_decoder = Decoder(c_config=c_conf).to(c_conf.device)
# summary(c_decoder, (c_conf.embedding_dim, ))              # 쉼표가 중요. Decoder의 Dimension 문제 떄문
# ===============================================================
# # Decoder
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
        _num_blocks  = c_config.fundamental_config['NETWORK_PARAMS']['BLOCKS']

        if (_kernel_size%2) == 1:
            _padding     = int(_kernel_size/2)
        else:
            DBG.dbg("We strongly recommend an odd size of kernel")
            DBG.dbg("This kernel : %d" %_kernel_size)
            DBG.dbg("Process Terminated!!")
            exit(0)

        # To calculate the z_dimension which is the final dimension before mean and logvar : 64x(2x2) _z_dim=2
        _z_dim = c_config.image_size / pow(2, _num_blocks)
        #----------------------------------------------------------------
        # Model Description
        #----------------------------------------------------------------
        self.latents = c_config.embedding_dim
        self.z_dimension = int(_features * (_z_dim * _z_dim))
        self.features= _features

        self.fc = nn.Sequential(
            nn.Linear(in_features=self.latents, out_features=self.z_dimension),
            nn.BatchNorm1d(num_features=self.z_dimension),
            nn.LeakyReLU()
        )

        self.trans_conv_module = nn.ModuleList()
        for _ in range(_num_blocks):
            trans_conv = nn.Sequential(
                nn.ConvTranspose2d(in_channels=_features, out_channels=_features,
                                   kernel_size=_kernel_size, stride=_stride, padding=_padding, output_padding=_padding),
                nn.BatchNorm2d(num_features=_features),
                nn.LeakyReLU()
            )
            self.trans_conv_module.append(trans_conv)

        self.output = nn.Sequential(
            nn.Conv2d(in_channels=_features,
                      out_channels=_data_ch,
                      kernel_size=_kernel_size,
                      padding=_padding),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)                                  # 200 -> 256 dimension : Latents -> compatible dimensions
        x = x.reshape(x.shape[0], self.features, 2, 2)  # 256 -> 64(features) x (2x2)
        for module in self.trans_conv_module:
            x = module(x)
        return self.output(x)

# decoder = Decoder(latents=EMBEDDING_DIM).to(DEVICE)
# summary(decoder, (EMBEDDING_DIM,))

# ----------------------------------------------------------------
# Main Classes : Simple VAE
# ----------------------------------------------------------------
# Variational AutoEncoder
class VAE_4_CELEBA(nn.Module):
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

#vae = VAE(EMBEDDING_DIM).to(DEVICE)
#summary(vae, (3, 64, 64))

