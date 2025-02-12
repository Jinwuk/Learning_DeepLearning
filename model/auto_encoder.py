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

from torch import nn
from torch.nn import functional as F
from torchsummary import summary
import lib.my_debug as DBG
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
            nn.Conv2d(in_channels=_data_ch, out_channels=32,    kernel_size=_kernel_size, stride=_stride, padding=_padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=32,       out_channels=64,    kernel_size=_kernel_size, stride=_stride, padding=_padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=64,       out_channels=128,   kernel_size=_kernel_size, stride=_stride, padding=_padding),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=_features, out_features=self.latents)
        )
    def forward(self, x):
        return self.model(x)

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
            nn.ConvTranspose2d(in_channels=64,  out_channels=32,  kernel_size=_kernel_size, stride=_stride, padding=_padding, output_padding=_padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=_data_ch, kernel_size=_kernel_size, stride=1, padding=_padding)
        )
        '''
        1. Padding 과 Outpadding의 Size가 같은 이유는 Kernel Size 때문이다. Decoder에서는 영상 Size가 4x4에서 32x32로 확대된다 (channel은 줄어든다). 
        따라서, 원래 3x3 Kernel 때문에 필요한 padding size=1 외에, 다음 차례의 padding에서 필요한 output_padding도 3x3 Kernel 때문에 1이 필요하다.
        2. 맨 마지막 nn.Conv2d 는 출력 Dimension이 image와 같아야 하므로 stride=1 이 된다.  
        '''
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
# Main Classes : AutoEncoder
# ----------------------------------------------------------------
class AutoEncoder(nn.Module):
    def __init__(self, c_config):
        super().__init__()
        self.encoder = Encoder(c_config= c_config)
        self.decoder = Decoder(c_config= c_config)

    def forward(self, x):
        _encoded = self.encoder(x)
        _decoded = self.decoder(_encoded)
        return _decoded, _encoded

    def generate(self, z):
        return F.sigmoid(self.decoder(z))

    # Service function
    def print_summary(self, _shape, _quite=True):
        if _quite == False:
            summary(self, _shape)
        else: pass
# ----------------------------------------------------------------
# Sub Classes : Full Connected Network
# ----------------------------------------------------------------



# =================================================================
# Main Routine
# =================================================================
import interface_function as IF

if __name__ == "__main__":
    c_conf = None
    c_encoder   = Encoder(c_config=c_conf).to('cpu')
    summary(c_encoder, (1, 32, 32))

    print("===================================================")
    print("Process Finished ")
    print("===================================================")