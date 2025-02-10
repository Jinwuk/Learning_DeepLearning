#!/usr/bin/python
# -*- coding: utf-8 -*-
###########################################################################
# O'Reilly Generating Deeplearning the 2nd ED.
# Working Directory : ..\work_2025
# 2025 02 09 by Jinwuk Seok
###########################################################################
_description = '''\
====================================================
main_test.py : main_test.py for Generating Deeplearning the 2nd ED 
                    Written by Jinwuk @ 2025-02-09
====================================================
Example :  There is no Operation instruction. 
'''
from configuration import configuration
from data_proc import Fashion_MNIST
from model.auto_encoder import Encoder
from model.auto_encoder import Decoder
from model.auto_encoder import AutoEncoder
from torchsummary import summary

# =================================================================
# Main Routine
# =================================================================
if __name__ == "__main__":
    L_param=[]
    c_conf = configuration(L_param=L_param, _intro_msg=_description)
    c_data = Fashion_MNIST(conf_data=c_conf)
    # ----------------------------------------------------------------
    # 2. Network Setting
    # ----------------------------------------------------------------
    #c_encoder = Encoder(c_config=c_conf).to(c_conf.device)
    #summary(c_encoder, (c_conf.channels, c_conf.image_size, c_conf.image_size))
    #c_decoder = Decoder(c_config=c_conf).to(c_conf.device)
    #summary(c_decoder, (c_conf.embedding_dim, ))

    c_ae = AutoEncoder(c_config=c_conf).to(c_conf.device)
    summary(c_ae, (c_conf.channels, c_conf.image_size, c_conf.image_size))
    '''
    # ----------------------------------------------------------------
    # 1. Data setting
    # ----------------------------------------------------------------
    _data_transform = c_data.set_data_transform()
    train_loader, test_loader = c_data.get_dataloaders(_data_transform)
    # Check Data Shape
    print(next(iter(train_loader))[0].shape)
    '''



    print("===================================================")
    print("Process Finished ")
    print("===================================================")