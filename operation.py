#!/usr/bin/python
# -*- coding: utf-8 -*-
###########################################################################
# O'Reilly Generating Deeplearning the 2nd ED.
# Working Directory : ..\work_2025
# 2025 02 09 by Jinwuk Seok
###########################################################################
_description = '''\
====================================================
operation.py : training and validating for Generating Deeplearning the 2nd ED 
                    Written by Jinwuk @ 2025-02-11
====================================================
Example :  There is no Operation instruction. 
'''
g_line      = "----------------------------------------------------"

import torch
from torch.utils.tensorboard import SummaryWriter

class operation_fn:
    def __init__(self,conf_data):
        self.c_config   = conf_data
        self.writer     = SummaryWriter(self.c_config.SummaryWriterPATH)
    # ----------------------------------------------------
    # A single epoch train funcion
    # ----------------------------------------------------
    def train(self, model, dataloader, optimizer, loss_fn):
        # ----------------------------------------------------
        # Train Setting
        # ----------------------------------------------------
        DEVICE      =self.c_config.device
        #optimizer   =self.c_config.c_optimizer
        #loss_fn     =self.c_config.loss_fn
        # ----------------------------------------------------
        # Main routine
        # ----------------------------------------------------
        model.train()
        train_loss = 0
    
        for i, (train_x, train_y) in enumerate(dataloader):
            optimizer.zero_grad()
            train_x     = train_x.to(DEVICE)
            recon_x,    = model(train_x)
            loss = loss_fn(recon_x, train_x)
    
            loss.backward()
            optimizer.step()
    
            train_loss += loss.item()
    
        return train_loss / len(dataloader)

    #----------------------------------------------------
    # Validation function
    #----------------------------------------------------
    def validate(self, model, dataloader, loss_fn):
        # ----------------------------------------------------
        # Train Setting
        # ----------------------------------------------------
        DEVICE      =self.c_config.device
        #loss_fn     =self.c_config.loss_fn
        # ----------------------------------------------------
        # Main routine
        # ----------------------------------------------------
        model.eval()
        test_loss = 0
        for i, (test_x, test_y) in enumerate(dataloader):
            test_x = test_x.to(DEVICE)
            with torch.no_grad():
                recon_x = model(test_x)
                loss = loss_fn(recon_x, test_x)
    
            test_loss += loss
        return test_loss / len(dataloader)
    
    #----------------------------------------------------
    # Record and print the result to each epoch
    #----------------------------------------------------
    def record_result(self, _epoch, train_loss, test_loss):
        s_train_loss = "Train/loss"
        s_valid_loss = "Valid/loss"
        self.writer.add_scalar(s_train_loss, train_loss, _epoch)
        self.writer.add_scalar(s_valid_loss, test_loss, _epoch)

        print(f'Epoch {_epoch + 1: 3d}  ', s_train_loss, f"{train_loss:.4f}   ", s_valid_loss, f"{test_loss:.4f}")

