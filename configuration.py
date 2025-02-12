#!/usr/bin/python
# -*- coding: utf-8 -*-
###########################################################################
# O'Reilly Generating Deeplearning the 2nd ED.
# Working Directory : ..\work_2025
# 2025 02 09 by Jinwuk Seok
###########################################################################
_description = '''\
================================================================
data_proc.py : configuration.py for Generating Deeplearning the 2nd ED 
                    Written by Jinwuk @ 2025-02-09
================================================================
Example :  There is no Operation instruction. 
'''
g_line      = "----------------------------------------------------------------"

# ----------------------------------------------------------------
# Following Libraries are a fundamental requirements
# However, we employ only partial libraries within those
# ----------------------------------------------------------------
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as Transforms
from torchsummary import summary
from matplotlib import pyplot as plt

import os, sys
import lib.interface_function as IF
import lib.my_debug as DBG
class configuration:
    def __init__(self, L_param, _intro_msg=_description, bUseParam=False):
        self.args = IF.ArgumentParse(L_Param=L_param, _prog=__file__, _intro_msg=_intro_msg, bUseParam=bUseParam)
        # ----------------------------------------------------------------
        # Path and File
        #----------------------------------------------------------------
        self.root_path  = os.getcwd()
        self.model_path = os.path.join(self.root_path, 'model')
        self.lib_path   = os.path.join(self.root_path, 'lib')
        self.doc_path   = os.path.join(self.root_path, 'doc')
        self.data_path  = os.path.join(self.root_path, 'data')
        # ----------------------------------------------------------------
        # Fundamental Configure
        #----------------------------------------------------------------
        self.fundamental_config = IF.read_yaml(self.args.fundamental_configure_file)

        self.embedding_dim = self.fundamental_config['OP_SPEC']['EMBEDDING_DIM']
        self.epoch      = self.fundamental_config['OP_SPEC']['EPOCHS']
        self.buffer_size = self.fundamental_config['OP_SPEC']['BUFFER_SIZE']
        self.validation_split = self.fundamental_config['OP_SPEC']['VALIDATION_SPLIT']
        self.image_size = self.fundamental_config['DATASPEC']['IMAGE_SIZE']
        self.channels   = self.fundamental_config['DATASPEC']['CHANNELS']
        self.batch_size = self.fundamental_config['DATASPEC']['BATCH_SIZE']
        self.hidden_lyr = self.fundamental_config['CLASSIFIER']['HIDDEN']
        # For CUDA setting
        self.fundamental_config['DEVICE'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device     = self.fundamental_config['DEVICE']
        # For Miscellaneous
        self.SummaryWriterPATH  = self.fundamental_config['EXPERIMENT_PARAM']['SummaryWriterPATH']
        self.test_samples       = self.fundamental_config['EXPERIMENT_PARAM']['TEST_SAMPLES']
        self.model_file         = self.fundamental_config['EXPERIMENT_PARAM']['MODEL_FILE']
        # ----------------------------------------------------------------
        # Optimizer Setting
        #----------------------------------------------------------------
        s_algorithm_name    = self.fundamental_config['LEARNING_PARAMS']['OPTIMIZER']
        self.c_optimizer    = getattr(torch.optim, s_algorithm_name)
        self.learning_rate  = float(self.fundamental_config['LEARNING_PARAMS']['LEARNING_RATE'])
        self.loss_fn_param  = self.fundamental_config['LEARNING_PARAMS']['LOSS_PARAM']
        # ----------------------------------------------------------------
        # Operation Mode Setting
        #----------------------------------------------------------------
        if os.path.exists(self.model_file):
            self.loaded_model = torch.load(self.model_file)
        else:
            _op_msg = "Operation of inference mode is impossible. \nThere is not any saved model file"
            self.args.inference_mode = False
        _op_msg = "Inference mode" if self.args.inference_mode else "Normal learning mode"
        print(_op_msg + "\n" + g_line)
        # ----------------------------------------------------------------
        # Miscellaneous Setting
        #----------------------------------------------------------------
        self.label_path         = os.path.join(self.root_path, self.args.label_file)
        self.data_label         = IF.read_yaml(self.label_path)
        self.data_padding_size  = self.args.data_padding_size
        self.num_workers        = self.args.number_of_workers
        self.quite_mode         = self.args.quite_mode
        self.save_graphic       = self.args.save_graphic
    def __call__(self, model, **kwargs):
        # ----------------------------------------------------------------
        # Configure __call__ 함수는 Loss function과 Optimizer Setting에 사용
        #----------------------------------------------------------------
        _model_name = model.__class__.__name__
        try:
            if _model_name == 'AutoEncoder':
                cf_loss_fn    = nn.BCEWithLogitsLoss(reduction=self.loss_fn_param)
                cf_optimizer  = self.c_optimizer(model.parameters(), lr=self.learning_rate)
            elif _model_name == 'Classifier_for_autoencoder':
                cf_loss_fn    = nn.CrossEntropyLoss()
                cf_optimizer  = self.c_optimizer(model.parameters(), lr=self.learning_rate)
            else:
                DBG.dbg("Model has not been specified. It is error")
                exit(0)
        except Exception as e:
            DBG.dbg("Error : %s \nProgram Terminated !!" %e)
            exit(0)
        finally:
            print(f"Model Name    : %s" %_model_name)
            print(f"Optimizer     : %s.%s" %(self.c_optimizer.__module__, self.c_optimizer.__ne__))
            print(f" learning_rate: %f" %self.learning_rate)
            print(f"Loss Function : %s.%s" %(cf_loss_fn.__module__, cf_loss_fn.__ne__))
            print(f"    parameter : %s" %self.loss_fn_param)
            print(f"Device for OP : %s" %self.device)
            print(g_line)

        return cf_loss_fn, cf_optimizer

# =================================================================
# Test Routine
# =================================================================
if __name__ == "__main__":
    c_conf  = configuration(L_param=[])

    print("===================================================")
    print("Process Finished ")
    print("===================================================")


