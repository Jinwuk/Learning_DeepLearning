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

import torch
from torch import nn
from datetime import datetime
import os, sys, textwrap
import lib.interface_function as IF
import lib.my_debug as DBG
class configuration:
    def __init__(self, L_param, _intro_msg=_description, bUseParam=False):
        self.l_proc_msg = []
        self.args       = IF.ArgumentParse(L_Param=L_param, _prog=__file__, _intro_msg=_intro_msg, bUseParam=bUseParam)
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

        self.embedding_dim      = self.fundamental_config['OP_SPEC']['EMBEDDING_DIM']
        self.epoch              = self.fundamental_config['OP_SPEC']['EPOCHS']
        self.buffer_size        = self.fundamental_config['OP_SPEC']['BUFFER_SIZE']
        self.validation_split   = self.fundamental_config['OP_SPEC']['VALIDATION_SPLIT']
        self.kl_divergence_weight=float(self.fundamental_config['OP_SPEC']['KLDIVWEIGHT'])
        self.image_size         = self.fundamental_config['DATASPEC']['IMAGE_SIZE']
        self.channels           = self.fundamental_config['DATASPEC']['CHANNELS']
        self.batch_size         = self.fundamental_config['DATASPEC']['BATCH_SIZE']
        # For CUDA setting
        torch.cuda.set_device(self.args.gpu_id)
        self.fundamental_config['DEVICE'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device             = self.fundamental_config['DEVICE']
        # Additional Model
        self.hidden_lyr         = self.fundamental_config['CLASSIFIER']['HIDDEN']
        self.model_file_classfier = self.fundamental_config['CLASSIFIER']['MODEL_FILE']
        # For Miscellaneous
        self.SummaryWriterPATH  = self.fundamental_config['EXPERIMENT_PARAM']['SummaryWriterPATH']
        self.test_samples       = self.fundamental_config['EXPERIMENT_PARAM']['TEST_SAMPLES']
        self.model_file         = self.fundamental_config['EXPERIMENT_PARAM']['MODEL_FILE']
        self.summary_text       = self.fundamental_config['EXPERIMENT_PARAM']['SUMMARYTXT']
        self.text_write_mode    = self.fundamental_config['EXPERIMENT_PARAM']['TEXTWRITEMODE']
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
        self.model_setting()
        # ----------------------------------------------------------------
        # Miscellaneous Setting
        #----------------------------------------------------------------
        self.label_path         = os.path.join(self.root_path, self.args.label_file)
        #self.data_label         = IF.read_yaml(self.label_path)
        self.data_label         = []
        self.data_padding_size  = self.args.data_padding_size
        self.num_workers        = self.args.number_of_workers
        self.quite_mode         = self.args.quite_mode
        self.save_graphic       = self.args.save_graphic

        self.pprint("----------------------------------------------------------------")
        current_time = datetime.now()
        self.pprint(f" Test start time : {current_time}")

        self.pprint("----------------------------------------------------------------")

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
            elif _model_name == 'VAE':
                cf_loss_fn    = []
                cf_loss_fn.append(nn.BCEWithLogitsLoss(reduction=self.loss_fn_param))
                cf_loss_fn.append(self.kl_divergence)
                cf_optimizer = self.c_optimizer(model.parameters(), lr=self.learning_rate)
            else:
                DBG.dbg("Model has not been specified. It is error")
                exit(0)
        except Exception as e:
            DBG.dbg("Error : %s \nProgram Terminated !!" %e)
            exit(0)
        finally:
            self.pprint(f"Model Name    : %s" %_model_name)
            self.pprint(f"Optimizer     : %s.%s" %(self.c_optimizer.__module__, self.c_optimizer.__ne__))
            self.pprint(f" learning_rate : %f" %self.learning_rate)
            if isinstance(cf_loss_fn, list):
                for _k, _fn in enumerate(cf_loss_fn):
                    try:
                        self.pprint(f"Loss Function%d: %s.%s" %(_k, _fn.__name__, _fn.__ne__))
                    except:
                        self.pprint(f"Loss Function%d: %s.%s" % (_k, _fn.__module__, _fn.__ne__))
            else:
                self.pprint(f"Loss Function : %s.%s" %(cf_loss_fn.__module__, cf_loss_fn.__ne__))
            self.pprint(f"    parameter     : %s" %self.loss_fn_param)
            self.pprint(f"Device for OP : %s" %self.device)
            self.pprint(f"Physical GPU  : {self.args.gpu_id}")
            self.pprint(g_line)

        return cf_loss_fn, cf_optimizer

    #----------------------------------------------------------------
    # Internal Service
    #----------------------------------------------------------------
    def kl_divergence(self, mean, logvar):
        loss = -0.5 * (1 + logvar - torch.square(mean) - torch.exp(logvar)).sum(axis=1)
        return loss.mean()

    def model_setting(self):
        if self.args.processing_mode == 1:
            try:
                self.loaded_model   = torch.load(self.model_file)
                _op_msg = "Inference mode" if self.args.inference_mode else "Normal Learning mode"
            except Exception as e:
                _op_msg = "Operation of inference mode is impossible. \nThere are not saved model files"
                _op_msg += f"\n Error : {e}"
                self.args.inference_mode = False
            print(_op_msg + "\n" + g_line)
        elif self.args.processing_mode == 2:
            try:
                self.loaded_model   = torch.load(self.model_file)
            except Exception as e:
                _op_msg = "Operation of inference mode is impossible. \nThere are not saved AutoEncoder model files\n"
                _op_msg+= "You should run Autoencoder processing and generate the AutoEncoder model file\n"
                _op_msg+= f"\n Error : {e}"
                print(_op_msg + "\n" + g_line)
                exit(0)

            try:
                self.loaded_cf_model= torch.load(self.model_file_classfier)
                _op_msg = "Inference mode" if self.args.inference_mode else "Normal Learning mode"
            except Exception as e:
                _op_msg = "Operation of inference mode is impossible. \nThere are not saved model files"
                _op_msg += f"\n Error : {e}"
                print(_op_msg + "\n" + g_line)
                self.args.inference_mode = False
        else: # You should modify the below codes appropriately.
            try:
                self.loaded_model = torch.load(self.model_file)
                self.loaded_cf_model = torch.load(self.model_file_classfier)
                _op_msg = "Inference mode" if self.args.inference_mode else "Normal Learning mode"
            except Exception as e:
                _op_msg = "Operation of inference mode is impossible. \nThere are not saved model files"
                _op_msg += f"\n Error : {e}"
                self.args.inference_mode = False
            print(_op_msg + "\n" + g_line)
    #----------------------------------------------------------------
    # Outer Service
    #----------------------------------------------------------------
    def set_data_label(self, l_label):
        self.data_label = l_label
    def pprint(self, *msg, _active=True):
        if _active:
            _msg = textwrap.dedent(*msg)
            self.l_proc_msg.append(*msg)
            print(_msg)
        else:
            pass

    def write_txt_result(self):
        _content = ''
        for k, _str in enumerate(self.l_proc_msg):
            _content += (_str + '\n')
        _content += g_line
        IF.put_result(_outfile=self.summary_text, _contents=_content, _mode=self.text_write_mode)
# =================================================================
# Test Routine
# =================================================================
if __name__ == "__main__":
    c_conf  = configuration(L_param=[])

    print("===================================================")
    print("Process Finished ")
    print("===================================================")


