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
Example :  python main_test.py  
'''
g_line      = "----------------------------------------------------"

from configuration import configuration
from data_proc import Fashion_MNIST
from model.auto_encoder import Encoder
from model.auto_encoder import Decoder
from model.auto_encoder import AutoEncoder
from operation import operation_fn
from report_op import report_AutoEncoder
import time
# =================================================================
# Main Routine
# =================================================================
if __name__ == "__main__":
    L_param=[]
    c_conf = configuration(L_param=L_param, _intro_msg=_description)
    c_data = Fashion_MNIST(conf_data=c_conf)
    c_oper = operation_fn(conf_data=c_conf)
    c_repo = report_AutoEncoder(conf_data=c_conf, figsize=(8, 8), alpha=0.8, s=3)
    # ----------------------------------------------------------------
    # 1. Network Setting
    # ----------------------------------------------------------------
    c_ae = AutoEncoder(c_config=c_conf).to(c_conf.device)
    c_ae.print_summary(_shape=(c_conf.channels, c_conf.image_size, c_conf.image_size), _quite=c_conf.args.quite_mode)
    # Model Setting
    cf_loss_fn, cf_optimizer = c_conf(model=c_ae)
    # ----------------------------------------------------------------
    # 2. Data setting
    # ----------------------------------------------------------------
    train_loader, test_loader   = c_data.get_dataloaders()

    # ----------------------------------------------------------------
    # 3. Train and Evaluate
    # ----------------------------------------------------------------
    start_time = time.time()

    for i in range(c_conf.epoch):
        train_loss  = c_oper.train(model=c_ae, dataloader=train_loader, optimizer=cf_optimizer, loss_fn=cf_loss_fn)
        test_loss   = c_oper.validate(model=c_ae, dataloader=test_loader, loss_fn=cf_loss_fn)

        c_oper.record_result(_epoch=i, train_loss=train_loss, test_loss=test_loss)

    elapsed_time = time.time() - start_time
    # ----------------------------------------------------------------
    # 4. Report Result
    # ----------------------------------------------------------------
    print(f"Processing Time : {elapsed_time: .2f}")
    c_repo(model=c_ae, test_loader=test_loader)

    print("===================================================")
    print("Process Finished ")
    print("===================================================")