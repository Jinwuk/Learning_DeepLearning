#!/usr/bin/python
# -*- coding: utf-8 -*-
###########################################################################
# O'Reilly Generating Deeplearning the 2nd ED.
# Working Directory : ..\work_2025
# 여기에서의 함수는 이미 main_test에서 테스트가 끝난 함수를 보관하기 위해서이다.
# 2025 02 09 by Jinwuk Seok
###########################################################################
_description = '''\
================================================================
processing.py : processing.py for Generating Deeplearning the 2nd ED 
                    Written by Jinwuk @ 2025-02-09
================================================================
Example :  python main_test.py  
'''
g_line      = "----------------------------------------------------------------"

from configuration import configuration
from lib.data_proc import Fashion_MNIST
from model.auto_encoder import AutoEncoder
from lib.operation import operation_fn
from lib.report_op import report_AutoEncoder
import time

def standard_autoencoder_proc(c_conf, _intro_msg=_description, **kwargs):
    # ----------------------------------------------------------------
    # 0. Network Setting
    #----------------------------------------------------------------
    c_data = Fashion_MNIST(conf_data=c_conf)
    c_oper = operation_fn(conf_data=c_conf)
    c_repo = report_AutoEncoder(conf_data=c_conf, c_op=c_oper, figsize=(8, 8), alpha=0.8, s=3)
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
    if c_conf.args.inference_mode :
        # Only evaluation processing (Verify)
        c_ae.load_state_dict(c_conf.loaded_model)
        train_loss  = c_oper.validate(model=c_ae, dataloader=train_loader, loss_fn=cf_loss_fn)
        test_loss   = c_oper.validate(model=c_ae, dataloader=test_loader, loss_fn=cf_loss_fn)
        c_conf.pprint(f'Epoch 0  ', "Train/loss", f"{train_loss:.4f}   ", "Valid/loss", f"{test_loss:.4f}")
    else:
        # Normal Learning processing
        for i in range(c_conf.epoch):
            train_loss  = c_oper.train(model=c_ae, dataloader=train_loader, optimizer=cf_optimizer, loss_fn=cf_loss_fn)
            test_loss   = c_oper.validate(model=c_ae, dataloader=test_loader, loss_fn=cf_loss_fn)
            c_oper.record_result(_epoch=i, train_loss=train_loss, test_loss=test_loss)

    elapsed_time = time.time() - start_time
    # ----------------------------------------------------------------
    # 4. Report Result
    # ----------------------------------------------------------------
    c_conf.pprint(f"\nProcessing Time : {elapsed_time: .2f} sec")
    # ----------------------------------------------------------------
    _msg = f"\n{__name__} : Save graphics mode. Please wait\n" if c_conf.args.save_graphic else f"\n{__name__} : Please Check Window\n"
    print(g_line + _msg + g_line)
    # ----------------------------------------------------------------
    c_repo(model=c_ae, test_loader=test_loader)
    c_conf.write_txt_result()

    print("===================================================")
    print("Process Finished ")
    print("===================================================")
