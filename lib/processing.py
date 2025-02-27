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
from model.auto_encoder import Classifier_for_autoencoder
from model.variable_autoencoder import VAE
from lib.operation import operation_fn
from lib.report_op import report_AutoEncoder
from lib.report_op import report_Classfier_for_AutoEncoder
from lib.report_op import report_VAE

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

def autoencoder_classfication(c_conf, _intro_msg=_description, **kwargs):
    # ----------------------------------------------------------------
    # 0. Network Setting
    #----------------------------------------------------------------
    c_data = Fashion_MNIST(conf_data=c_conf)
    c_oper = operation_fn(conf_data=c_conf)
    _c_ae_repo = report_AutoEncoder(conf_data=c_conf, c_op=c_oper, figsize=(8, 8), alpha=0.8, s=3)
    c_repo = report_Classfier_for_AutoEncoder(conf_data=c_conf, c_op=c_oper, ae_repo=_c_ae_repo)
    # ----------------------------------------------------------------
    # 1. Network Setting
    # ----------------------------------------------------------------
    c_ae = AutoEncoder(c_config=c_conf).to(c_conf.device)
    c_cf = Classifier_for_autoencoder(c_config=c_conf).to(c_conf.device)
    c_ae.print_summary(_shape=(c_conf.channels, c_conf.image_size, c_conf.image_size), _quite=c_conf.args.quite_mode)
    c_cf.print_summary(_shape=(1, c_conf.embedding_dim), _quite=c_conf.args.quite_mode)
    # Model Setting
    ae_loss_fn, ae_optimizer = c_conf(model=c_ae)
    cf_loss_fn, cf_optimizer = c_conf(model=c_cf)
    lc_model = []
    lc_model.append(c_ae)
    lc_model.append(c_cf)
    # ----------------------------------------------------------------
    # 2. Data setting
    # ----------------------------------------------------------------
    train_loader, test_loader   = c_data.get_dataloaders()
    # ----------------------------------------------------------------
    # 3. Train and Evaluate
    # ----------------------------------------------------------------
    # 3.1 Load AutoEncdoer model
    c_ae.load_state_dict(c_conf.loaded_model)
    start_time = time.time()
    if c_conf.args.inference_mode :
        # Only evaluation processing (Verify)
        c_cf.load_state_dict(c_conf.loaded_cf_model)
        train_loss, _correct_tr= c_oper.validate_classifier(l_model=lc_model, dataloader=train_loader, loss_fn=cf_loss_fn)
        test_loss,  _correct_te= c_oper.validate_classifier(l_model=lc_model, dataloader=test_loader, loss_fn=cf_loss_fn)

        c_conf.pprint(f"Train/loss  {train_loss:.4f} Valid/loss {test_loss:.4f} Correct_TR {_correct_tr:.4f} Correct_TE {_correct_te:.4f}")
    else:
        # 1. Normal AutoLearning processing
        for i in range(c_conf.epoch):
            train_loss          = c_oper.train_classifier(l_model=lc_model, dataloader=train_loader,
                                                  optimizer=cf_optimizer, loss_fn=cf_loss_fn)
            test_loss, _correct = c_oper.validate_classifier(l_model=lc_model, dataloader=test_loader, loss_fn=cf_loss_fn)
            c_oper.record_result(_epoch=i, train_loss=train_loss, test_loss=test_loss, correct=_correct)

    elapsed_time = time.time() - start_time
    # ----------------------------------------------------------------
    # 4. Report Result
    # ----------------------------------------------------------------
    c_conf.pprint(f"\nProcessing Time : {elapsed_time: .2f} sec")
    # ----------------------------------------------------------------
    _msg  = f"\n{__name__} : Save graphics mode. Please wait\n" if c_conf.args.save_graphic else f"\n{__name__} : Please Check Window\n"
    print(g_line + _msg + g_line)
    # ----------------------------------------------------------------
    c_repo(l_model=lc_model, test_loader=test_loader, c_result=c_oper.sample_classinfo)
    c_conf.write_txt_result()

def vae_fashion_MNIST(c_conf, _intro_msg=_description, **kwargs):
    # ----------------------------------------------------------------
    # 0. Operation Setting
    #----------------------------------------------------------------
    c_data      = Fashion_MNIST(conf_data=c_conf)
    c_oper      = operation_fn(conf_data=c_conf)
    c_repo      = report_AutoEncoder(conf_data=c_conf, c_op=c_oper, figsize=(8, 8), alpha=0.8, s=3)
    # ----------------------------------------------------------------
    # 1. Network Setting
    # ----------------------------------------------------------------
    c_vae = VAE(c_config=c_conf).to(c_conf.device)
    c_vae.print_summary(_shape=(c_conf.channels, c_conf.image_size, c_conf.image_size), _quite=c_conf.args.quite_mode)
    # Model Setting
    vae_loss_fn, vae_optimizer = c_conf(model=c_vae)
    lc_model = []
    lc_model.append(c_vae)
    # ----------------------------------------------------------------
    # 2. Data setting
    # ----------------------------------------------------------------
    train_loader, test_loader   = c_data.get_dataloaders()
    # ----------------------------------------------------------------
    # 3. Train and Evaluate
    # ----------------------------------------------------------------
    # 3.1 Load AutoEncdoer model
    start_time = time.time()
    if c_conf.args.inference_mode :
        # Only evaluation processing (Verify)
        train_loss, _correct_tr= c_oper.validate_classifier(l_model=lc_model, dataloader=train_loader, loss_fn=vae_loss_fn)
        test_loss,  _correct_te= c_oper.validate_classifier(l_model=lc_model, dataloader=test_loader, loss_fn=vae_loss_fn)

        c_conf.pprint(f"Train/loss  {train_loss:.4f} Valid/loss {test_loss:.4f} Correct_TR {_correct_tr:.4f} Correct_TE {_correct_te:.4f}")
    else:
        # 1. Normal AutoLearning processing
        for i in range(c_conf.epoch):
            train_loss_bce, train_loss_kl = c_oper.train_vae   (l_model=lc_model, dataloader=train_loader, optimizer=vae_optimizer, l_loss_fn=vae_loss_fn)
            test_loss_bce, test_loss_kl   = c_oper.validate_vae(l_model=lc_model, dataloader=test_loader, l_loss_fn=vae_loss_fn)
            c_oper.record_vae_result(_epoch=i, train_loss_bce=train_loss_bce, train_loss_kl=train_loss_kl,
                                     test_loss_bce=test_loss_bce, test_loss_kl=test_loss_kl)

    elapsed_time = time.time() - start_time
    # ----------------------------------------------------------------
    # 4. Report Result
    # ----------------------------------------------------------------
    c_conf.pprint(f"\nProcessing Time : {elapsed_time: .2f} sec")
    # ----------------------------------------------------------------
    _msg  = f"\n{__name__} : Save graphics mode. Please wait\n" if c_conf.args.save_graphic else f"\n{__name__} : Please Check Window\n"
    print(g_line + _msg + g_line)
    # ----------------------------------------------------------------
    # Generate embs and samples
    l_embs, l_samples = c_repo(model=c_vae, test_loader=test_loader)
    c_repo_vae  = report_VAE(conf_data=c_conf, c_op=c_oper, rep_ae=c_repo)
    c_repo_vae.plot_embs_distribution(l_embs = l_embs)
    c_conf.write_txt_result()
