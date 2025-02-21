#!/usr/bin/python
# -*- coding: utf-8 -*-
###########################################################################
# O'Reilly Generating Deeplearning the 2nd ED.
# Working Directory : ..\work_2025
# 2025 02 09 by Jinwuk Seok
###########################################################################
_description = '''\
================================================================
main_test.py : main_test.py for Generating Deeplearning the 2nd ED 
                    Written by Jinwuk @ 2025-02-09
================================================================
Example :  python main_test.py  
'''
g_line      = "----------------------------------------------------------------"

import os, sys
root_path   = os.getcwd()
model_path  = os.path.join(root_path, 'model')
lib_path    = os.path.join(root_path, 'lib')
sys.path.append(os.path.join(root_path, model_path))
sys.path.append(os.path.join(root_path, lib_path))

from configuration import configuration
from lib.data_proc import Fashion_MNIST
from lib.data_proc import CelebA
from model.auto_encoder import AutoEncoder
from model.auto_encoder import Classifier_for_autoencoder
from model.variable_autoencoder import VAE
from model.vae_for_celebA import VAE_4_CELEBA
from lib.operation import operation_fn
from lib.report_op import report_AutoEncoder
from lib.report_op import report_Classfier_for_AutoEncoder
from lib.report_op import report_VAE
from lib.report_op import report_VAE_celeb_a
import lib.processing as proc

import torch
import time

# =================================================================
# Main Routine
# =================================================================
if __name__ == "__main__":
    L_param= []
    c_conf = configuration(L_param=L_param, _intro_msg=_description)
    if c_conf.args.processing_mode > 0:
        proc_function = []
        proc_function.append(proc.standard_autoencoder_proc)
        proc_function.append(proc.autoencoder_classfication)
        proc_function.append(proc.vae_fashion_MNIST)
        proc_function[c_conf.args.processing_mode - 1](c_conf=c_conf, _intro_msg=_description)
        sys.exit()
    else: pass
    # ----------------------------------------------------------------
    # 0. Operation Setting
    # ----------------------------------------------------------------
    c_data      = CelebA(conf_data=c_conf)
    c_oper      = operation_fn(conf_data=c_conf)
    c_repo      = report_VAE_celeb_a(conf_data=c_conf, c_op=c_oper, figsize=(8, 8), alpha=0.8, s=3)
    # ----------------------------------------------------------------
    # 1. Network Setting
    # ----------------------------------------------------------------
    c_vae = VAE_4_CELEBA(c_config=c_conf).to(c_conf.device)
    c_vae = c_oper.set_performace_optimization(c_vae)
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
        print("Evaluation begin")
        #train_loss_mse, train_loss_kl= c_oper.validate_vae_celeba(l_model=lc_model, dataloader=train_loader, l_loss_fn=vae_loss_fn)
        #test_loss_mse,  test_loss_kl = c_oper.validate_vae_celeba(l_model=lc_model, dataloader=test_loader, l_loss_fn=vae_loss_fn)
        #c_conf.pprint(f"Train/loss (MSE)  {train_loss_mse:.4f}  Test/loss (MSE)  {test_loss_mse:.4f}  Train/loss (KL)  {train_loss_kl:.4f}  Test/loss (KL)  {test_loss_kl:.4f}")

        pretrained_model = torch.load(c_conf.model_file)
        c_vae.load_state_dict(pretrained_model)
        original_x, recons_x = c_oper.reconstruction_image_from_vae(model=c_vae, dataloader=test_loader)
        c_repo.plot_reconstruction_image(original_x=original_x, recons_x=recons_x)
        c_repo.plot_latent_space_distribution(model=c_vae)
        generated_faces, [g_width, g_height]  = c_oper.generate_new_faces(model=c_vae)
        c_repo.plot_new_faces_from_vae(generated_faces=generated_faces, width=g_width, height=g_height)
        print("debugging")

    else:
        # 1. Normal AutoLearning processing
        for i in range(c_conf.epoch):
            train_loss_mse, train_loss_kl = c_oper.train_vae_celeba(l_model=lc_model, dataloader=train_loader, optimizer=vae_optimizer, l_loss_fn=vae_loss_fn)
            test_loss_mse, test_loss_kl   = c_oper.validate_vae_celeba(l_model=lc_model, dataloader=test_loader, l_loss_fn=vae_loss_fn)
            c_oper.record_vae_celebA_result(_epoch=i, train_loss_mse=train_loss_mse, train_loss_kl=train_loss_kl,
                                            test_loss_mse=test_loss_mse, test_loss_kl=test_loss_kl)
    elapsed_time = time.time() - start_time
    # ----------------------------------------------------------------
    # 4. Report Result
    # ----------------------------------------------------------------
    c_conf.pprint(f"\nProcessing Time : {elapsed_time: .2f} sec")
    # ----------------------------------------------------------------
    _msg  = f"\n{__name__} : Save graphics mode. Please wait\n" if c_conf.args.save_graphic else f"\n{__name__} : Please Check Window\n"
    print(g_line + _msg + g_line)
    # ----------------------------------------------------------------
    # Generate samples
    c_repo(model=c_vae, data_loader=test_loader)

    #l_embs, l_samples = c_repo(model=c_vae, test_loader=test_loader)
    #c_repo_vae  = report_VAE(conf_data=c_conf, c_op=c_oper, rep_ae=c_repo)
    #c_repo_vae.plot_embs_distribution(l_embs = l_embs)
    c_conf.write_txt_result()

    print("===================================================")
    print("Process Finished ")
    print("===================================================")