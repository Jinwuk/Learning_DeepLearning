#!/usr/bin/python
# -*- coding: utf-8 -*-
###########################################################################
# O'Reilly Generating Deeplearning the 2nd ED.
# Working Directory : ..\work_2025
# 2025 02 09 by Jinwuk Seok
###########################################################################
import torch
import my_debug as DBG

_description = '''\
================================================================
report_op.py : training and validating for Generating Deeplearning the 2nd ED 
                    Written by Jinwuk @ 2025-02-11
================================================================
Example :  There is no Operation instruction. 
'''
g_line      = "----------------------------------------------------------------"

import numpy as np
import os
from matplotlib import pyplot as plt
from scipy.stats import norm

class report_AutoEncoder:
    def __init__(self, conf_data, c_op, **kwargs):
        #----------------------------------------------------
        # Spec of kwargs
        # figsize=(8, 8), alpha=0.8, s=3
        # ----------------------------------------------------
        # Data for plotting
        self.c_op       = c_op
        self.c_conf     = conf_data
        self.save_graphic= conf_data.save_graphic
        self.graphic_path= conf_data.doc_path
        self._count     = 0
        self.data_label = conf_data.data_label
        # For Plotting window
        self.figsize    = kwargs['figsize']
        self.alpha      = kwargs['alpha']
        self.s          = kwargs['s']

    def __call__(self, model, test_loader, _mode=1):
        # 1. Save Learned model
        torch.save(model.state_dict(), self.c_conf.model_file)
        # 2. Generate Embedding Points and Labels
        output_embs, output_labels  = self.c_op.generate_embeds_and_labels(model=model, test_loader=test_loader)
        samples, output_imgs        = self.c_op.generate_samples(c_config=self.c_conf, model=model, output_embs=output_embs)
        # 3. Plotting the results
        if _mode == 0:
            self.plot_embeds_and_labels(output_embs=output_embs, output_labels=output_labels)
        elif _mode == 1:
            self.plot_samples_and_recon_images(output_embs=output_embs, output_labels=output_labels,
                                               samples=samples, output_imgs=output_imgs)
        else:
            print("No operation for report")

        return [output_embs, output_labels], [samples, output_imgs]

    def plt_show_method(self):
        if self.save_graphic:
            _file_name  = __name__ + f"_{self._count:d}.png"
            g_file_name = os.path.join(self.graphic_path, _file_name)
            plt.savefig(g_file_name)
            self._count += 1
        else:
            plt.show()

    def plot_embeds_and_labels(self, output_embs, output_labels):
        _x = output_embs[:, 0]
        _y = output_embs[:, 1]
        _class = output_labels

        plt.figure(figsize=self.figsize)
        plt.scatter( _x, _y, c=_class,
                    alpha=self.alpha, s=self.s)
        plt.colorbar()
        self.plt_show_method()

    def plot_samples_and_recon_images(self, output_embs, output_labels, samples, output_imgs, **kwargs):
        # Processing of kwargs
        _exist_kwargs = True if 'c_result' in kwargs else False
        if _exist_kwargs:
            c_result        = kwargs['c_result']
            test_y, pred_y  = c_result['test_y'], c_result['pred_y']
        else : pass
        # First scattering Data 5000
        plt.figure(figsize=(6, 6))
        plt.scatter(output_embs[:, 0],
                    output_embs[:, 1],
                    c=output_labels,
                    alpha=0.5,
                    s=3)
        plt.scatter(samples[:, 0],
                    samples[:, 1],
                    c='red',
                    alpha=0.5,
                    s=20)
        plt.colorbar()
        self.plt_show_method()

        # processing for kwargs
        if _exist_kwargs:
            print(f" index        test  prediction")
        else: pass
        # Generate new images from sampled embeddings
        fig, axes = plt.subplots(nrows=3, ncols=6, figsize=(10, 6))
        for i in range(output_imgs.shape[0]):
            curr_row = i // 6
            curr_col = i % 6
            ax = axes[curr_row, curr_col]

            ax.set_title(f'({i:2d}: {samples[i][0]:.1f}, {samples[i][1]:.1f})')
            ax.axis('off')
            ax.imshow(output_imgs[i], cmap='gray')

            if _exist_kwargs:
                _test_idx, _pred_idx = int(test_y[i]), int(pred_y[i])
                _msg_str  = f"{i:2d} ({samples[i][0]:5.1f}, {samples[i][1]:5.1f}) {test_y[i]:2d}  {pred_y[i]:2d}"
                _msg_str += f" | {self.data_label[_test_idx]:11s}  {self.data_label[_pred_idx]:11s}"
                print(_msg_str)
            else: pass
        self.plt_show_method()

class report_Classfier_for_AutoEncoder:
    def __init__(self, conf_data, c_op, **kwargs):
        #----------------------------------------------------
        # Spec of kwargs
        # ae_repo = AutoEncoder class
        # ----------------------------------------------------
        # kwargs
        self.ae_repo = kwargs['ae_repo']
        # Data for plotting
        self.c_op       = c_op
        self.c_conf     = conf_data
        self._count     = 0
    def __call__(self, l_model, test_loader, **kwargs):
        #----------------------------------------------------
        # Spec of kwargs
        # ----------------------------------------------------
        ae_model, cf_model  = l_model[0], l_model[1]
        c_result            = kwargs['c_result']
        # 1. Save Learned model
        torch.save(cf_model.state_dict(), self.c_conf.model_file_classfier)
        # 2. print classification result
        output_embs, output_labels  = self.c_op.generate_embeds_and_labels(model=ae_model, test_loader=test_loader)
        samples, output_imgs        = self.c_op.generate_samples(c_config=self.c_conf, model=ae_model, output_embs=output_embs)

        self.ae_repo.plot_samples_and_recon_images(output_embs=output_embs, output_labels=output_labels,
                                               samples=samples, output_imgs=output_imgs, c_result=c_result)

class report_VAE:
    def __init__(self, conf_data, c_op, **kwargs):
        #----------------------------------------------------
        # Spec of kwargs
        # ae_repo = AutoEncoder class
        # ----------------------------------------------------
        # kwargs
        self.c_rep_ae   = kwargs['rep_ae']
        # Data for plotting
        self.c_op       = c_op
        self.c_conf     = conf_data
        self._count     = 0

    def plot_embs_distribution(self, l_embs):
        [output_embs, output_labels]    = l_embs

        p = norm.cdf(output_embs)

        plt.figure(figsize=(9, 8))
        plt.scatter(p[:, 0],
                    p[:, 1],
                    c=output_labels,
                    alpha=0.8,
                    s=3)
        plt.colorbar()
        self.c_rep_ae.plt_show_method()

# ----------------------------------------------------------------
# For the report of VAE_for_celeb_A
# ----------------------------------------------------------------
class report_VAE_celeb_a(report_AutoEncoder):
    def __init__(self,conf_data, c_op, **kwargs):
        super().__init__(conf_data=conf_data, c_op=c_op, kwargs=kwargs)

    def __call__(self, model, test_loader, _mode=1):
        # 1. Save Learned model
        torch.save(model.state_dict(), self.c_conf.model_file)

        DBG.dbg("Debug proc")