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
====================================================
report_op.py : training and validating for Generating Deeplearning the 2nd ED 
                    Written by Jinwuk @ 2025-02-11
====================================================
Example :  There is no Operation instruction. 
'''
g_line      = "----------------------------------------------------"

import numpy as np
import os
from matplotlib import pyplot as plt
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
        self.graphic_path= conf_data.doc
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
        if _mode == 0:
            self.plot_embeds_and_labels(output_embs=output_embs, output_labels=output_labels)
        elif _mode == 1:
            self.plot_samples_and_recon_images(output_embs=output_embs, output_labels=output_labels,
                                               samples=samples, output_imgs=output_imgs)
        else:
            print("No operation for report")

    def plt_show_method(self):
        if self.save_graphic:
            plt.savefig(os.path.join(self.graphic_path, __name__))
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

    def plot_samples_and_recon_images(self, output_embs, output_labels, samples, output_imgs):
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
        self.plt_show_method()

        # Generate new images from sampled embeddings
        fig, axes = plt.subplots(nrows=3, ncols=6, figsize=(10, 6))
        for i in range(output_imgs.shape[0]):
            curr_row = i // 6
            curr_col = i % 6
            ax = axes[curr_row, curr_col]

            ax.set_title(f'({samples[i][0]:.1f}, {samples[i][1]:.1f})')
            ax.axis('off')
            ax.imshow(output_imgs[i], cmap='gray')

        self.plt_show_method()
