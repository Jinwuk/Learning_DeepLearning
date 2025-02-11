#!/usr/bin/python
# -*- coding: utf-8 -*-
###########################################################################
# O'Reilly Generating Deeplearning the 2nd ED.
# Working Directory : ..\work_2025
# 2025 02 09 by Jinwuk Seok
###########################################################################
_description = '''\
====================================================
report_op.py : training and validating for Generating Deeplearning the 2nd ED 
                    Written by Jinwuk @ 2025-02-11
====================================================
Example :  There is no Operation instruction. 
'''
g_line      = "----------------------------------------------------"

import numpy as np
import torch
from matplotlib import pyplot as plt
class report_AutoEncoder:
    def __init__(self, conf_data, **kwargs):
        #----------------------------------------------------
        # Spec of kwargs
        # figsize=(8, 8), alpha=0.8, s=3
        # ----------------------------------------------------
        # For result data
        self.num_samples    = conf_data.test_samples
        self.num_iters      = np.ceil(self.num_samples / conf_data.batch_size).astype(int)
        self.device         = conf_data.device
        self.output_embs    = None
        self.output_labels  = None
        # For Plotting window
        self.figsize        = kwargs['figsize']
        self.alpha          = kwargs['alpha']
        self.s              = kwargs['s']

    def __call__(self, model, test_loader):
        self.generate_embeds_and_labels(model=model, test_loader=test_loader)
        self.plot_embeds_and_labels()

    def generate_embeds_and_labels(self, model, test_loader):
        for i, (test_x, test_y) in enumerate(test_loader):
            test_x = test_x.to(self.device)
            with torch.no_grad():
                embeddings = model.encoder(test_x)
            if i == 0:
                self.output_embs = embeddings
                self.output_labels = test_y
            else:
                self.output_embs = torch.concatenate([self.output_embs, embeddings])
                self.output_labels = torch.concatenate([self.output_labels, test_y])
            if i == self.num_iters - 1: break

        self.output_embs     = self.output_embs.detach().cpu().numpy()
        self.output_labels   = self.output_labels.detach().cpu().numpy()

    def plot_embeds_and_labels(self):
        _x = self.output_embs[:, 0]
        _y = self.output_embs[:, 1]
        _class = self.output_labels
        plt.figure(figsize=self.figsize)
        plt.scatter( _x, _y, c=_class,
                    alpha=self.alpha, s=self.s)
        plt.colorbar()
        plt.show()