#!/usr/bin/python
# -*- coding: utf-8 -*-
###########################################################################
# O'Reilly Generating Deeplearning the 2nd ED.
# Working Directory : ..\work_2025
# 2025 02 09 by Jinwuk Seok
###########################################################################
_description = '''\
================================================================
operation.py : training and validating for Generating Deeplearning the 2nd ED 
                    Written by Jinwuk @ 2025-02-11
================================================================
Example :  There is no Operation instruction. 
'''
g_line      = "----------------------------------------------------------------"

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import my_debug as DBG

class operation_fn:
    def __init__(self,conf_data):
        self.c_config   = conf_data
        self.writer     = SummaryWriter(self.c_config.SummaryWriterPATH)
        # For result data
        self.num_samples    = conf_data.test_samples
        self.num_iters      = np.ceil(self.num_samples / conf_data.batch_size).astype(int)
        self.device         = conf_data.device
        self.output_embs    = None
        self.output_labels  = None
        # sample data for example
        self.sample_classinfo= {}

    # ----------------------------------------------------
    # A single epoch train funcion
    # ----------------------------------------------------
    def train(self, model, dataloader, optimizer, loss_fn):
        # ----------------------------------------------------
        # Train Setting
        # ----------------------------------------------------
        DEVICE      =self.c_config.device
        # ----------------------------------------------------
        # Main routine
        # ----------------------------------------------------
        model.train()
        train_loss = 0
    
        for i, (train_x, train_y) in enumerate(dataloader):
            optimizer.zero_grad()
            train_x     = train_x.to(DEVICE)
            recon_x, _  = model(train_x)
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
        # ----------------------------------------------------
        # Main routine
        # ----------------------------------------------------
        model.eval()
        test_loss = 0
        for i, (test_x, test_y) in enumerate(dataloader):
            test_x = test_x.to(DEVICE)
            with torch.no_grad():
                recon_x, _ = model(test_x)
                loss = loss_fn(recon_x, test_x)
    
            test_loss += loss
        return test_loss / len(dataloader)

    #----------------------------------------------------
    # Train Classifier
    #----------------------------------------------------
    def train_classifier(self, l_model, dataloader, optimizer, loss_fn):
        # ----------------------------------------------------
        # Train Setting
        # ----------------------------------------------------
        DEVICE              = self.c_config.device
        correct_preds       = 0
        total_preds         = 0
        ae_model, cf_model  = l_model[0], l_model[1]
        # ----------------------------------------------------
        # Main routine
        # ----------------------------------------------------
        cf_model.train()
        train_loss = 0.0

        for i, (train_x, train_y) in enumerate(dataloader):
            optimizer.zero_grad()
            train_x, train_y = train_x.to(DEVICE), train_y.to(DEVICE)
            # Get Latent of AutoEncoder
            _, latent   = ae_model(train_x)
            # block the gradient propagation
            latent_x    = latent.detach()
            pred_y      = cf_model(latent_x)
            loss = loss_fn(pred_y, train_y)

            loss.backward()
            optimizer.step()

            correct_preds += sum(pred_y.argmax(dim=-1) == train_y).item()
            total_preds += len(pred_y)
            train_loss += loss.item()

        return train_loss / len(dataloader)
    #----------------------------------------------------
    # Validate Classifier
    #----------------------------------------------------
    def validate_classifier(self, l_model, dataloader, loss_fn):
        # ----------------------------------------------------
        # Train Setting
        # ----------------------------------------------------
        DEVICE = self.c_config.device
        total_preds, correct_preds = 0, 0
        ae_model, cf_model = l_model[0], l_model[1]
        # ----------------------------------------------------
        # Main routine
        # ----------------------------------------------------
        cf_model.eval()
        test_loss = 0
        for i, (test_x, test_y) in enumerate(dataloader):
            test_x, test_y = test_x.to(DEVICE), test_y.to(DEVICE)
            with torch.no_grad():
                _, latent_x = ae_model(test_x)
                modelout_y  = cf_model(latent_x)        #dim(modelout_y) = [128, 10] 128 samples in batch, 10 out nodes
                loss = loss_fn(modelout_y, test_y)

            pred_y          = modelout_y.argmax(dim=-1) #dim(pred_y) = [128] only 1 out-nodes index of 10 out nodes.
            test_loss       += loss.item()
            correct_preds   += sum(pred_y == test_y).item() #dim(test_y)=[128]
            total_preds     += len(pred_y)
            # samples of classification result (for first 128 samples)
            if i == 0:
                self.sample_classinfo= {'test_y': test_y, 'pred_y': pred_y}
            else: pass

        return test_loss / len(dataloader), correct_preds / total_preds

    #----------------------------------------------------
    # Generate Images
    #----------------------------------------------------
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

        return self.output_embs, self.output_labels

    def generate_samples(self, c_config, model, output_embs):
        _device = c_config.device
        # Sampling from the embedding space
        x_min, x_max = output_embs[:, 0].min(), output_embs[:, 0].max()
        y_min, y_max = output_embs[:, 1].min(), output_embs[:, 1].max()

        xs = np.random.uniform(x_min, x_max, size=(18, 1))
        ys = np.random.uniform(y_min, y_max, size=(18, 1))
        samples = np.hstack([xs, ys])

        samples_torch = torch.tensor(samples, device=_device, dtype=torch.float32)
        # print(samples.shape)
        with torch.no_grad():
            output_imgs = model.generate(samples_torch).detach().cpu().numpy()
        output_imgs = output_imgs.transpose((0, 2, 3, 1))
        DBG.dbg(np.shape(output_imgs), _active=c_config.args.debug_mode)

        return samples, output_imgs
    #----------------------------------------------------
    # Record and print the result to each epoch
    #----------------------------------------------------
    def record_result(self, _epoch, train_loss, test_loss, **kwargs):
        s_train_loss = "Train/loss"
        s_valid_loss = "Valid/loss"
        self.writer.add_scalar(s_train_loss, train_loss, _epoch)
        self.writer.add_scalar(s_valid_loss, test_loss, _epoch)
        #----------------------------------------------------
        # **kwargs correct = _correct
        # ----------------------------------------------------
        if len(kwargs) > 0 :
            s_correctness= "Correct"
            _correctness = kwargs['correct']
            self.writer.add_scalar(s_correctness, _correctness, _epoch)
            print(f'Epoch {_epoch + 1: 3d}  ', s_train_loss, f"{train_loss:.4f} ",
              s_valid_loss, f"{test_loss:.4f}", s_correctness, f"{_correctness:.4f}")
        else:
            print(f'Epoch {_epoch + 1: 3d}  ', s_train_loss, f"{train_loss:.4f} ", s_valid_loss, f"{test_loss:.4f}")

    def save_model_parameter(self, model, _file_name):
        torch.save(model.state.dict(), _file_name)
        print("Save the learned model to %s")