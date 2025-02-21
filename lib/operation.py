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
        # Loss function
        self.kl_divergence_weight= conf_data.kl_divergence_weight
    # ----------------------------------------------------
    # Train/Validate AE
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

    def validate(self, model, dataloader, loss_fn):
        # ----------------------------------------------------
        # Validate Setting
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
    # Train / Validate Classifier
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

    def validate_classifier(self, l_model, dataloader, loss_fn):
        # ----------------------------------------------------
        # Validate Setting
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

    # ----------------------------------------------------
    # Train/Validate VAE
    # ----------------------------------------------------
    def train_vae(self, l_model, dataloader, optimizer, l_loss_fn):
        # ----------------------------------------------------
        # Train Setting
        # ----------------------------------------------------
        DEVICE = self.c_config.device
        vae_model = l_model[0]
        bce_loss_fn, kl_loss_fn = l_loss_fn[0], l_loss_fn[1]
        # ----------------------------------------------------
        # Main routine
        # ----------------------------------------------------
        vae_model.train()
        running_bce_loss, running_kl_loss = 0, 0

        for i, (train_x, train_y) in enumerate(dataloader):
            optimizer.zero_grad()
            train_x = train_x.to(DEVICE)
            recon_x, mean, logvar, _ = vae_model(train_x)

            bce_loss    = bce_loss_fn(recon_x, train_x)
            kl_loss     = kl_loss_fn(mean, logvar)
            loss        = bce_loss + kl_loss * self.kl_divergence_weight

            loss.backward()
            optimizer.step()

            running_bce_loss += bce_loss.item()
            running_kl_loss  += kl_loss.item()

        return running_bce_loss / len(dataloader), running_kl_loss / len(dataloader)

    def validate_vae(self, l_model, dataloader, l_loss_fn):
        # ----------------------------------------------------
        # Validate Setting
        # ----------------------------------------------------
        DEVICE = self.c_config.device
        vae_model = l_model[0]
        bce_loss_fn, kl_loss_fn = l_loss_fn[0], l_loss_fn[1]
        # ----------------------------------------------------
        # Main routine
        # ----------------------------------------------------
        vae_model.eval()
        running_bce_loss, running_kl_loss = 0, 0

        for i, (test_x, test_y) in enumerate(dataloader):
            test_x = test_x.to(DEVICE)
            with torch.no_grad():
                recon_x, mean, logvar, _ = vae_model(test_x)
                bce_loss    = bce_loss_fn(recon_x, test_x)
                kl_loss     = kl_loss_fn(mean, logvar)

            running_bce_loss+= bce_loss.item()
            running_kl_loss += kl_loss.item()

        return running_bce_loss / len(dataloader), running_kl_loss / len(dataloader)
    # ----------------------------------------------------
    # Train/Validate VAE for CelebA : GPU Memory 절약 방법
    # ----------------------------------------------------
    # A single epoch train funcion
    def train_vae_celeba(self, l_model, dataloader, optimizer, l_loss_fn):
        # ----------------------------------------------------
        # Train Setting
        # ----------------------------------------------------
        DEVICE  = self.c_config.device
        BETA    = self.c_config.kl_divergence_weight
        vae_model = l_model[0]
        mse_loss_fn, kl_loss_fn = l_loss_fn[0], l_loss_fn[1]
        # ----------------------------------------------------
        # Main routine
        # ----------------------------------------------------
        vae_model.train()
        running_mse_loss = 0
        running_kl_loss = 0

        for i, train_x in enumerate(dataloader):
            optimizer.zero_grad()
            train_x = train_x.to(DEVICE)
            recon_x, mean, logvar = vae_model(train_x)
            mse_loss= mse_loss_fn(recon_x, train_x)
            kl_loss = kl_loss_fn(mean, logvar)

            loss    = mse_loss + BETA * kl_loss

            loss.backward()
            optimizer.step()

            running_mse_loss += mse_loss.item()     # item()은 Tensor를 Scalar로 변경
            running_kl_loss  += kl_loss.item()      # mean()은 Tensor의 전체 평균

            # GPU Memory 절약을 위함
            del train_x, recon_x
            torch.cuda.empty_cache()

        return running_mse_loss / len(dataloader), running_kl_loss / len(dataloader)

    # Validation function
    def validate_vae_celeba(self, l_model, dataloader, l_loss_fn):
        # ----------------------------------------------------
        # Validate Setting
        # ----------------------------------------------------
        DEVICE = self.c_config.device
        vae_model = l_model[0]
        mse_loss_fn, kl_loss_fn = l_loss_fn[0], l_loss_fn[1]
        # ----------------------------------------------------
        # Main routine
        # ----------------------------------------------------
        vae_model.eval()
        running_mse_loss = 0
        running_kl_loss = 0

        for i, test_x in enumerate(dataloader):
            test_x = test_x.to(DEVICE)
            with torch.no_grad():
                recon_x, mean, logvar = vae_model(test_x)
                mse_loss = mse_loss_fn(recon_x, test_x)
                kl_loss = kl_loss_fn(mean, logvar)

            running_mse_loss += mse_loss.item()
            running_kl_loss  += kl_loss.item()

            del test_x, recon_x
            torch.cuda.empty_cache()

        return running_mse_loss / len(dataloader), running_kl_loss / len(dataloader)
    # Performace Optimization for pytorch > 2.0
    def set_performace_optimization(self, model):
        if torch.__version__.split('.')[0] == '2':
            torch.set_float32_matmul_precision(self.c_config.float_precision)
            # It is important to use eager backend here to avoid
            # distribution mismatch in training and predicting
            c_vae = torch.compile(model, backend=self.c_config.torch_compile)
            print('model compiled')
        else: pass
        return model

    #----------------------------------------------------
    # Generate Images
    #----------------------------------------------------
    def generate_embeds_and_labels(self, model, test_loader):
        _model_name = model.__module__
        for i, (test_x, test_y) in enumerate(test_loader):
            test_x = test_x.to(self.device)
            with torch.no_grad():
                if _model_name == "model.auto_encoder":
                    embeddings = model.encoder(test_x)
                elif _model_name == "model.variable_autoencoder":
                    _mean, _logvar  = model.encoder(test_x)
                    embeddings      = model.reparameterize(mean=_mean, logvar=_logvar)
                else:
                    DBG.dbg("No module name specified. it is error")
                    exit(0)

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
        self.writer.add_scalar("Train/loss", train_loss, _epoch)
        self.writer.add_scalar("Valid/loss", test_loss, _epoch)
        #----------------------------------------------------
        # **kwargs correct = _correct
        # ----------------------------------------------------
        if len(kwargs) > 0 :
            _correctness = kwargs['correct']
            self.writer.add_scalar("Correct", _correctness, _epoch)
            self.c_config.pprint(f"Epoch {_epoch + 1: 3d}  Train/loss  {train_loss:.4f}  Valid/loss  {test_loss:.4f}  Correct  {_correctness:.4f}")
        else:
            self.c_config.pprint(f"Epoch {_epoch + 1: 3d}  Train/loss  {train_loss:.4f}  Valid/loss  {test_loss:.4f}")

    def record_vae_result(self, _epoch, **kwargs):
        # ----------------------------------------------------
        # Setting parameters with **kwargs
        # ----------------------------------------------------
        train_loss_bce  = kwargs['train_loss_bce']
        train_loss_kl   = kwargs['train_loss_kl']
        test_loss_bce   = kwargs['test_loss_bce']
        test_loss_kl    = kwargs['test_loss_kl']

        self.writer.add_scalar("Train/loss (BCE)", train_loss_bce, _epoch)
        self.writer.add_scalar("Train/loss (KL)",  train_loss_kl,  _epoch)
        self.writer.add_scalar("Test/loss (BCE)",  test_loss_bce,  _epoch)
        self.writer.add_scalar("Test/loss (KL)",   test_loss_kl,   _epoch)
        #----------------------------------------------------
        # **kwargs correct = _correct
        # ----------------------------------------------------
        self.c_config.pprint(f"Epoch {_epoch + 1: 3d}  Train/loss (BCE)  {train_loss_bce:.4f}  Test/loss (BCE)  {test_loss_bce:.4f}  Train/loss (KL)  {train_loss_kl:.4f}  Test/loss (KL)  {test_loss_kl:.4f}")

    def record_vae_celebA_result(self, _epoch, **kwargs):
        # ----------------------------------------------------
        # Setting parameters with **kwargs
        # ----------------------------------------------------
        train_loss_bce  = kwargs['train_loss_mse']
        train_loss_kl   = kwargs['train_loss_kl']
        test_loss_bce   = kwargs['test_loss_mse']
        test_loss_kl    = kwargs['test_loss_kl']

        self.writer.add_scalar("Train/loss (MSE)", train_loss_bce, _epoch)
        self.writer.add_scalar("Train/loss (KL)",  train_loss_kl,  _epoch)
        self.writer.add_scalar("Test/loss (MSE)",  test_loss_bce,  _epoch)
        self.writer.add_scalar("Test/loss (KL)",   test_loss_kl,   _epoch)
        #----------------------------------------------------
        # **kwargs correct = _correct
        # ----------------------------------------------------
        self.c_config.pprint(f"Epoch {_epoch + 1: 3d}  Train/loss (MSE)  {train_loss_bce:.4f}  Test/loss (MSE)  {test_loss_bce:.4f}  Train/loss (KL)  {train_loss_kl:.4f}  Test/loss (KL)  {test_loss_kl:.4f}")





    def save_model_parameter(self, model, _file_name):
        torch.save(model.state.dict(), _file_name)
        print("Save the learned model to %s")


