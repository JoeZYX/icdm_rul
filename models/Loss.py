import sys
sys.path.append("..")
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


class Weighted_MSE_Loss(nn.Module):
    
    def __init__(self, seq_length=20, sigma_faktor=10, anteil=15,device = "cuda"):

        super(Weighted_MSE_Loss, self).__init__()
        self.seq_length = seq_length
        self.sigma      = seq_length/sigma_faktor
        
        x = np.linspace(1, seq_length, seq_length)
        #mu = self.seq_length/2
        # mu = 0
        mu = seq_length
        y = stats.norm.pdf(x, mu, self.sigma)
        #y = 2*np.max(y)-y
        y = y + np.max(y)/anteil
        print(anteil, sigma_faktor)
        y = y/np.sum(y)*seq_length
        plt.plot(x, y)
        plt.show()
        with torch.no_grad():
            self.weights = torch.Tensor(y).double().to(device)  
        #self.weights = torch.Tensor([1]*seq_length).double().to(device)  
        
    def forward(self, pred, target):
        se  = (pred-target)**2
        out = se * self.weights.expand_as(target)
        loss = out.mean() 
        return loss


criterion_dict = {"MSE"            :nn.MSELoss,
                  "CrossEntropy"   :nn.CrossEntropyLoss,
                  "WeightMSE"      :Weighted_MSE_Loss}

class HTSLoss(nn.Module):
    def __init__(self, 
                 enc_pred_loss = "MSE", 
                 final_pred_loss = "WeightMSE", 
                 seq_length = 40,
                 sigma_faktor = 10,
                 anteil      = 15,
                 final_smooth_loss = None,
                 d_layers = 2, 
                 lambda_final_pred = 2,
                 lambda_final_smooth = 1,
                 include_enc_loss = False,
                 device  = "cuda"):

        super(HTSLoss, self).__init__()
        self.d_layers               = d_layers
        self.include_enc_loss       = include_enc_loss
        self.enc_pred_loss          = enc_pred_loss   
        print("enc_pred_criterion")
        if self.enc_pred_loss == "WeightMSE":
            self.enc_pred_criterion     =  criterion_dict["WeightMSE"](seq_length, sigma_faktor, anteil,device)
        else:
            self.enc_pred_criterion     =  criterion_dict[self.enc_pred_loss]()
        if self.d_layers > 0:
            self.final_pred_loss        =  final_pred_loss  # this is a list , it can also be none, if it is none, d_layers should = 0
            self.final_pred_criterion   =  None
            if final_pred_loss is not None:
                print("final_pred_loss")
                if final_pred_loss == "WeightMSE":
                    print("WeightMSE")
                    self.final_pred_criterion = criterion_dict["WeightMSE"](seq_length, sigma_faktor, anteil,device)
                else:
                    self.final_pred_criterion = criterion_dict[self.final_pred_loss]()
            self.lambda_final_pred       = lambda_final_pred


            self.final_smooth_loss      =  final_smooth_loss

            if final_smooth_loss is not None:
                print("final_smooth_loss")
                self.final_smooth_criterion = None
            else:
                self.final_smooth_criterion = None
            self.lambda_final_smooth     = lambda_final_smooth

            self.d_layers               =  d_layers


    def forward(self, outputs, batch_y):
        if self.d_layers == 0: 
            # no decoder , only the prediction from encoder "enc_pred"
            enc_pred              = outputs[0]
            enc_pred_loss         = self.enc_pred_criterion(enc_pred, batch_y)
            loss                  = enc_pred_loss
            return loss
        else : 
            # yes decoder, there are two predictions one is prediction from encoder "enc_pred"
            #                                        one is prediction from decoder "final_pred"
            if self.include_enc_loss:
                enc_pred              = outputs[0]
                final_pred            = outputs[1]
                enc_pred_loss         = self.enc_pred_criterion(enc_pred, batch_y)
                dec_pred_loss         = self.final_pred_criterion(final_pred, batch_y)
                loss                  = enc_pred_loss + self.lambda_final_pred*dec_pred_loss
                if self.final_smooth_criterion is not None:
                    smooth_loss       = self.final_smooth_criterion(final_pred)
                    loss              = loss + self.lambda_final_smooth*smooth_loss

                return loss
            else:
                final_pred            = outputs[1]
                loss                  = self.final_pred_criterion(final_pred, batch_y)
                return loss