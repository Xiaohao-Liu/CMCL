import torch

import torch.nn as nn
import torch.optim as optim
from .base import Base, get_projector
from dataloader import ModalityType


class DNS(Base):
    def __init__(self, embed_dim, n_step, lambda_min):
        super(DNS, self).__init__(embed_dim, n_step)
        
        self.feature_covariance = {modality: [torch.zeros(embed_dim, embed_dim), False] for modality in vars(ModalityType).values()}
        
        self.feature_covariance2 = {modality: [torch.zeros(embed_dim, embed_dim), False] for modality in vars(ModalityType).values()}
        
        self.current_training_modalities = []
        self.n_bar = 0
        
        self.lambda_min = lambda_min
        
        self.lr = 1e-4
        
        self.higher_order_term = []
        
    
    def update_covariance(self, feature_pairs, modalities):
        device = next(self.parameters()).device
        modalityA, modalityB = modalities
        n_bar_new = self.n_bar + feature_pairs[:,0].shape[0]
        
        feature_covariance = (feature_pairs[:,0].T @ feature_pairs[:,0]).to(device) / n_bar_new
        self.feature_covariance[modalityA][0] = self.feature_covariance[modalityA][0].to(device)
        self.feature_covariance[modalityA][0] = self.feature_covariance[modalityA][0] * (self.n_bar / n_bar_new)  + feature_covariance 
        
        feature_covariance2 = self.linears[modalityA].weight @ feature_covariance @ self.linears[modalityA].weight.T
        self.feature_covariance2[modalityB][0] = self.feature_covariance2[modalityB][0].to(device)
        self.feature_covariance2[modalityB][0] = self.feature_covariance2[modalityB][0] * (self.n_bar / n_bar_new) + feature_covariance2
        
        feature_covariance = (feature_pairs[:,1].T @ feature_pairs[:,1]).to(device) / n_bar_new
        self.feature_covariance[modalityB][0] = self.feature_covariance[modalityB][0].to(device)
        self.feature_covariance[modalityB][0] = self.feature_covariance[modalityB][0]  * (self.n_bar / n_bar_new) + feature_covariance 
        
        feature_covariance2 = self.linears[modalityB].weight @ feature_covariance @ self.linears[modalityB].weight.T
        self.feature_covariance2[modalityA][0] = self.feature_covariance2[modalityA][0].to(device)
        self.feature_covariance2[modalityA][0] = self.feature_covariance2[modalityA][0] * (self.n_bar / n_bar_new) + feature_covariance2
        
        self.n_bar = n_bar_new
        
    def pre_step(self, dataset):
        
        self.current_training_modalities = dataset.modalities[:2]
        modalityA, modalityB = self.current_training_modalities
        device = next(self.parameters()).device
        self.weight1 = self.linears[modalityA].weight.detach().clone()
        self.weight2 = self.linears[modalityB].weight.detach().clone()
        
        self.Z_1 = self.feature_covariance[modalityA][0].detach().clone()
        self.Z_1_ = self.feature_covariance2[modalityA][0].detach().clone()
        
        self.Z_2 = self.feature_covariance[modalityB][0].detach().clone()
        self.Z_2_ = self.feature_covariance2[modalityB][0].detach().clone()
        
        if self.feature_covariance[modalityA][1]:
            self.P_1 = get_projector(self.Z_1, lambda_min=self.lambda_min).to(device)
            self.P_1_ = get_projector(self.Z_1_, lambda_min=0).to(device)
        else:
            self.P_1 = torch.zeros_like(self.Z_1).to(device)
            self.P_1_ = torch.zeros_like(self.Z_1_).to(device)
        
        if self.feature_covariance[modalityB][1]:
            self.P_2 = get_projector(self.Z_2, lambda_min=self.lambda_min).to(device)
            self.P_2_ = get_projector(self.Z_2_, lambda_min=0).to(device)
        else:
            self.P_2 = torch.zeros_like(self.Z_2).to(device)
            self.P_2_ = torch.zeros_like(self.Z_2_).to(device)

        
    def post_step(self, dataset):
        modalityA, modalityB = self.current_training_modalities
        self.update_covariance(torch.stack( dataset.features), dataset.modalities)
        self.feature_covariance[modalityA][1] = True
        self.feature_covariance[modalityB][1] = True
        
    def update_gradients(self):
                
        with torch.no_grad():
            
            modalityA, modalityB = self.current_training_modalities
            
            gradient1 = self.linears[modalityA].weight.grad
            gradient2 = self.linears[modalityB].weight.grad
            
            gradient1 -=  self.P_1_ @ gradient1 @ self.P_1  
            gradient2 -=  self.P_2_ @ gradient2 @ self.P_2
            
            self.linears[modalityA].weight.grad = gradient1
            self.linears[modalityB].weight.grad = gradient2
            
        self.prev_loss = torch.tensor(self.curr_loss)

    
    
        
                    
