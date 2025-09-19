import torch

import torch.nn as nn
import torch.optim as optim
from dataloader import ModalityType

class Base(nn.Module):
    def __init__(self, embed_dim, n_step):
        super(Base, self).__init__()
        self.linears = nn.ModuleDict({modality: nn.Linear(embed_dim, embed_dim) for modality in vars(ModalityType).values()})
        self.init()
        self.temperture = 1
        self.current_step = 0
        self.n_step = n_step
        self.prev_loss = torch.tensor(0)
        self.curr_loss = torch.tensor(0)
        
    def init(self):
        for linear in self.linears.values():
            nn.init.eye_(linear.weight)
            nn.init.zeros_(linear.bias)
    
    def forward(self, modalities, ids, feature_pairs):
        modalityA, modalityB = modalities[0] # assert that the modalities are the same in a batch
        
        projected_featureA = self.linears[modalityA](feature_pairs[:,0])
        projected_featureB = self.linears[modalityB](feature_pairs[:,1])
        
        return torch.sum(projected_featureA * projected_featureB, dim=1)
    

    def get_features(self, feature, modality):
        return self.linears[modality](feature)
    
    def next_step(self):
        self.current_step += 1

    def train_loss(self, modalities, ids, feature_pairs):
        modalityA, modalityB = modalities[0] # assert that the modalities are the same in a batch
                
        projected_featureA = self.linears[modalityA](feature_pairs[:,0])
        projected_featureB = self.linears[modalityB](feature_pairs[:,1])
        
        logits = torch.div(torch.matmul(projected_featureA, projected_featureB.T), self.temperture)
        loss = nn.CrossEntropyLoss()(logits, torch.arange(projected_featureA.shape[0]).to(logits.device))
        self.curr_loss = loss.detach().item()
        return loss
    
    def compute_loss(self, modalities, ids, feature_pairs):
        return self.train_loss(modalities, ids, feature_pairs)
    
    def pre_step(self, dataset):
        pass
    
    def post_step(self, dataset):
        pass
    def update_gradients(self):
        pass

def get_projector(A, lambda_min=1e-1): 
    assert A.shape[0] == A.shape[1]
    U, S, V = torch.svd(A)
    k = torch.sum(S > lambda_min).item()
    U = U[:, :k]
    
    return U @ U.T