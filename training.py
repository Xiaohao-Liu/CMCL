from perceptors import ImagebindPreceptor
from modal_encoder.model import data, load_model, get_embed_dim
from utils import Meter, Logger
import json
import time

from transformers import Trainer, TrainingArguments, TrainerCallback

from feature_dataset import FeatureDataset
from dataloader import ModalityType, DATASETS, TaskType, ModeType
import os
from tqdm import tqdm

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from tabulate import tabulate
from eval_util import data_collator, eval, statistics

class Projector(nn.Module):
    def __init__(self, embed_dim):
        super(Projector, self).__init__()
        self.linears = nn.ModuleDict({modality: nn.Linear(embed_dim, embed_dim ) for modality in vars(ModalityType).values()})
        for linear in self.linears.values():
            nn.init.eye_(linear.weight)
    
    def forward(self, modalities, ids, feature_pairs):
        modalityA, modalityB = modalities[0] # assert that the modalities are the same in a batch

        projected_featureA = self.linears[modalityA](feature_pairs[:,0])
        projected_featureB = self.linears[modalityB](feature_pairs[:,1])
        
        return torch.sum(projected_featureA * projected_featureB, dim=1)
    
    def compute_loss(self, modalities, ids, feature_pairs):
        modalityA, modalityB = modalities[0] # assert that the modalities are the same in a batch
        
        projected_featureA = self.linears[modalityA](feature_pairs[:,0])
        projected_featureB = self.linears[modalityB](feature_pairs[:,1])
        
        logits = projected_featureA @ projected_featureB.T
        loss = nn.CrossEntropyLoss()(logits, torch.arange(projected_featureA.shape[0]).to(logits.device))
        
        return loss
        

class Trainer2(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        modalities = inputs["modalities"]
        ids = inputs["ids"]
        feature_pairs = inputs["feature_pairs"]
        
        # Forward pass
        loss = model.compute_loss(modalities, ids, feature_pairs.to(next(model.parameters()).device))
        
        return (loss, None) if return_outputs else loss
    
class TrainerCallback2(TrainerCallback):
    def on_pre_optimizer_step(self, args, state, control, model, **kwargs):
        model.update_gradients()
        
from torch.optim import AdamW

def main(
    disable_wandb: bool = True,
    perceptor = "imagebind",
    method = "DNS",
    weight_decay = 1e-3,
    learning_rate = 1e-4,
    batch_size = 64,
    lambda_min = 1e-1,
    seed=42,
    datasets_name = "~".join([
        "ucf101",
        "esc50",
        "nyudv2:0,2",
        "clotho",
        "vggsound_s:0,1", "vggsound_s:0,2", "vggsound_s:1,2", 
        "tvl:0,2", "tvl:1,2", "tvl:0,1",
        "llvip"
    ]),
    log_name = "full_train_test",
    only_statistics = False,
):
    datasets_name = datasets_name.split("~")
    datasets = {"train":[],"test":[]}
    
    
    if perceptor == "languagebind":
        learning_rate = 1e-3
    
    os.environ['WANDB_API_KEY'] = "YOUR_API_KEY"
    os.environ['WANDB_PROJECT'] = "CMCL"
    os.environ["WANDB_DISABLED"] = "true" if disable_wandb else "false"

    for d in tqdm(datasets_name):
        # if you want to assign modalities for trianing
        if ":" in d:
            modalities = [int(i) for i in d.split(":")[1].split(",")]
            d = d.split(":")[0]
        else:
            modalities = [0,1]
        if ModeType.TRAIN in DATASETS[d].MODES:
            datasets["train"].append(FeatureDataset(perceptor, d, modalities = modalities, task="train"))
        if ModeType.TEST in DATASETS[d].MODES:
            datasets["test"].append(FeatureDataset(perceptor, d, modalities = modalities,  task="test"))    
    
    from cmcl import DNS

    if method == "DNS":
        model = DNS(get_embed_dim(perceptor), n_step=len(datasets["train"]), lambda_min=lambda_min)
    else:
        raise ValueError(f"Method {method} not found")
        
    model = model.cuda()

    logger = Logger(
        perceptor,
        method,
        datasets_name,
        name = log_name,
        save_dir=f"./results/{perceptor}/{method}",
        load = only_statistics
    )
    
    if not only_statistics:
        logger.add_step("S")
        time_record = []
        for step, train_dataset in enumerate(datasets["train"]):
            with torch.no_grad():
                model.eval()
                eval(model, datasets, logger.last_step["data_dir"])
            
            time_record_ = []
            s_time = time.time()
            if method == "individual":
                model.init() # reset the model for each step
            model.train()
            
            logger.add_step(train_dataset.dataset_name)
            
            time_record_.append(train_dataset.dataset_name)
            
            time_record_.append(f"{time.time() - s_time:.5f}")
            print("init:", time_record_[-1])
            
            model.pre_step(train_dataset)
            
            time_record_.append(f"{time.time() - s_time:.5f}")
            print("pre_step:",time_record_[-1])
            
            training_args = TrainingArguments(
                output_dir=os.path.join(logger.last_step["data_dir"], "ckpt"),          # output directory
                num_train_epochs=5,              # total number of training epochs
                per_device_train_batch_size=batch_size, # batch size for training
                per_device_eval_batch_size=batch_size,  # batch size for evaluation
                optim="adamw_torch",
                learning_rate=learning_rate,              # learning rate
                warmup_steps=0,                # number of warmup steps for learning rate scheduler
                weight_decay=weight_decay,               # strength of weight decay
                logging_dir='./logs',             # directory for storing logs
                logging_steps=10, 
                save_total_limit=3, 
                save_strategy="epoch", 
                seed=int(seed),
            )

            # Update the Trainer instance to use the custom data collator
            trainer = Trainer2(
                model=model,                # the instantiated 🤗 Transformers model to be trained
                args=training_args,                  # training arguments, defined above
                train_dataset=train_dataset,         # training dataset
                data_collator=data_collator,         # custom data collator
                callbacks=[TrainerCallback2()] if step > 0 else [],
                optimizers = (None, None),
            )
            
            # Train the model
            trainer.train()
            
            time_record_.append(f"{time.time() - s_time:.5f}")
            print("train:", time_record_[-1])
            
            model.post_step(train_dataset)
            model.next_step()
            
            time_record_.append(f"{time.time() - s_time:.5f}")
            print("post_step:", time_record_[-1])
            
            print(tabulate([time_record_], headers=["dataset", "init", "pre_step", "train", "post_step"]))
            time_record.append(time_record_)
        
        print(tabulate(time_record, headers=["dataset", "init", "pre_step", "train", "post_step"]))
        with torch.no_grad():
            model.eval()
            eval(model, datasets, logger.last_step["data_dir"])
    
    statistics(f"./results/{perceptor}/{method}", logger=logger, datasets=datasets)

if __name__ == "__main__":
    
    from fire import Fire
    Fire(main)
