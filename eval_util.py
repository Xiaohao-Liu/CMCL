
from utils import Meter
from dataloader import DATASETS
from dataloader import ModalityTypeAbb, TaskType, ModalityType
from torch.utils.data import DataLoader
import torch
import json
from tqdm import tqdm
import numpy as np
import random


def compute_recall(scores):            
    assert len(scores.shape) == 2
    _, top_1_indices = torch.topk(scores, 1, dim=1)
    _, top_5_indices = torch.topk(scores, 5, dim=1)
    _, top_10_indices = torch.topk(scores, 10, dim=1)

    N = scores.size(0)
    ground_truth_indices = torch.arange(N).view(N, 1).to(scores.device)
    
    recall1 = torch.any(top_1_indices == ground_truth_indices, dim=1).cpu().float().mean().item()
    recall5 = torch.any(top_5_indices == ground_truth_indices, dim=1).cpu().float().mean().item()
    recall10 = torch.any(top_10_indices == ground_truth_indices, dim=1).cpu().float().mean().item()
    
    return recall1, recall5, recall10

def compute_mAP(similarity_matrix, relevance_matrix):
    N = similarity_matrix.size(0)
    
    sorted_indices = torch.argsort(similarity_matrix, dim=1, descending=True)
    
    sorted_relevance = torch.gather(relevance_matrix, 1, sorted_indices)
    
    cum_rel = torch.cumsum(sorted_relevance, dim=1)
    
    precision = cum_rel / torch.arange(1, N + 1, dtype=torch.float32, device=similarity_matrix.device).unsqueeze(0)
    
    ap = torch.sum(precision * sorted_relevance, dim=1) / torch.sum(sorted_relevance, dim=1)
    
    ap[torch.isnan(ap)] = 0
    
    mAP = torch.mean(ap).cpu().item()
    
    return mAP
    
def data_collator(batch):
    modalities = [feature[0] for feature in batch]
    ids = [feature[1] for feature in batch]
    feature_tensors = torch.stack([feature[2][0].cpu() for feature in batch])
    return {"modalities":modalities, "ids": ids, "feature_pairs": feature_tensors}

def eval(model, datasets, save_dir):
    # Evaluate the model
    for dataset in tqdm(datasets["train"]):
        with torch.no_grad():
            train_loss = Meter(f"train_loss__{dataset.dataset_name}", save_dir=save_dir)
            for inputs in DataLoader(dataset, batch_size=2048, shuffle=False, num_workers=0, collate_fn=data_collator):
                modalities = inputs["modalities"]
                ids = inputs["ids"]
                feature_pairs = inputs["feature_pairs"]
                loss = model.train_loss(modalities, ids, feature_pairs.to(next(model.parameters()).device))
                train_loss.update(loss.cpu().item())
            train_loss.save()
    
    for dataset in tqdm(datasets["test"]):        
        
        tasks = dataset.tasks
        feature_pairs = torch.stack(dataset.features).to(next(model.parameters()).device) # [n, 2, dim]
        A_pos, B_pos = dataset.modalities_indices
        modalityA, modalityB = dataset.modalities
        
        if TaskType.RECALL in tasks:
            featuresA = model.get_features(feature_pairs[:,A_pos], modalityA)
            featuresB = model.get_features(feature_pairs[:,B_pos], modalityB)
            
            recall1_meter = Meter(f"test_recall@1_A->B__{dataset.dataset_name}", save_dir=save_dir)
            recall5_meter = Meter(f"test_recall@5_A->B__{dataset.dataset_name}", save_dir=save_dir)
            recall10_meter = Meter(f"test_recall@10_A->B__{dataset.dataset_name}", save_dir=save_dir)
            
            recall1_meter_inverse = Meter(f"test_recall@1_B->A__{dataset.dataset_name}", save_dir=save_dir)
            recall5_meter_inverse = Meter(f"test_recall@5_B->A__{dataset.dataset_name}", save_dir=save_dir)
            recall10_meter_inverse = Meter(f"test_recall@10_B->A__{dataset.dataset_name}", save_dir=save_dir)
            
            for _ in range(5):
                indices = torch.randperm(featuresA.shape[0])[:1000].to(next(model.parameters()).device)
                scores = featuresA[indices] @ featuresB[indices].T
                recall1, recall5, recall10 = compute_recall(scores) # consider the A->B retrieval
                recall1_meter.update(recall1)   
                recall5_meter.update(recall5)
                recall10_meter.update(recall10)
                
                recall1, recall5, recall10 = compute_recall(scores.T) # consider the B->A retrieval
                recall1_meter_inverse.update(recall1)
                recall5_meter_inverse.update(recall5)
                recall10_meter_inverse.update(recall10)
                
            recall1_meter.save()
            recall5_meter.save()
            recall10_meter.save()
            recall1_meter_inverse.save()
            recall5_meter_inverse.save()
            recall10_meter_inverse.save()
        
        if TaskType.ACC in tasks and ModalityType.TEXT in dataset.modalities:
            acc_meter = Meter(f"test_acc__{dataset.dataset_name}", save_dir=save_dir)
            q_pos = A_pos
            text_m = modalityB
            q_m = modalityA
            
            if modalityA == ModalityType.TEXT:
                q_pos = B_pos
                text_m = modalityA
                q_m = modalityB
            
            class_features = torch.stack(dataset.class_features).to(next(model.parameters()).device)
            class_features = model.get_features(class_features, text_m)
            
            features = model.get_features(feature_pairs[:, q_pos], q_m)
            
            scores = features @ class_features.T
            acc = torch.mean((torch.argmax(scores, dim=1) == torch.tensor(dataset.classes_).to(scores.device)).float())
            
            acc_meter.update(acc.cpu().item())
            acc_meter.save()
        
        
        if TaskType.mAP in tasks:
            
            map_meter = Meter(f"test_mAP__{dataset.dataset_name}", save_dir=save_dir)
            
                        
            featuresA = model.get_features(feature_pairs[:,A_pos], modalityA)
            featuresB = model.get_features(feature_pairs[:,B_pos], modalityB)
            relevance_matrix = dataset.relevance_matrix.to(next(model.parameters()).device)
            
            scores = featuresA @ featuresB.T
            
            map = compute_mAP(scores, relevance_matrix)
            map_meter.update(map)
            map_meter.save()
    
        
def statistics(save_dir, logger, datasets):
    results = {}

    for name in logger.data["list"]: # steps
        log_dir = logger.get_step(name)["data_dir"]
        
        results[name] = {
            "train":{},
            "test":{},
        }
        
        for i in datasets["train"]:
            train_losse_meter = Meter(f"train_loss__{i.dataset_name}", save_dir=log_dir, load=True)
            results[name]["train"][i.dataset_name] = {"train_loss": train_losse_meter.average()}
            
        for i in datasets["test"]:
            results[name]["test"][i.dataset_name] = {}
            for task in DATASETS[i.dataset_name_d].TASKS:
                if task == TaskType.RECALL:
                    recall1_meter = Meter(f"test_recall@1_A->B__{i.dataset_name}", save_dir=log_dir, load=True)
                    recall5_meter = Meter(f"test_recall@5_A->B__{i.dataset_name}", save_dir=log_dir, load=True)
                    recall10_meter = Meter(f"test_recall@10_A->B__{i.dataset_name}", save_dir=log_dir, load=True)
                    
                    recall1_meter_inverse = Meter(f"test_recall@1_B->A__{i.dataset_name}", save_dir=log_dir, load=True)
                    recall5_meter_inverse = Meter(f"test_recall@5_B->A__{i.dataset_name}", save_dir=log_dir, load=True)
                    recall10_meter_inverse = Meter(f"test_recall@10_B->A__{i.dataset_name}", save_dir=log_dir, load=True)
                    
                    results[name]["test"][i.dataset_name]["recallA->B"] = [
                            recall1_meter.average(),
                            recall5_meter.average(),
                            recall10_meter.average()
                        ]
                    results[name]["test"][i.dataset_name]["recallB->A"] = [
                            recall1_meter_inverse.average(),
                            recall5_meter_inverse.average(),
                            recall10_meter_inverse.average()
                        ]
                
                elif task == TaskType.ACC and ModalityType.TEXT in i.modalities:
                    acc_meter = Meter(f"test_acc__{i.dataset_name}", save_dir=log_dir, load=True)
                    results[name]["test"][i.dataset_name]["acc"] = acc_meter.average()
                elif task == TaskType.mAP:
                    map_meter = Meter(f"test_mAP__{i.dataset_name}", save_dir=log_dir, load=True)
                    results[name]["test"][i.dataset_name]["mAP"] =  map_meter.average()
                    
    with open(f"{save_dir}/{logger.name}_results.json", "w") as f:
        json.dump(results, f)
    
    return results
