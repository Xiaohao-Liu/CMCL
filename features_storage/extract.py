import os
import torch
from torch.utils.data import DataLoader
from perceptors import ImagebindPreceptor, LanguageBindPreceptor, UniBindPreceptor
from dataloader import DATASETS
import h5py
import json
from tqdm import tqdm
from dataloader import ModeType

from perceptors import PERCEPTORS



def extract_features(dataset_class, preceptor_class, dataset_root, mode):
    dataset = dataset_class(root=dataset_root, mode=mode)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)

    preceptor = preceptor_class(dataset=dataset_class.__name__.lower(), split=mode, freeze=True, feature_retrieval=False)
    preceptor.eval()

    features = []
    indices = []

    with torch.no_grad():
        for data in tqdm(dataloader):
            outputs = preceptor(data[0])

def build_tasks(gpus):
    datasets = {
        "train": [],
        "test": [],
    }
    remaining_tasks = []
    for dataclass in DATASETS:
        dataset_class = DATASETS[dataclass]
        modalities = dataset_class.MODALITIES
        tasks = dataset_class.TASKS
        modes = dataset_class.MODES
        for mode in ["train", "test"]:
            if mode in modes:
                datasets["train"].append(dataclass)
                for precetorclass in PERCEPTORS:
                    preceptor_class = PERCEPTORS[precetorclass]
                    
                    if os.path.exists(f"features_storage/{dataclass}_{mode}_{precetorclass}.h5"):
                        with h5py.File(f"features_storage/{dataclass}_{mode}_{precetorclass}.h5", 'r', swmr=True) as f:
                            for key in f.keys():
                                feat = torch.tensor(f[key][:])
                                break
                        if feat.shape[0] < len(modalities):
                            comfirm = input(f"Dataset {dataclass} {mode} {precetorclass} has {feat.shape[0]} modalities (expected {len(modalities)}), do you want to recompute? The file will be deleted. (y/n)")
                            if comfirm == "y":
                                os.remove(f"features_storage/{dataclass}_{mode}_{precetorclass}.h5")
                            remaining_tasks.append([dataclass, mode, precetorclass])
                        continue
                    else:
                        remaining_tasks.append([dataclass, mode, precetorclass])
    
    for idx, t in enumerate(remaining_tasks):
        gpu = gpus[idx % len(gpus)]
        print(
            f"CUDA_VISIBLE_DEVICES={gpu} python3 -m features_storage.extract --task=run --mode={t[1]} --dataclass={t[0]} --precetorclass={t[2]}"
        )


def run(dataset_root="dataset",
         mode="train",
         dataclass="nyudv2",
         precetorclass="imagebind"):    

    if dataclass in DATASETS:
        dataset_class = DATASETS[dataclass]
    else:
        raise ValueError(f"Dataset {dataclass} not supported")

    if precetorclass in PERCEPTORS:
        preceptor_class = PERCEPTORS[precetorclass]
    else:
        raise ValueError(f"Preceptor {precetorclass} not supported")

    extract_features(dataset_class, preceptor_class, dataset_root, mode)

def main(task="build", 
         dataset_root="dataset",
         mode="train",
         dataclass="nyudv2",
         precetorclass="imagebind",
         gpus=["0","1"]):   
    
    if task == "run":
        run(dataset_root, mode, dataclass, precetorclass)
    elif task == "build":
        build_tasks(gpus)


if __name__ == "__main__":
    from fire import Fire
    Fire(main)
    
