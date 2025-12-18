import os
import torch
from torch.utils.data import DataLoader
from perceptors import ImagebindPreceptor, LanguageBindPreceptor, UniBindPreceptor
from dataloader import DATASETS
from perceptors.base import FeatureStorage
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

def save_classes(dataset_name, perceptor):
    classes_set = set()
    classes_map = {}
    if "train" in DATASETS[dataset_name].MODES:
        s, m = DATASETS[dataset_name](mode="train").get_classes()
        classes_set = classes_set | s
        classes_map.update(m)
    if "test" in DATASETS[dataset_name].MODES:
        s, m = DATASETS[dataset_name](mode="test").get_classes()
        classes_set = classes_set | s
        classes_map.update(m)
        
    classes_name = list(classes_set)
    classes_name.sort()
    from perceptors.base import FeatureStorage
    
    
    class_feature_storage = FeatureStorage(f"{dataset_name}_class_{perceptor}.h5")
    
    if len(class_feature_storage.features) == len(classes_name):
        return
    
    train_feature_storage = FeatureStorage(f"{dataset_name}_train_{perceptor}.h5")
    test_feature_storage = FeatureStorage(f"{dataset_name}_test_{perceptor}.h5")

    class_set = []
    classes_ = []
    class_features = {}
    try:
        text_pos =  DATASETS[dataset_name].MODALITIES.index(ModalityType.TEXT)
    except:
        text_pos = 0
    for feature_storage in [train_feature_storage, test_feature_storage]:
        for id, features in feature_storage.features.items():
            flag=False
            if id.startswith("tensor("):
                id = id[7:-1]
            for i in range(1,5):
                sp_ = id.split("__")
                sample_id = "__".join(sp_[:i])
                if len(sp_) > i:
                    sample_id += "_" if sp_[i][0] == "_" else ""
                if sample_id in classes_map:
                    class_ = classes_map[sample_id]
                    classes_.append(classes_name.index(class_))
                    if class_ not in class_set:
                        class_set.append(class_) 
                        class_features[class_] = features[text_pos]
                    flag = True
                    break
            if not flag:
                raise ValueError("")
            
    indices = class_features.keys()
    features = [ class_features[i] for i in indices]
    if dataset_name not in ["tvl"]:
        assert len(indices) == len(classes_name), f"{dataset_name} {perceptor} {len(indices)} {len(classes_name)}"
    class_feature_storage.save_features(features, indices)

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
    
"""
# Extract Classes
for d in DATASETS:
    for perceptor in ["imagebind", "languagebind", "unibind"]:
        save_classes(d, perceptor)
"""
