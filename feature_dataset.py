
import h5py
import torch
from avalanche.benchmarks.utils import AvalancheDataset
from perceptors.base import FeatureStorage
from dataloader import DATASETS
from dataloader import ModalityTypeAbb, TaskType, ModalityType

class FeatureDataset(AvalancheDataset):
    def __init__(self, perceptor, dataset_name, modalities=[0,1], task="train"):
        self.dataset_name_d = dataset_name
        self.feature_storage = FeatureStorage(f"{dataset_name}_{task}_{perceptor}.h5") # adopt h5 file to store/retrieve features
        self.existing_indices = list(self.feature_storage.features.keys())
        # order self.existing_indices
        self.existing_indices.sort()
        dataset_modalities = DATASETS[dataset_name].MODALITIES
        self.modalities = [dataset_modalities[m] for m in modalities]
        self.modalities_indices = modalities
        
        modality_name = "-".join([ModalityTypeAbb[dataset_modalities[m]] for m in modalities])
        self.dataset_name = dataset_name + f"({modality_name})"
        self.tasks = DATASETS[dataset_name].TASKS
        
        self.features = [self.feature_storage.features[str(i)] for i in self.existing_indices]
        del self.feature_storage
        self.class_feature_storage = FeatureStorage(f"{dataset_name}_class_{perceptor}.h5")
        
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

        self.classes_name = list(classes_set)
        self.classes_name.sort()
        self.classes_map = classes_map      
        if self.dataset_name_d in ["nyudv2", "msr_vtt", "vggsound_s", "ave", "caltech101", "ucf101", "esc50", "audioset", "imagenet"]:  
            self.class_matrix()
    
    def class_matrix(self):
        try:
            text_pos =  DATASETS[self.dataset_name_d].MODALITIES.index(ModalityType.TEXT)
        except:
            text_pos = 0
        classes_ = []
        for id in self.existing_indices:
            flag=False
            if id.startswith("tensor("):
                id = id[7:-1]
            for i in range(1,5):
                sp_ = id.split("__")
                sample_id = "__".join(sp_[:i])
                if len(sp_) > i:
                    sample_id += "_" if sp_[i][0] == "_" else ""
                if sample_id in self.classes_map:
                    class_ = self.classes_map[sample_id]
                    classes_.append(self.classes_name.index(class_))
                    flag = True
                    break
            if not flag:
                raise ValueError(f"{self.dataset_name_d} Sample {id} not found in the classes map")
                
        classes_ = torch.tensor(classes_)
        classes_row = classes_.unsqueeze(1)  # Shape: (N, 1)
        classes_col = classes_.unsqueeze(0)  # Shape: (1, N)
        relevance_matrix = torch.eq(classes_row, classes_col).int()
        
        self.relevance_matrix = relevance_matrix
        self.classes_ = classes_
        self.class_features = [self.class_feature_storage.features[str(i)] for i in self.classes_name]
        
    def __len__(self):
        return len(self.existing_indices)
    
    def __getitem__(self, idx):
        features = self.features[idx]
        return self.modalities, self.existing_indices[idx], [features[self.modalities_indices]]

