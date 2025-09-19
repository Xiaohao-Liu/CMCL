import sys
sys.path.append("modal_encoder")
import os
from dataloader import ModalityType
from modal_encoder.model import data, load_model, get_embed_dim, LanguageBind
from .base import FeatureStorage, Preceptor
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from modal_encoder.languagebind.languagebind import to_device, LanguageBindImageTokenizer


modalityMap = lambda x:{
        ModalityType.VISION: "image",
        ModalityType.TACTILE: "image",
        ModalityType.TEXT: "language",
    }.get(x, x)

class LanguageBindPreceptor(Preceptor):
    def __init__(self, dataset="msrvtt", split="train", freeze=True, feature_retrieval=False):
        super(LanguageBindPreceptor, self).__init__(dataset, freeze)
        
        self.n_outputs = get_embed_dim("languagebind")
        
        if feature_retrieval: 
            self.model = lambda x: {key: torch.randn(len(x[key]), self.n_outputs) for key in x} # avoid some errors. please prepara the features before the training, so that such function will not be called
        else:
            if freeze:
                model = load_model("languagebind").to("cuda") # mannuall set to cuda
                self.__dict__["model"] = model # discard the registration of parameters
                for param in self.model.parameters():
                    param.requires_grad = False
            else:
                self.model = load_model("languagebind")
        self.dataset = dataset
        
        
        self.feature_storage = FeatureStorage(f"{dataset}_{split}_languagebind.h5") # adopt h5 file to store/retrieve features
        self.existing_indices = self.feature_storage.indices()
        
        
        # languagebind- specific
        if feature_retrieval:
            clip_types = ["video", "audio", "thermal", "image", "depth"]
            self.modality_config = torch.load(LanguageBind.config_path)
            self.modality_transform = {c: LanguageBind.transform_dict[c](self.modality_config[c]) for c in clip_types}
            self.tokenizer = LanguageBindImageTokenizer.from_pretrained(LanguageBind.image_ckpt_path, cache_dir=os.path.join(LanguageBind.cache_dir, "tokenizer_cache_dir"))
        else:
            self.modality_transform = self.model.modality_transform
            self.tokenizer = self.model.tokenizer
        
        
    def forward(self, x):
        modalityLoader = self.modality_transform
        # x is a minibatch of data
        device = "cuda"
        datas = {m:[] for m in ModalityType.__dict__.values()}
        reterived_indices = []
        generate_indices = [] # (modal, pos)
                
        for idx, id in enumerate(x["id"]):
            is_stored = id in self.existing_indices
            if is_stored:
                reterived_indices.append(id)
                continue
            generate_indices.append(id)
            for modal in x["data"]:
                datas[modal].append(x['data'][modal][idx])
        
        # inputs
        features_m = {}
        for m_type in datas:
            if len(datas[m_type]) == 0:
                continue
            inputs = {}
            if m_type == ModalityType.TEXT:
                inputs[modalityMap(m_type)] = to_device(self.tokenizer(datas[m_type], max_length=77, padding='max_length',
                                             truncation=True, return_tensors='pt'), device)
            else:
                inputs[modalityMap(m_type)] = to_device(modalityLoader[modalityMap(m_type)](datas[m_type]), device)
                
            features_m[m_type] = self.model(inputs)[modalityMap(m_type)]
        
        features_reterived = []
        if len(reterived_indices) > 0:
            features_reterived = self.feature_storage.load_features(reterived_indices)
                
        features = []
        store_indices = []
        store_features = []
        for idx, id in enumerate(generate_indices):
            feat = torch.stack([features_m[m][idx]  for m in features_m], dim=0)
            features.append(feat)
            
            store_indices.append(id)
            store_features.append(feat)
        
        if len(store_indices) > 0: # store new features (update)
            self.feature_storage.save_features(store_features, store_indices)
            self.existing_indices = self.feature_storage.indices()

        if len(generate_indices) == 0:
            return features_reterived
        else:
            features = torch.stack(features)
            if len(features_reterived) >0:
                features = torch.concat([features, features_reterived], dim=0)
                
        # normalize
        features = features / features.norm(dim=-1, keepdim=True)
        
        return features
    
    def update(self, minibatches, unlabeled=None):
        all_x = []
        for x, y in minibatches:
            all_x.extend(x)
        self.forward(all_x)