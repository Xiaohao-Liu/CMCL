import sys
sys.path.append("modal_encoder")
import os
from dataloader import ModalityType
from modal_encoder.model import data, load_model, get_embed_dim
from .base import FeatureStorage, Preceptor
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from modal_encoder.unibind.utils.data_transform import load_and_transform_vision_data, load_and_transform_text, load_and_transform_audio_data, load_and_transform_thermal_data, load_and_transform_point_data, load_and_transform_video_data

def load_and_transform_depth_data(depth_paths, device):
    if depth_paths is None:
        return None
    device = torch.device(device)

    depth_outputs = []
    for depth_path in depth_paths:
        data_transform = transforms.Compose(
            [
                transforms.Resize(
                    224, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ]
        )
        with open(depth_path, "rb") as fopen:
            image = Image.open(fopen).convert("L")

        
        image = np.array(image, dtype=np.float32) / 255.0
        disparity = Image.fromarray(image)

        disparity = data_transform(disparity).to(device)

        depth_outputs.append(disparity)

    return torch.stack(depth_outputs, dim=0)


modalityLoader = lambda x:{
        ModalityType.TEXT: load_and_transform_text,
        ModalityType.VISION: load_and_transform_vision_data,
        ModalityType.AUDIO: load_and_transform_audio_data,
        ModalityType.VIDEO: load_and_transform_video_data,
        ModalityType.DEPTH: load_and_transform_depth_data,
        ModalityType.THERMAL: load_and_transform_thermal_data,
        ModalityType.POINT: load_and_transform_point_data,
    }.get(x, data.load_and_transform_vision_data)

modalityMap = lambda x:{
       ModalityType.VIDEO: ModalityType.VISION,
       ModalityType.TACTILE: ModalityType.VISION,
    }.get(x, x)

class UniBindPreceptor(Preceptor):
    def __init__(self, dataset="msrvtt", split="train", freeze=True, feature_retrieval=False):
        super(UniBindPreceptor, self).__init__(dataset, freeze)
        
        self.n_outputs = get_embed_dim("unibind")
        
        if feature_retrieval: 
            self.model = lambda x: {key: torch.randn(len(x[key]), self.n_outputs) for key in x} # avoid some errors. please prepara the features before the training, so that such function will not be called
        else:
            if freeze:
                model = load_model("unibind").to("cuda") # mannuall set to cuda
                self.__dict__["model"] = model # discard the registration of parameters
                for param in self.model.parameters():
                    param.requires_grad = False
            else:
                self.model = load_model("unibind")
        self.dataset = dataset
        
        
        self.feature_storage = FeatureStorage(f"{dataset}_{split}_unibind.h5") # adopt h5 file to store/retrieve features
        self.existing_indices = self.feature_storage.indices()
                
        
    def forward(self, x):
        # x is a minibatch of data
        device = "cuda"
        datas = {m:[] for m in ModalityType.__dict__.values()}
        reterived_indices = []
        embed_pos = [] # (modal, pos)
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
            inputs[modalityMap(m_type)] = modalityLoader(m_type)(datas[m_type], device)
                
            features_m[m_type] = self.model(inputs)[modalityMap(m_type)]
        
        features_reterived = []
        if len(reterived_indices) > 0:
            features_reterived = self.feature_storage.load_features(reterived_indices)
        # reorganize the output
        
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