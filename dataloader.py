import os
import torch
import json
from multiprocessing import Pool
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset


from types import SimpleNamespace

def extract_audio(param):
    video_path, audio_path = param
    if os.path.exists(audio_path):
        return True
    extract_audio_cmd = f"ffmpeg -i {video_path} -ab 160k -ac 2 -ar 44100 -vn {audio_path}  > /dev/null 2>&1"
    status = os.system(extract_audio_cmd)
    return status == 0

# follow imagebind
ModalityType = SimpleNamespace(
    VISION="vision", # "image" for languagebind, 
    TEXT="text", # # "language" for languagebind, 
    AUDIO="audio",
    THERMAL="thermal",
    DEPTH="depth",
    IMU="imu",
    VIDEO="video",
    TACTILE="tactile", # for tvl
    POINT="point", # for unibind
)

TaskType = SimpleNamespace(
    RECALL="Recall",
    ACC="Acc",
    mAP="mAP",
)
 
ModeType = SimpleNamespace(
    TRAIN="train",
    TEST="test",
)

ModalityTypeAbb = {
    ModalityType.VISION: "V",
    ModalityType.TEXT: "T",
    ModalityType.AUDIO: "A",
    ModalityType.THERMAL: "TH",
    ModalityType.DEPTH: "D",
    ModalityType.IMU: "IM",
    ModalityType.VIDEO: "VI",
    ModalityType.TACTILE: "TA",
    ModalityType.POINT: "P"
}

def reorder(modalities):
    return sorted(modalities, key=lambda x: list(ModalityType.__dict__.values()).index(x))

class MultiModalPairedDataLoaderBase(Dataset):
    def __init__(self, root="./dataset", mode="train", fewshot=-1, classes = {}):
        self.root = root
        self.data = []
        self.mode = mode
        self.fewshot = fewshot
        self.modalities_path = os.path.join(self.path, f"paired_data_{mode}.json")
        self.classes = classes

    def __len__(self):
        if self.fewshot == -1:
            return len(self.data)
        return self.fewshot
    
    def get_classes(self):
        classes_name = set()
        classes_map = {}
        for i in self.data:
            classes_map[str(i[0]["id"])] = i[0]["name"]
            classes_name.add(i[0]["name"])
            
        return classes_name, classes_map
            
    
    def statistics(self):
        classes_num = len(self.get_classes()[0])
        only_recall = self.TASKS == [TaskType.RECALL]
        return {
            "Dataset": self.__class__.__name__.replace("-", ""),
            "Modalities": self.MODALITIES,
            "Tasks": self.TASKS,
            "Modes": self.mode,
            "#Examples": self.__len__(),
            "#Classes": classes_num if classes_num > 1 and not only_recall else "-",
        }
            
    def __getitem__(self, idx):
        return self.data[idx]

    def load_data(self, path):
        raise NotImplementedError("Subclasses should implement this method.")    

class NYUDv2(MultiModalPairedDataLoaderBase):
    N_WORKERS = 8
    MODALITIES = reorder([ModalityType.DEPTH, ModalityType.VISION, ModalityType.TEXT])
    TASKS = [TaskType.RECALL]
    MODES = [ModeType.TRAIN, ModeType.TEST]
    def __init__(self, root="./dataset", mode="train", fewshot=-1, classes = {}):
        self.path = os.path.join(root, "nyu-d-2/raw_data")
        super().__init__(root, mode, fewshot)
        self.input_shape = (3, 48,48) # no sense
        
        if os.path.exists(self.modalities_path):
            with open(self.modalities_path, "r") as f:
                self.data = json.load(f)
        else:
            self.load_data()
            with open(self.modalities_path, "w") as f:
                json.dump(self.data, f)
            
    def load_data(self):
        indices = set()
        mode = {"train": "train", "test": "val"}[self.mode]
        path = os.path.join(self.path, f"{mode}")
        for folder in os.listdir(path):
            path2 = os.path.join(path, folder)
            label = folder.split("_")[0]
            if label not in self.classes:
                self.classes[label] = len(self.classes)
            
            for i in os.listdir(path2):
                path3 = os.path.join(path2, i)
                id = folder + "_" + i
                indice = ModalityType.DEPTH + "_" + str(id)
                if indice not in indices:
                    indices.add(indice)
                
                self.data.append(({
                    "id": str(id),
                    "data": 
                        {
                            ModalityType.DEPTH: os.path.join(path3, "depth.png"),
                            ModalityType.VISION:  os.path.join(path3, "rgb.png"),
                            ModalityType.TEXT: label,
                        }
                    ,
                    "name": label,
                    "class": self.classes[label]
                },))
           
class VGGSound(MultiModalPairedDataLoaderBase):
    MODALITIES = reorder([ModalityType.VIDEO, ModalityType.AUDIO, ModalityType.TEXT])
    TASKS = [TaskType.RECALL, TaskType.ACC]
    MODES = [ModeType.TRAIN, ModeType.TEST]
    def __init__(self, root="./dataset", mode="train", fewshot=-1, classes={}):
        self.path = os.path.join(root, "vggsound/scratch/shared/beegfs/hchen/train_data/VGGSound_final/")
        super().__init__(root, mode, fewshot, classes)
        self.csv_path = os.path.join(root, "vggsound/vggsound.csv")
        reload_data = False
        self.input_shape = (3, 48,48) # no sense 
        
        if os.path.exists(self.modalities_path):
            with open(self.modalities_path, "r") as f:
                self.data = json.load(f)
        else:
            self.load_data()
            with open(self.modalities_path, "w") as f:
                json.dump(self.data, f)
        
        
    def extract_audio_from_video(self):
        
        data_pd = pd.read_csv(self.csv_path, names=["youtube_id","start_sec","label", "split"])
        
        tasks = []
        for i in data_pd.iterrows():
            youtube_id, start_sec, label, split = i[1]
            id = youtube_id + f"_{start_sec:06d}"
            video_path = os.path.join(self.path, "video", id+".mp4")
            audio_path = os.path.join(self.path, "audio", id+".mp3")
            
            tasks.append((video_path, audio_path))
        
        with Pool() as pool:
            pool.map(extract_audio, tasks)
    
    def load_data(self, max_samples=1e10):
        indices = set()
        os.makedirs(os.path.join(self.path, "audio"), exist_ok=True)
        self.extract_audio_from_video()

        data_pd = pd.read_csv(self.csv_path, names=["youtube_id","start_sec","label", "split"])
        count = 0
        mode = {"train": "train", "test": "test"}[self.mode]
        for i in tqdm(data_pd.iterrows(), total=len(data_pd)):
            youtube_id, start_sec, label, split = i[1]
            if mode != split:
                continue
            if not label in self.classes:
                self.classes[label] = len(self.classes)
                
            id = youtube_id + f"_{start_sec:06d}"
            if not os.path.exists(os.path.join(self.path, "video", id+".mp4")):
                continue
            if count > max_samples:
                break
            count += 1
            video_path = os.path.join(self.path, "video", id+".mp4")
            audio_path = os.path.join(self.path, "audio", id+".mp3")
            if not (os.path.exists(video_path) and os.path.exists(audio_path)):
                continue
            indice = id
            if indice not in indices:
                indices.add(indice)
                self.data.append(
                    ({
                        "id": indice,
                        "data": {
                            ModalityType.VIDEO: video_path,
                            ModalityType.AUDIO: audio_path,
                            ModalityType.TEXT: label,
                        },
                        "name": label,
                        "class": self.classes[label]
                    },)
                )

class VGGSound_S(VGGSound):
    def load_data(self, _=None):
        if self.mode == "train":
            return super().load_data(10000)
        elif self.mode == "test":
            return super().load_data(2000)

class TVL(MultiModalPairedDataLoaderBase):
    MODALITIES = reorder([ModalityType.TACTILE, ModalityType.VISION, ModalityType.TEXT])
    TASKS = [TaskType.RECALL]
    MODES = [ModeType.TRAIN, ModeType.TEST]
    def __init__(self, root="./dataset", mode="train", fewshot=-1, classes={}):
        self.path = os.path.join(root, "./tvl/tvl_dataset")
        super().__init__(root, mode, fewshot, classes)
                
        self.input_shape = (3, 48,48) # no sense
        self.classes = {}
        
        if os.path.exists(self.modalities_path):
            with open(self.modalities_path, "r") as f:
                self.data = json.load(f)
        else:
            self.load_data()
            with open(self.modalities_path, "w") as f:
                json.dump(self.data, f)
        
        
        self.num_classes = len(self.classes) # classes

    
    def load_data(self):
        indices = set()
        
        mode = {"train": "train", "test": "test"}[self.mode]
        def load_(name):
            path = os.path.join(self.path, f"{name}/{mode}.csv")
            data_pd = pd.read_csv(path) # columns: url,tactile,caption
            columns = ["url","caption"] if name == "ssvtp" and mode == "test" else ["url","tactile","caption"]
            data_pd = data_pd[columns]
            for id, i in enumerate(data_pd.iterrows()):
                if name == "ssvtp" and mode == "test":
                    url, caption = i[1]
                    tactile = url.replace("rgb", "tac")
                else:
                    url,tactile,caption = i[1]
                try:
                    caption = caption.replace("\r\n", "").replace(".", "")
                except:
                    continue
                
                for class_ in caption.split(","):
                    class_ = class_.strip()
                    if class_ not in self.classes:
                        self.classes[class_] = len(self.classes)
                label = [self.classes[class_.strip()] for class_ in caption.split(",")]
                
                indice = ModalityType.VISION + "_" + name.replace("/", "_") + "_" + str(id)
                if indice not in indices:
                    indices.add(indice)
                    self.data.append(
                        ({
                            "id": indice,
                            "data": {
                                ModalityType.VISION: os.path.join(self.path, f"{name}/{url}"),
                                ModalityType.TACTILE: os.path.join(self.path, f"{name}/{tactile}"),
                                ModalityType.TEXT: caption,
                            },
                            "name": caption,
                            "class": ",".join([str(i) for i in label])
                        },)
                    )

        load_("ssvtp")
        load_("hct/data1")
        load_("hct/data2")
        load_("hct/data3")
  

class AudioCaps(MultiModalPairedDataLoaderBase):
    N_WORKERS = 8
    MODALITIES = reorder([ModalityType.AUDIO, ModalityType.TEXT])
    TASKS = [TaskType.RECALL]
    MODES = [ModeType.TEST]
    def __init__(self, root="./dataset", mode="test", fewshot=-1, classes = {}):
        self.path = os.path.join(root, "audiocaps_test")
        super().__init__(root, mode, fewshot)
        self.input_shape = (3, 48,48) # no sense
        
        
        self.processed_folder = os.path.join(self.path, "processed")
        
        
        if os.path.exists(self.modalities_path):
            with open(self.modalities_path, "r") as f:
                self.data = json.load(f)
        else:
            self.load_data()
            with open(self.modalities_path, "w") as f:
                json.dump(self.data, f)
            
    def load_data(self):
        indices = set()
        mode = {"test": "test"}[self.mode]
        
        if not os.path.exists(self.processed_folder):
            # only test
            os.makedirs(os.path.join(self.processed_folder, "test/audio"), exist_ok=True)
            
        for i in os.listdir(os.path.join(self.path, "data")):
            if i.endswith(".parquet"):
                df = pd.read_parquet(os.path.join(self.path, "data", i))
                for _, row in df.iterrows():
                    indice = len(indices)
                    audio_path = os.path.join(self.processed_folder, "test/audio", str(indice) + ".mp3")
                    
                                        
                    description = row['answer']
                    label = "no label"
                    if label not in self.classes:
                        self.classes[label] = len(self.classes)
                    # write audio in to 'test/audio'
                    audio_bytes = row["context"]['bytes']
                    with open(audio_path, "wb") as f:
                        f.write(audio_bytes)
                    
                    if indice not in indices:
                        indices.add(indice)
                        self.data.append(
                            ({
                                "id": indice,
                                "data": {
                                    ModalityType.AUDIO: audio_path,
                                    ModalityType.TEXT: description,
                                },
                                "name": label,
                                "class": self.classes[label]
                            },)
                        )
                   
class AudioSet(MultiModalPairedDataLoaderBase):
    N_WORKERS = 8
    MODALITIES = reorder([ModalityType.AUDIO, ModalityType.TEXT])
    TASKS = [TaskType.mAP]
    MODES = [ModeType.TEST]
    def __init__(self, root="./dataset", mode="test", fewshot=-1, classes = {}):
        self.path = os.path.join(root, "audioset_test")
        super().__init__(root, mode, fewshot)
        self.input_shape = (3, 48,48) # no sense
        
        
        self.processed_folder = os.path.join(self.path, "processed")
        
        
        if os.path.exists(self.modalities_path):
            with open(self.modalities_path, "r") as f:
                self.data = json.load(f)
        else:
            self.load_data()
            with open(self.modalities_path, "w") as f:
                json.dump(self.data, f)
            
    def load_data(self):
        indices = set()
        mode = {"test": "test"}[self.mode]
        
        if not os.path.exists(self.processed_folder):
            # only test
            os.makedirs(os.path.join(self.processed_folder, "test/audio"), exist_ok=True)
            
        for i in os.listdir(os.path.join(self.path, "data/test")):
            if i.endswith(".parquet"):
                df = pd.read_parquet(os.path.join(self.path, "data/test", i))
                for _, row in df.iterrows():
                    indice = row["index"].split("test/")[-1]
                    audio_path = os.path.join(self.processed_folder, "test/audio", indice + ".mp3")
                    
                                        
                    description = row['raw_text'][1].split(": ")[-1][1:-1].replace("'", "").lower()
                    label = description.split(",")[0]
                    if label not in self.classes:
                        self.classes[label] = len(self.classes)
                    # write audio in to 'test/audio'
                    audio_bytes = row["audio"]['bytes']
                    with open(audio_path, "wb") as f:
                        f.write(audio_bytes)
                    
                    if indice not in indices:
                        indices.add(indice)
                        self.data.append(
                            ({
                                "id": indice,
                                "data": {
                                    ModalityType.AUDIO: audio_path,
                                    ModalityType.TEXT: description,
                                },
                                "name": label,
                                "class": self.classes[label]
                            },)
                        )

class Clotho(MultiModalPairedDataLoaderBase):
    N_WORKERS = 8
    MODALITIES = reorder([ModalityType.AUDIO, ModalityType.TEXT])
    TASKS = [TaskType.RECALL]
    MODES = [ModeType.TRAIN, ModeType.TEST]
    def __init__(self, root="./dataset", mode="train", fewshot=-1, classes = {}):
        self.path = os.path.join(root, "clotho")
        super().__init__(root, mode, fewshot)
        self.input_shape = (3, 48,48) # no sense
        
        if os.path.exists(self.modalities_path):
            with open(self.modalities_path, "r") as f:
                self.data = json.load(f)
        else:
            self.load_data()
            with open(self.modalities_path, "w") as f:
                json.dump(self.data, f)
            
    def load_data(self):
        indices = set()
        mode = {"train": "development", "test": "evaluation"}[self.mode]
        path = os.path.join(self.path, f"clotho_captions_{mode}.csv")
        
        df = pd.read_csv(path)
        
        for _, row in df.iterrows():
            file_name = row["file_name"]
            audio_path = os.path.join(self.path, mode, file_name)
            text = row["caption_1"]
            indice = file_name.split(".")[0]
            label = "no label"
            if label not in self.classes:
                self.classes[label] = len(self.classes)
            
            if indice not in indices:
                indices.add(indice)
                self.data.append(
                    ({
                        "id": indice,
                        "data": {
                            ModalityType.AUDIO: audio_path,
                            ModalityType.TEXT: text,
                        },
                        "name": label,
                        "class": self.classes[label]
                    },)
                )

class COCO(MultiModalPairedDataLoaderBase):
    N_WORKERS = 8
    MODALITIES = reorder([ModalityType.VISION, ModalityType.TEXT])
    TASKS = [TaskType.RECALL]
    MODES = [ModeType.TEST]
    def __init__(self, root="./dataset", mode="test", fewshot=-1, classes = {}):
        self.path = os.path.join(root, "coco_test")
        super().__init__(root, mode, fewshot)
        self.input_shape = (3, 48,48) # no sense        
        
        if os.path.exists(self.modalities_path):
            with open(self.modalities_path, "r") as f:
                self.data = json.load(f)
        else:
            self.load_data()
            with open(self.modalities_path, "w") as f:
                json.dump(self.data, f)
            
    def load_data(self):
        indices = set()
        mode = {"test": "test"}[self.mode]
        
        path = os.path.join(self.path, "test_5k_mscoco_2014.csv")
        
        df = pd.read_csv(path)
        
        for _, row in df.iterrows():
            image_path = os.path.join(self.path, "images_mscoco_2014_5k_test", row["filename"])
            text = " ".join(eval(row["tokens"])[0])
            
            label = "no label"
            if label not in self.classes:
                self.classes[label] = len(self.classes)
            
            indice = row["cocoid"]
            if indice not in indices:
                indices.add(indice)
                self.data.append(
                    ({
                        "id": indice,
                        "data": {
                            ModalityType.VISION: image_path,
                            ModalityType.TEXT: text,
                        },
                        "name": label,
                        "class": self.classes[label]
                    },)
                )

class ESC50(MultiModalPairedDataLoaderBase):
    N_WORKERS = 8
    MODALITIES = reorder([ModalityType.AUDIO, ModalityType.TEXT])
    TASKS = [TaskType.RECALL, TaskType.ACC]
    MODES = [ModeType.TRAIN, ModeType.TEST]
    def __init__(self, root="./dataset", mode="train", fewshot=-1, classes = {}):
        self.path = os.path.join(root, "esc50")
        super().__init__(root, mode, fewshot)
        self.input_shape = (3, 48,48) # no sense   
        
        self.processed_folder = os.path.join(self.path, "processed")     
        
        if os.path.exists(self.modalities_path):
            with open(self.modalities_path, "r") as f:
                self.data = json.load(f)
        else:
            self.load_data()
            with open(self.modalities_path, "w") as f:
                json.dump(self.data, f)
            
    def load_data(self):
        indices = set()
        mode = {"train": "train", "test": "test"}[self.mode]
        # train: 1600
        # test: 400
        
        path1 = os.path.join(self.path, "data/train-00000-of-00002-2f1ab7b824ec751f.parquet")
        path2 = os.path.join(self.path, "data/train-00001-of-00002-27425e5c1846b494.parquet")
        
        if not os.path.exists(os.path.join(self.processed_folder, f"{mode}/audio")):
            os.makedirs(os.path.join(self.processed_folder, f"{mode}/audio"), exist_ok=True)            
        
        def read_row(row):
            audio_path = os.path.join(self.processed_folder, f"{mode}/audio", row["filename"])
            audio_bytes = row["audio"]['bytes']
            with open(audio_path, "wb") as f:
                f.write(audio_bytes)
                    
            text = row["category"]
            indice = row["filename"]
            label = text
            if label not in self.classes:
                self.classes[label] = len(self.classes)
            
            if indice not in indices:
                indices.add(indice)
                self.data.append(
                    ({
                        "id": indice,
                        "data": {
                            ModalityType.AUDIO: audio_path,
                            ModalityType.TEXT: text,
                        },
                        "name": label,
                        "class": self.classes[label]
                    },)
                )
                
        if mode == "test":
            df = pd.read_parquet(path2)
            for _, row in df.iterrows():
                if _ < 600:
                    continue
                read_row(row)
        
        elif mode == "train":
            df = pd.read_parquet(path1)
            for _, row in df.iterrows():
                read_row(row)
            
            df = pd.read_parquet(path2)
            for _, row in df.iterrows():
                if _ < 600:
                    read_row(row)

class ImageNet(MultiModalPairedDataLoaderBase):
    N_WORKERS = 8
    MODALITIES = reorder([ModalityType.VISION, ModalityType.TEXT])
    TASKS = [TaskType.ACC]
    MODES = [ModeType.TEST]
    def __init__(self, root="./dataset", mode="test", fewshot=-1, classes = {}):
        self.path = os.path.join(root, "imagenet1k_test")
        super().__init__(root, mode, fewshot)
        self.input_shape = (3, 48,48) # no sense
        
        
        self.processed_folder = os.path.join(self.path, "processed")
        
        
        if os.path.exists(self.modalities_path):
            with open(self.modalities_path, "r") as f:
                self.data = json.load(f)
        else:
            self.load_data()
            with open(self.modalities_path, "w") as f:
                json.dump(self.data, f)
            
    def load_data(self):
        indices = set()
        mode = {"test": "test"}[self.mode]
        
        if not os.path.exists(self.processed_folder):
            # only test
            os.makedirs(os.path.join(self.processed_folder, "test/image"), exist_ok=True)
        
        labels_name = json.load(open(os.path.join(self.path, "label2text.json"), "r"))
        
        for i in os.listdir(os.path.join(self.path, "data")):
            if i.endswith(".parquet"):
                df = pd.read_parquet(os.path.join(self.path, "data", i))
                for _, row in df.iterrows():
                    indice = len(indices)
                    image_path = os.path.join(self.processed_folder, "test/image", str(indice) + ".jpg")                    
                                        
                    description = labels_name[int(row["label"])]
                    label = description
                    if label not in self.classes:
                        self.classes[label] = len(self.classes)
                    # write image to 'test/image'
                    imagebytes = row["image"]['bytes']
                    with open(image_path, "wb") as f:
                        f.write(imagebytes)
                    
                    if indice not in indices:
                        indices.add(indice)
                        self.data.append(
                            ({
                                "id": indice,
                                "data": {
                                    ModalityType.VISION: image_path,
                                    ModalityType.TEXT: description,
                                },
                                "name": label,
                                "class": self.classes[label]
                            },)
                        )

class LLVIP(MultiModalPairedDataLoaderBase):
    N_WORKERS = 8
    MODALITIES = reorder([ModalityType.VISION, ModalityType.THERMAL])
    TASKS = [TaskType.RECALL]
    MODES = [ModeType.TRAIN, ModeType.TEST]
    def __init__(self, root="./dataset", mode="train", fewshot=-1, classes = {}):
        self.path = os.path.join(root, "llvip")
        super().__init__(root, mode, fewshot)
        self.input_shape = (3, 48,48) # no sense
        
        if os.path.exists(self.modalities_path):
            with open(self.modalities_path, "r") as f:
                self.data = json.load(f)
        else:
            self.load_data()
            with open(self.modalities_path, "w") as f:
                json.dump(self.data, f)
            
    def load_data(self):
        indices = set()
        mode = {"train": "train", "test": "test"}[self.mode]
        
        for i in os.listdir(os.path.join(self.path, "LLVIP/visible", mode)):
            if i.endswith("jpg"):
                image_path = os.path.join(self.path, "LLVIP/visible", mode, i)
                label = "no label"
                if label not in self.classes:
                    self.classes[label] = len(self.classes)
                
                indice = i.split(".")[0]
                if indice not in indices:
                    indices.add(indice)
                    self.data.append(
                        ({
                            "id": indice,
                            "data": {
                                ModalityType.VISION: image_path,
                                ModalityType.THERMAL: os.path.join(self.path, "LLVIP/infrared", mode, i),
                            },
                            "name": label,
                            "class": self.classes[label]
                        },)
                    )

class MOSEI(MultiModalPairedDataLoaderBase):
    N_WORKERS = 8
    MODALITIES = reorder([ModalityType.VISION, ModalityType.TEXT])
    TASKS = [TaskType.ACC]
    MODES = [ModeType.TRAIN, ModeType.TEST]
    def __init__(self, root="./dataset", mode="train", fewshot=-1, classes = {}):
        self.path = os.path.join(root, "mosei")
        super().__init__(root, mode, fewshot)
        self.input_shape = (3, 48,48) # no sense
        
        self.processed_folder = os.path.join(self.path, "processed")
        
        
        if os.path.exists(self.modalities_path):
            with open(self.modalities_path, "r") as f:
                self.data = json.load(f)
        else:
            self.load_data()
            with open(self.modalities_path, "w") as f:
                json.dump(self.data, f)
            
    def load_data(self):
        indices = set()
        mode = {"train":"train", "test": "test"}[self.mode]
        
        if not os.path.exists(self.processed_folder):
            os.makedirs(os.path.join(self.processed_folder, "test/image"), exist_ok=True)
            os.makedirs(os.path.join(self.processed_folder, "train/image"), exist_ok=True)
        
        labels_name = json.load(open(os.path.join(self.path, "label2text.json"), "r"))
        
        df = pd.read_parquet(os.path.join(self.path, "data", f"{mode}-00000-of-00001.parquet"))
        for _, row in df.iterrows():
            indice = len(indices)
            image_path = os.path.join(self.processed_folder, "test/image", str(indice) + ".jpg")                    
                                
            description = labels_name[int(row["label"])]
            label = description
            if label not in self.classes:
                self.classes[label] = len(self.classes)
            imagebytes = row["image"]['bytes']
            with open(image_path, "wb") as f:
                f.write(imagebytes)
            
            if indice not in indices:
                indices.add(indice)
                self.data.append(
                    ({
                        "id": indice,
                        "data": {
                            ModalityType.VISION: image_path,
                            ModalityType.TEXT: description,
                        },
                        "name": label,
                        "class": self.classes[label]
                    },)
                )

class UCF101(MultiModalPairedDataLoaderBase):
    N_WORKERS = 8
    MODALITIES = reorder([ModalityType.VIDEO, ModalityType.TEXT])
    TASKS = [TaskType.ACC]
    MODES = [ModeType.TRAIN, ModeType.TEST]
    def __init__(self, root="./dataset", mode="train", fewshot=-1, classes = {}):
        self.path = os.path.join(root, "ucf101")
        super().__init__(root, mode, fewshot)
        self.input_shape = (3, 48,48) # no sense
        
        if os.path.exists(self.modalities_path):
            with open(self.modalities_path, "r") as f:
                self.data = json.load(f)
        else:
            self.load_data()
            with open(self.modalities_path, "w") as f:
                json.dump(self.data, f)

            
    def load_data(self):
        indices = set()
        modes = {"train": "train", "test": "test+val"}[self.mode]
        
        for mode in modes.split("+"):
            for text in os.listdir(os.path.join(self.path, "UCF101", mode)):
                for i in os.listdir(os.path.join(self.path, "UCF101", mode, text)):
                    if i.endswith(".avi"):
                        video_path = os.path.join(self.path, "UCF101", mode, text, i)
                        label = text
                        if label not in self.classes:
                            self.classes[label] = len(self.classes)
                        
                        desciption = ''.join([' ' + char.lower() if char.isupper() else char for char in text]).strip()
                        indice = text+"_"+i.split(".")[0]
                        
                        audio_path = os.path.join(self.path, "audio", mode, indice +".mp3")
                        if indice not in indices:
                            indices.add(indice)
                            self.data.append(
                                ({
                                    "id": indice,
                                    "data": {
                                        ModalityType.VIDEO: video_path,
                                        # ModalityType.AUDIO: audio_path,
                                        ModalityType.TEXT: desciption,
                                    },
                                    "name": label,
                                    "class": self.classes[label]
                                },)
                            )
        
class FLIR(MultiModalPairedDataLoaderBase):
    N_WORKERS = 8
    MODALITIES = reorder([ModalityType.VISION, ModalityType.THERMAL])
    TASKS = [TaskType.RECALL]
    MODES = [ModeType.TEST]
    def __init__(self, root="./dataset", mode="train", fewshot=-1, classes = {}):
        self.path = os.path.join(root, "flir")
        super().__init__(root, mode, fewshot)
        self.input_shape = (3, 48,48) # no sense
        
        if os.path.exists(self.modalities_path) and False:
            with open(self.modalities_path, "r") as f:
                self.data = json.load(f)
        else:
            self.load_data()
            with open(self.modalities_path, "w") as f:
                json.dump(self.data, f)
    
            
    def load_data(self):
        indices = set()
        mode = {"test": "test"}[self.mode]
        
        rgb2th = None
        with open(os.path.join(self.path, "rgb_to_thermal_vid_map.json"), "r") as f:
            rgb2th = json.load(f)
                        
        for i in os.listdir(os.path.join(self.path, f"video_rgb_test", "data")):
            if i.endswith("jpg"):
                thermal_name = rgb2th[i]
                image_path = os.path.join(self.path, f"video_rgb_test", "data", i)
                thermal_path = os.path.join(self.path, f"video_thermal_test", "data", thermal_name)
                label = "no label"
                if label not in self.classes:
                    self.classes[label] = len(self.classes)
                    
                indice = i.split(".jpg")[0]
                    
                if indice not in indices:
                    indices.add(indice)
                    self.data.append(
                        ({
                            "id": indice,
                            "data": {
                                ModalityType.VISION: image_path,
                                ModalityType.THERMAL: thermal_path,
                            },
                            "name": label,
                            "class": self.classes[label]
                        },)
                    )
    

DATASETS = {
    "ucf101": UCF101, # 2 modalities, acc
    "esc50": ESC50, # 2 modalities, acc
    "nyudv2": NYUDv2, # 3 modalities, recall
    "vggsound_s": VGGSound_S, # 3 modalities, recall & acc
    "clotho": Clotho, # 2 modalities, recall
    "tvl": TVL, # 3 modalities, recall
    "llvip": LLVIP, # 2 modalities, recall
    }

if __name__ == "__main__":
    
    dataset = TVL(root="./dataset", mode="test")
    dataset = TVL(root="./dataset", mode="train")
