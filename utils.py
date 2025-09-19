import json
import os
import hashlib



class Meter:
    def __init__(self, name, save_dir=None, load=False):
        self.name = name
        self.save_dir = save_dir
        self.reset()
        if load:
            self.load()

    def reset(self):
        self.values = []
        self.count = 0
        self.sum = 0.0

    def update(self, value, n=1):
        self.values.append(value)
        self.count += n
        self.sum += value * n

    def average(self):
        return self.sum / self.count if self.count != 0 else 0.0
    
    def max(self):
        return max(self.values)

    def min(self):
        return min(self.values)

    def get_values(self):
        return self.values
    
    def save(self):
        if self.save_dir is None:
            raise ValueError("Save directory is not specified.")
        
        data = {
            'values': self.values,
            'count': self.count,
            'average': self.average()
        }
        
        os.makedirs(self.save_dir, exist_ok=True)
        file_path = os.path.join(self.save_dir, f"{self.name}.json")
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)

    def load(self):
        if self.save_dir is None:
            raise ValueError("Save directory is not specified.")
        
        file_path = os.path.join(self.save_dir, f"{self.name}.json")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No saved data found at {file_path}")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        self.values = data['values']
        self.count = data['count']
        self.sum = sum(self.values)

import time
class Logger:
    def __init__(self, 
        perceptor,
        method,
        datasets_name,
        name,
        save_dir,
        load=False,
        ):
        
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.perceptor = perceptor
        self.method = method
        self.datasets_name = datasets_name
        self.name = name
        self.start_time = time.time()

        self.data = {
            "steps": {},
            "list": []
        }
        if load:
            self.load()

    def add_step(self, step_name):
        self.data["list"].append(step_name)
        
        hash_object = hashlib.md5((self.name + ">".join(self.data["list"])).encode())
        hash_name = hash_object.hexdigest()
        
        self.data["steps"][step_name] = {
            "step": len(self.data["steps"]),
            "id": hash_name,
            "name": ">".join(self.data["list"]),
            "data_dir": os.path.join(self.save_dir, hash_name),
            "time": time.time() - self.start_time
        }
        
        self.save() # save the data once a new step is added
        
    def get_step(self, step_name):
        return self.data["steps"][step_name]
    
    @property
    def last_step(self):
        return self.data["steps"][self.data["list"][-1]]

    def save(self):
        with open(os.path.join(self.save_dir, f"{self.name}.json"), 'w') as f:
            json.dump(self.data, f, indent=4)
            
    def load(self):
        with open(os.path.join(self.save_dir, f"{self.name}.json"), 'r') as f:
            self.data = json.load(f)
        