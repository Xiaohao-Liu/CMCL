import torch
import os
import time
from itertools import product

def manual_multi_gpu_launcher(commands):
    try:
        # Get list of GPUs from env, split by ',' and remove empty string ''
        # To handle the case when there is one extra comma: `CUDA_VISIBLE_DEVICES=0,1,2,3, python3 ...`
        available_gpus = [x for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',') if x != '']
    except Exception:
        # If the env variable is not set, we use all GPUs
        available_gpus = [str(x) for x in range(torch.cuda.device_count())]
        
    n_gpus = len(available_gpus)
    command_chunks = [commands[i::n_gpus] for i in range(n_gpus)]
    
    sh_file_dir = f"run_sh/{time.time()}"
    os.makedirs(sh_file_dir, exist_ok=True)
    
    for gpu_idx, command_chunk in zip(available_gpus, command_chunks):
        with open(f"{sh_file_dir}/run_{gpu_idx}.sh", "w") as f:
            f.write("\n".join([f'CUDA_VISIBLE_DEVICES={gpu_idx} {cmd}' for cmd in command_chunk]))
    
    print("Please run the following commands:")
    print("\n".join([f"bash {sh_file_dir}/run_{gpu_idx}.sh " for gpu_idx in available_gpus]))
    print("")
    print("OR")
    print("\n".join([f"nohup bash {sh_file_dir}/run_{gpu_idx}.sh > {sh_file_dir}/run_{gpu_idx}.log 2>&1 &" for gpu_idx in available_gpus]))
    

def generate_commands(
    mode="full",
):
    if mode == "full":
        parameters = {
            "perceptor": ["languagebind"],
            "method": ["DNS"],
            "log_name": ["full_testing_final"],
            "dataset_name": [
                "~".join([
                    "ucf101",
                    "esc50",
                    "nyudv2:0,2",
                    "clotho",
                    "vggsound_s:0,1", "vggsound_s:0,2", "vggsound_s:1,2", 
                    "tvl:0,2", "tvl:1,2", "tvl:0,1",
                    "llvip",
                    "flir",
                    "audiocaps",
                    "audioset",
                    "coco",
                    "imagenet",
                ])
            ],
            "seed": [1,2,3,4,5,6,7,8,9,10]
        }
        
    run_file="training.py"
    
    keys, values = zip(*parameters.items())
    commands = [dict(zip(keys, v)) for v in product(*values)]
    
    command_lines = []
    for i in commands:
        if mode == "full":
            i['log_name'] = i["log_name"] + f"_{i['seed']}"
        finished = os.path.exists(f"./results/{i['perceptor']}/{i['method']}/{i['log_name']}_results.json")
        if finished:
            continue
        command = f"python3 {run_file} "
        for key, value in i.items():
            command += f'--{key}="{value}" '
        command_lines.append(command)
    
    print(f"{len(command_lines)}/{len(commands)}")
    
    return command_lines


if __name__ == "__main__":
    commands = generate_commands()
    manual_multi_gpu_launcher(commands)
    