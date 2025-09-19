CUDA_VISIBLE_DEVICES=1 python3 -m features_storage.extract --task=run --mode=test --dataclass=nyudv2 --precetorclass=imagebind
CUDA_VISIBLE_DEVICES=1 python3 -m features_storage.extract --task=run --mode=test --dataclass=nyudv2 --precetorclass=languagebind
CUDA_VISIBLE_DEVICES=1 python3 -m features_storage.extract --task=run --mode=test --dataclass=nyudv2 --precetorclass=unibind

# ....