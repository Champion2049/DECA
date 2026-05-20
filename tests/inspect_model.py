import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch

m = torch.load('logs/facescape_finetune_model_c/model.tar', map_location='cpu')
print('Top-level keys:', list(m.keys()))
for k in m.keys():
    if isinstance(m[k], dict):
        print(f'  {k}: {len(m[k])} subkeys, e.g. {list(m[k].keys())[:3]}')
    else:
        print(f'  {k}: {type(m[k])}')
