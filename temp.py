from icnet import icnet
import torch

model = icnet()

pth = torch.load('assets/icnet_cityscapes_trainval_90k.pth')
print(pth['model_state'].keys())

pth = torch.load('assets/icnetBN_cityscapes_train_30k.pth')
print(pth['model_state'].keys())

pth2 = torch.load('assets/depth_cityscapes_best_model.pkl')
print(pth2['model_state'].keys())