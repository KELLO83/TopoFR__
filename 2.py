import torch
import lightning as L
from torch.utils.data import DataLoader, TensorDataset
from backbones.iresnet import IResNet, IBasicBlock
import time

backbone = IResNet(IBasicBlock, [6, 26, 60, 6] , num_classes=360232 ).to('cuda')

start_time = time.time()
backbone = torch.compile(  backbone , mode="max-autotune")
print("컴파일 소요시간 : {:.2f}초".format(time.time() - start_time))
dummy_data = torch.rand((32,3,112,112),dtype=torch.float16).to('cuda')


out = backbone(dummy_data)
print(out)