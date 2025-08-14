import torch
from backbones.iresnet import IResNet , IBasicBlock
import torchinfo


weight = torch.load('Glint360K_R200_TopoFR_9784.pt', map_location='cpu')
backbone = IResNet(IBasicBlock, [6, 26, 60, 6] , num_classes=360232 ).to('cuda')

#weight = torch.load('MS1MV2_R200_TopoFR_9712_cosface.pt' , map_location='cpu')
#backbone = IResNet(IBasicBlock, [6, 26, 60, 6] , num_classes=85742 )





# weight = torch.load('Glint360K_R100_TopoFR_9760.pt' , map_location='cpu')
# backbone = IResNet(IBasicBlock, [3, 13, 30, 3] , num_classes=360232 )


# weight = torch.load('/home/ubuntu/TopoFR/Glint360K_R50_TopoFR_9727.pt' , map_location='cpu')
# backbone = IResNet(IBasicBlock , [3,4,14,3] , num_classes= 360232)
backbone.eval()
backbone = torch.compile(backbone)
print(backbone)



load_result = backbone.load_state_dict(weight , strict = False)

print("누락된 가중치 : {}".format(load_result.missing_keys))
print("예상치못한 가중치 : {}".format(load_result.unexpected_keys))
print(load_result)
print("="*30)


dummy_input = torch.randn(256, 3, 112, 112).to('cuda')
out = backbone(dummy_input)
print(out[1].shape)
# model_info = torchinfo.summary(
#     backbone,
#     input_size=(756, 3, 112, 112),
#     verbose=False,
#     col_names=["input_size", "output_size", "num_params", "trainable","params_percent"],
#     row_settings=["depth"],
#     device='cuda:1',
#     mode='eval'
# )
# print(model_info)
# backbone.eval()
# backbone = backbone.to('cuda:1')
# dummy_input = dummy_input.to(torch.float32).to('cuda:1')
# output  = backbone(dummy_input)


# print(output[0].shape)  # logits
# print(output[1].shape)  # bottleneck_embedding

# who = torch.argmax(output[0], dim=1)  # 예측된 클래스 인덱스

# print("예측된 클래스 인덱스:", who.item())