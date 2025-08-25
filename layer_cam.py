import os
from glob import glob
from natsort import natsorted
import cv2
import numpy as np
import torch
from torchcam.methods import LayerCAM, SmoothGradCAMpp
import torchvision.transforms.v2 as v2
from backbones.iresnet import IResNet , IBasicBlock
import copy
import torch.nn as nn

Weight_path = 'Glint360K_R200_TopoFR_9784.pt'
backbone = IResNet(IBasicBlock, [6, 26, 60, 6] , num_classes=360232)
load_result = backbone.load_state_dict(torch.load(Weight_path, map_location='cpu'), strict=False)
    
        
backbone = backbone.to('cuda')
cam_model = copy.deepcopy(backbone)
cam_model = cam_model.to(torch.float32)

file_list = glob('frr_detected/*.jpg')
file_list = natsorted(file_list)

transforms_v2 = v2.Compose([
    v2.ToImage(), 
    v2.Lambda(lambda x: x.flip(dims=(0,))),
    v2.ToDtype(torch.float32, scale=True),
    v2.CenterCrop(size=(112, 112)),
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

print(backbone)
feature_maps_layer4 = []
def hook_fn_layer4(module, input, output):
    feature_maps_layer4.append(output)
target_layer4 = backbone.layer4
handle4 = target_layer4.register_forward_hook(hook_fn_layer4)
cv2.namedWindow('LayerCAM',cv2.WINDOW_NORMAL)
cv2.namedWindow('layer hook',cv2.WINDOW_NORMAL)
cv2.namedWindow("EigenCAM on Layer4")
for f in file_list:
    image = cv2.imread(f)

    if image is None:
        raise FileExistsError
    image = cv2.resize(image, (112, 112),interpolation=cv2.INTER_CUBIC if image.shape[0] > 112 else cv2.INTER_AREA)

    input_tensor = transforms_v2(image)
    cam_extractor_layercam = LayerCAM(backbone, target_layer=backbone.bn2)
    out_layercam = backbone(input_tensor.unsqueeze(0).to('cuda'))

    activation_map_layercam = cam_extractor_layercam(out_layercam.squeeze(0).argmax().item(), out_layercam)
    print("Activation layer cam shaep : ",activation_map_layercam[0].shape)

    raw_map_layercam = activation_map_layercam[0].cpu().numpy().squeeze()
    raw_map_layercam = np.nan_to_num(raw_map_layercam).astype(np.float32)
    resized_map_layercam = cv2.resize(raw_map_layercam, (image.shape[1], image.shape[0]))
    heatmap_layercam = cv2.normalize(resized_map_layercam, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    colored_heatmap_layercam = cv2.applyColorMap(heatmap_layercam, cv2.COLORMAP_JET)
    overlayed_image_layercam = cv2.addWeighted(image, 0.5, colored_heatmap_layercam, 0.5, 0)
    cv2.imshow("LayerCAM", overlayed_image_layercam)

    # with SmoothGradCAMpp(backbone, target_layer=backbone.bn2) as cam_extractor_sgcam:
    #     input_tensor = input_tensor.to('cuda').unsqueeze(0)
    #     out_sgcam = backbone(input_tensor)
    #     activation_map_sgcam = cam_extractor_sgcam(out_sgcam.squeeze(0).argmax().item(), out_sgcam)
    #     raw_map_sgcam = activation_map_sgcam[0].cpu().numpy().squeeze()
    #     raw_map_sgcam = np.nan_to_num(raw_map_sgcam).astype(np.float32)
    #     resized_map_sgcam = cv2.resize(raw_map_sgcam, (image.shape[1], image.shape[0]))
    #     heatmap_sgcam = cv2.normalize(resized_map_sgcam, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    #     colored_heatmap_sgcam = cv2.applyColorMap(heatmap_sgcam, cv2.COLORMAP_JET)
    #     overlayed_image_sgcam = cv2.addWeighted(image, 0.5, colored_heatmap_sgcam, 0.5, 0)
    #     cv2.imshow("SmoothGradCAMpp", overlayed_image_sgcam)


    layer4_feature_map = feature_maps_layer4[-1]
    cam4 = torch.mean(layer4_feature_map, dim=1).squeeze().detach().cpu().numpy()
    cam4 = (cam4 - np.min(cam4)) / (np.max(cam4) - np.min(cam4))

    original_img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape

    heatmap4 = cv2.resize(cam4.astype(np.float32), (112, 112))
    heatmap4 = np.uint8(255 * heatmap4)
    heatmap4_color = cv2.applyColorMap(heatmap4, cv2.COLORMAP_JET)
    superimposed4 = cv2.addWeighted(original_img_rgb, 0.5, heatmap4_color, 0.5, 0)
    cv2.imshow("layer hook", superimposed4)


    from pytorch_grad_cam import EigenCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    target_layers = [cam_model.layer4[-1].bn3]
    cam = EigenCAM(model=cam_model, target_layers=target_layers)
    input_tensor = input_tensor.unsqueeze(0)
    input_tensor = input_tensor.type(torch.float32)
    grayscale_cam = cam(input_tensor=input_tensor)
    grayscale_cam = grayscale_cam[0, :]
    print("Eigen cam shape ",grayscale_cam.shape)
    image_resized = cv2.resize(image, (112, 112))
    rgb_img_float = np.float32(cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)) / 255
    visualization = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)
    cv2.imshow("EigenCAM on Layer4", visualization)

    cv2.waitKey(0)