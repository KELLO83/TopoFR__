from backbones.iresnet import IResNet, IBasicBlock
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.v2 as v2
import torchinfo
import copy
from glob import glob
from natsort import natsorted
from torchcam.methods import LayerCAM, SmoothGradCAMpp

file_list = glob('19082031/*.jpg')
file_list = natsorted(file_list)

transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.CenterCrop(size=(112, 112)),
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

Weight_path = 'Glint360K_R200_TopoFR_9784.pt'
backbone = IResNet(IBasicBlock, [6, 26, 60, 6], num_classes=360232)
backbone
backbone.eval() 


try:
    load_result = backbone.load_state_dict(torch.load(Weight_path, map_location='cpu'), strict=False)
    print("가중치 로드 성공.")
    print("누락된 가중치: {}".format(load_result.missing_keys))
    print("예상치 못한 가중치: {}".format(load_result.unexpected_keys))
except FileNotFoundError:
    print(f"{Weight_path}에서 가중치 파일을 찾을 수 없습니다. 무작위 초기화된 모델을 사용합니다.")

backbone = backbone.to('cuda')
backbone_copy = copy.deepcopy(backbone)
feature_maps_layer4 = []

def hook_fn_layer4(module, input, output):
    feature_maps_layer4.append(output)

target_layer4 = backbone.layer4
handle4 = target_layer4.register_forward_hook(hook_fn_layer4)

cam_extractor_layercam = LayerCAM(backbone, target_layer=backbone.bn2)

cv2.namedWindow("layer hook", cv2.WINDOW_NORMAL)
cv2.namedWindow("grad cam", cv2.WINDOW_NORMAL)

import copy
for image_path in file_list:
    print(f"--- Processing {image_path} ---")
    feature_maps_layer4.clear()
    image = cv2.imread(image_path)
    original_img = cv2.imread(image_path)
    if original_img is None:
        print(f"  -> Image not found, skipping.")
        continue
    copy_image = copy.deepcopy(image)
    original_img_resized = cv2.resize(original_img, (112, 112))

    tensor_image = transform(original_img_resized)
    input_tensor = tensor_image.unsqueeze(0).to('cuda')
    output = backbone(input_tensor)

    # --- Visualization Method 1: Manual Hook on layer4 ---
    if not feature_maps_layer4:
        print("  -> Failed to get feature map from hook.")
    else:
        try:
            layer4_feature_map = feature_maps_layer4[-1]
            cam4 = torch.mean(layer4_feature_map, dim=1).squeeze().detach().cpu().numpy()
            cam4 = (cam4 - np.min(cam4)) / (np.max(cam4) - np.min(cam4))

            original_img_rgb = cv2.cvtColor(copy_image, cv2.COLOR_BGR2RGB)
            h, w, _ = copy_image.shape

            heatmap4 = cv2.resize(cam4.astype(np.float32), (112, 112))
            heatmap4 = np.uint8(255 * heatmap4)
            heatmap4_color = cv2.applyColorMap(heatmap4, cv2.COLORMAP_JET)
            superimposed4 = cv2.addWeighted(original_img_rgb, 0.5, heatmap4_color, 0.5, 0)
            cv2.imshow("layer hook", superimposed4)

        except Exception as e:
            print(f"  -> Failed to generate Hook visualization: {e}")

    # --- Visualization Method 2: LayerCAM on bn2 ---
    try:
        activation_map_lc = cam_extractor_layercam(output.squeeze(0).argmax().item(), output)
        raw_map_lc = activation_map_lc[0].cpu().numpy().squeeze()
        raw_map_lc = np.nan_to_num(raw_map_lc).astype(np.float32)
        resized_map_lc = cv2.resize(raw_map_lc, (112, 112))
        heatmap_lc = cv2.normalize(resized_map_lc, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        colored_heatmap_lc = cv2.applyColorMap(heatmap_lc, cv2.COLORMAP_JET)
        overlayed_image_lc = cv2.addWeighted(original_img_resized, 0.6, colored_heatmap_lc, 0.4, 0)
        cv2.imshow("grad cam", overlayed_image_lc)

    except Exception as e:
        print(f"  -> Failed to generate LayerCAM: {e}")

    from torchcam.methods import LayerCAM, SmoothGradCAMpp
    # --- Visualization Method 3: SmoothGradCAMpp on bn2 ---
    try:
        with SmoothGradCAMpp(backbone_copy, target_layer=backbone_copy.bn2) as cam_extractor_sgcam:
            input_tensor = input_tensor.to('cuda')
            out_sgcam = backbone_copy(input_tensor)
            activation_map_sgcam = cam_extractor_sgcam(out_sgcam.squeeze(0).argmax().item(), out_sgcam)

            # 2c. Activation map 정제 및 시각화
            raw_map_sgcam = activation_map_sgcam[0].cpu().numpy().squeeze()
            raw_map_sgcam = np.nan_to_num(raw_map_sgcam).astype(np.float32)
            resized_map_sgcam = cv2.resize(raw_map_sgcam, (copy_image.shape[1], copy_image.shape[0]))
            heatmap_sgcam = cv2.normalize(resized_map_sgcam, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            colored_heatmap_sgcam = cv2.applyColorMap(heatmap_sgcam, cv2.COLORMAP_JET)
            overlayed_image_sgcam = cv2.addWeighted(copy_image, 0.6, colored_heatmap_sgcam, 0.4, 0)
            cv2.imshow("SmoothGradCAMpp", overlayed_image_sgcam)
            
    except Exception as e:
        print(f"  -> Failed to generate SmoothGradCAMpp: {e}")
    cv2.waitKey(0)


# --- 3. Cleanup ---
handle4.remove()
print("\nProcessing complete. Visualizations saved in 'visualization_results' directory.")



