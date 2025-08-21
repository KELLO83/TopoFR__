from backbones.iresnet import IResNet, IBasicBlock
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.v2 as v2
import torchinfo

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


# model_info = torchinfo.summary(
#     backbone,
#     input_size=(1, 3, 112, 112),
#     verbose=True,
#     col_names=["input_size", "output_size", "num_params", "trainable","params_percent"],
#     row_settings=["depth"],
#     device='cuda:1',
#     mode='eval'
# )




feature_maps_layer3 = []
feature_maps_layer4 = []

def hook_fn_layer3(module, input, output): 
    feature_maps_layer3.append(output)

def hook_fn_layer4(module, input, output): 
    feature_maps_layer4.append(output)

# 두 레이어에 훅 등록
target_layer3 = backbone.layer3
target_layer4 = backbone.layer4
handle3 = target_layer3.register_forward_hook(hook_fn_layer3)
handle4 = target_layer4.register_forward_hook(hook_fn_layer4)


from glob import glob
file_list = glob('19082031/*.jpg')

cv2.namedWindow('l3',cv2.WINDOW_NORMAL)
cv2.namedWindow('l4',cv2.WINDOW_NORMAL)
for image_path in file_list:
    # 각 이미지마다 feature_maps 초기화
    feature_maps_layer3.clear()
    feature_maps_layer4.clear()
    
    original_img = cv2.imread(image_path)

    if original_img is None:
        print(f"이미지 파일을 찾을 수 없습니다: {image_path}")
    else:
        tensor_image = transform(original_img)
        input_tensor = tensor_image.unsqueeze(0)

        with torch.no_grad():
            output = backbone(input_tensor)

        if not feature_maps_layer3 or not feature_maps_layer4:
            print("훅을 통해 특징 맵을 얻지 못했습니다. 대상 레이어를 확인하세요.")
        else:
            # Layer3 처리
            layer3_feature_map = feature_maps_layer3[-1]
            print(f"{image_path} - Layer3 특징맵 크기: {layer3_feature_map.shape}")

            cam3 = torch.mean(layer3_feature_map, dim=1).squeeze().cpu().numpy()
            cam3 = (cam3 - np.min(cam3)) / (np.max(cam3) - np.min(cam3))

            # Layer4 처리
            layer4_feature_map = feature_maps_layer4[-1]
            print(f"{image_path} - Layer4 특징맵 크기: {layer4_feature_map.shape}")

            cam4 = torch.mean(layer4_feature_map, dim=1).squeeze().cpu().numpy()
            cam4 = (cam4 - np.min(cam4)) / (np.max(cam4) - np.min(cam4))

            # 원본 이미지 준비
            original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            h, w, _ = original_img_rgb.shape

            # Layer3 히트맵 생성
            heatmap3 = cv2.resize(cam3, (w, h))
            heatmap3 = np.uint8(255 * heatmap3)
            heatmap3_color = cv2.applyColorMap(heatmap3, cv2.COLORMAP_JET)
            superimposed3 = cv2.addWeighted(original_img_rgb, 0.5, heatmap3_color, 0.5, 0)

            # Layer4 히트맵 생성
            heatmap4 = cv2.resize(cam4, (w, h))
            heatmap4 = np.uint8(255 * heatmap4)
            heatmap4_color = cv2.applyColorMap(heatmap4, cv2.COLORMAP_JET)
            superimposed4 = cv2.addWeighted(original_img_rgb, 0.5, heatmap4_color, 0.5, 0)
            
            key = cv2.waitKey(0)
            if key == 27:  # ESC 키로 종료
                break

            cv2.imshow("l3",superimposed3)
            cv2.imshow("l4",superimposed4)

# 모든 이미지 처리 완료 후 훅 제거
handle3.remove()
handle4.remove()
cv2.destroyAllWindows()



