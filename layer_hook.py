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




feature_maps = []
def hook_fn(module, input, output): feature_maps.append(output)


target_layer = backbone.layer4
handle = target_layer.register_forward_hook(hook_fn)


from glob import glob
file_list = glob('19082031/*.jpg')

for image_path in file_list:
    # 각 이미지마다 feature_maps 초기화
    feature_maps.clear()
    
    original_img = cv2.imread(image_path)

    if original_img is None:
        print(f"이미지 파일을 찾을 수 없습니다: {image_path}")
    else:
        tensor_image = transform(original_img)
        input_tensor = tensor_image.unsqueeze(0)

        with torch.no_grad():
            output = backbone(input_tensor)

        if not feature_maps:
            print("훅을 통해 특징 맵을 얻지 못했습니다. 대상 레이어를 확인하세요.")
        else:
            last_feature_map = feature_maps[-1]
            print(f"{image_path} - 특징맵 크기: {last_feature_map.shape}")

            cam = torch.mean(last_feature_map, dim=1).squeeze().cpu().numpy()
            cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))

            original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
            h, w, _ = original_img_rgb.shape
            heatmap = cv2.resize(cam, (w, h))
            heatmap = np.uint8(255 * heatmap)

            heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            superimposed_img = cv2.addWeighted(original_img_rgb, 0.5, heatmap_color, 0.5, 0)

            cv2.namedWindow("sup",cv2.WINDOW_NORMAL)
            cv2.imshow('sup',superimposed_img)
            key = cv2.waitKey(0)
            if key == 27:  # ESC 키로 종료
                break

# 모든 이미지 처리 완료 후 훅 제거
handle.remove()
cv2.destroyAllWindows()



