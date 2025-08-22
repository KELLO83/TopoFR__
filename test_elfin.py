import os
import glob
import cv2
import itertools
from natsort import natsorted
from backbones.iresnet import IResNet , IBasicBlock
import torch
import numpy as np
import matplotlib.pyplot as plt


def calculate_cosine_similarity_numpy(vec1, vec2):
    # GPU 텐서를 CPU numpy로 변환
    if isinstance(vec1, torch.Tensor):
        vec1 = vec1.detach().cpu().numpy()
    if isinstance(vec2, torch.Tensor):
        vec2 = vec2.detach().cpu().numpy()

    vec1 = vec1.flatten()
    vec2 = vec2.flatten()
    
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    vec1_norm = vec1 / norm1
    vec2_norm = vec2 / norm2
    

    cosine_similarity = np.dot(vec1_norm, vec2_norm)
    

    if np.isfinite(cosine_similarity):
        return cosine_similarity
    else:
        return 0.0


representative_image = cv2.imread('frr_detected/0.jpg')
far_list = glob.glob('far_detected/*.jpg')  # 다른 인물 (Negative samples)
frr_list = glob.glob('frr_detected/*.jpg')  # 같은 인물 (Positive samples)

far_list = natsorted(far_list)
frr_list = natsorted(frr_list)

print(f"FAR 이미지 수: {len(far_list)} (다른 인물)")
print(f"FRR 이미지 수: {len(frr_list)} (같은 인물)")
print(f"총 평가 이미지 수: {len(far_list) + len(frr_list)}")

Weight_path = 'Glint360K_R100_TopoFR_9760.pt'
backbone = IResNet(IBasicBlock, [3, 13, 30, 3] , num_classes=360232)
load_result = backbone.load_state_dict(torch.load(Weight_path, map_location='cpu'), strict=False)
backbone.eval()
backbone = backbone.to('cuda')
print("누락된 가중치 : {}".format(load_result.missing_keys))
print("예상치못한 가중치 : {}".format(load_result.unexpected_keys))

if not load_result.missing_keys and not load_result.unexpected_keys:
    print("모델 가중치가 성공적으로 로드되었습니다.")

import torchvision.transforms.v2 as v2
from sklearn.metrics import roc_curve, auc, confusion_matrix

transforms_v2 = v2.Compose([
    v2.ToImage(), # cv2 HWC ---> Tensor C H W
    v2.ToDtype(torch.float32, scale=True),
    v2.CenterCrop(size=(112, 112)),
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# representative 이미지도 동일한 전처리 적용
if representative_image.shape[0] != 112 or representative_image.shape[1] != 112:
    representative_image_resized = cv2.resize(representative_image, (112, 112), 
                                            interpolation=cv2.INTER_CUBIC if representative_image.shape[0] > 112 else cv2.INTER_AREA)
else:
    representative_image_resized = representative_image.copy()

# BGR → RGB 변환
representative_image_rgb = cv2.cvtColor(representative_image_resized, cv2.COLOR_BGR2RGB)

representative_image_tensor = transforms_v2(representative_image_rgb)  # ✅ RGB 순서로 처리
representative_image_vector = representative_image_tensor.unsqueeze(0).to('cuda')
_ , representative_image_vector = backbone(representative_image_vector)

cv2.namedWindow('Representative Image', cv2.WINDOW_NORMAL)
cv2.namedWindow("compare_image",cv2.WINDOW_NORMAL)

# 모든 유사도와 라벨 저장
all_similarities = []
all_labels = []

# 1. FRR 이미지들 처리 (같은 인물 - Positive samples, label=1)
print("\n=== FRR 이미지 처리 (같은 인물) ===")
for i, frr_img_path in enumerate(frr_list):
    if i == 0:  # representative 이미지는 제외
        continue
        
    image = cv2.imread(frr_img_path)
    
    # top_k_eval.py와 동일한 전처리 적용
    if image.shape[0] != 112 or image.shape[1] != 112:
        image = cv2.resize(image, (112, 112), 
                          interpolation=cv2.INTER_CUBIC if image.shape[0] > 112 else cv2.INTER_AREA)
    
    # BGR → RGB 변환
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Transform 적용
    image_tensor = transforms_v2(image_rgb)  # ✅ RGB 순서로 처리
    image_tensor = image_tensor.unsqueeze(0).to('cuda')

    with torch.no_grad():
        output = backbone(image_tensor)
        if isinstance(output , tuple):
            _ , vectors  = output
        else:
            vectors = output

    cos_similarity = torch.nn.functional.cosine_similarity(representative_image_vector, vectors, dim=1)
    cos_similarity = cos_similarity.detach().cpu().numpy()[0]
    
    all_similarities.append(cos_similarity)
    all_labels.append(1)  # Positive sample
    
    print(f"FRR {i}: {frr_img_path} -> Similarity: {cos_similarity:.4f}")
    
# 2. FAR 이미지들 처리 (다른 인물 - Negative samples, label=0)
print("\n=== FAR 이미지 처리 (다른 인물) ===")
for i, far_img_path in enumerate(far_list):
    image = cv2.imread(far_img_path)
    
    # top_k_eval.py와 동일한 전처리 적용
    if image.shape[0] != 112 or image.shape[1] != 112:
        image = cv2.resize(image, (112, 112), 
                          interpolation=cv2.INTER_CUBIC if image.shape[0] > 112 else cv2.INTER_AREA)
    
    # BGR → RGB 변환
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Transform 적용
    image_tensor = transforms_v2(image_rgb)  # ✅ RGB 순서로 처리
    image_tensor = image_tensor.unsqueeze(0).to('cuda')

    with torch.no_grad():
        output = backbone(image_tensor)
        if isinstance(output , tuple):
            _ , vectors  = output
        else:
            vectors = output

    cos_similarity = torch.nn.functional.cosine_similarity(representative_image_vector, vectors, dim=1)
    cos_similarity = cos_similarity.detach().cpu().numpy()[0]
    
    all_similarities.append(cos_similarity)
    all_labels.append(0)  # Negative sample
    
    print(f"FAR {i}: {far_img_path} -> Similarity: {cos_similarity:.4f}")

cv2.destroyAllWindows()

# 3. 성능 평가 수행
print("\n=== 성능 평가 ===")
similarities_np = np.array(all_similarities)
labels_np = np.array(all_labels)

print(f"총 평가 쌍: {len(similarities_np)}개")
print(f"Positive 쌍 (같은 인물): {np.sum(labels_np)}개")
print(f"Negative 쌍 (다른 인물): {np.sum(labels_np == 0)}개")

if len(similarities_np) > 0:
    # ROC 커브 계산
    fpr, tpr, thresholds = roc_curve(labels_np, similarities_np)
    roc_auc = auc(fpr, tpr)
    
    # EER 계산
    frr = 1 - tpr  # FRR = 1 - TPR
    eer_index = np.nanargmin(np.abs(fpr - frr))
    eer = fpr[eer_index]
    eer_threshold = thresholds[eer_index]
    
    # EER 임계점으로 분류
    predictions = (similarities_np >= eer_threshold).astype(int)
    cm = confusion_matrix(labels_np, predictions)
    
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    # 성능 지표 계산
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # FAR과 FRR 계산
    far = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Accept Rate
    frr_rate = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Reject Rate
    
    print("\n--- 📊 최종 성능 결과 ---")
    print(f"🎯 ROC-AUC: {roc_auc:.4f}")
    print(f"🎯 EER: {eer:.4f} (임계값: {eer_threshold:.4f})")
    print(f"📈 Accuracy: {accuracy:.4f}")
    print(f"📈 Precision: {precision:.4f}")
    print(f"📈 Recall (TPR): {recall:.4f}")
    print(f"📈 Specificity (TNR): {specificity:.4f}")
    print(f"📈 F1-Score: {f1_score:.4f}")
    print(f"📈 FAR (False Accept Rate): {far:.4f}")
    print(f"📈 FRR (False Reject Rate): {frr_rate:.4f}")
    
    print("\n--- 🔢 Confusion Matrix ---")
    print(f"True Positive (TP): {tp} (같은 인물을 같은 인물로 인식)")
    print(f"True Negative (TN): {tn} (다른 인물을 다른 인물로 인식)")
    print(f"False Positive (FP): {fp} (다른 인물을 같은 인물로 오인식)")
    print(f"False Negative (FN): {fn} (같은 인물을 다른 인물로 오인식)")

    # --- Enhanced Statistics ---
    p10 = np.percentile(similarities_np, 10)
    p50 = np.median(similarities_np) # Median
    p90 = np.percentile(similarities_np, 90)

    print("\n--- 유사도 분포 분석 ---")
    print(f"🔴 유사도 (총 {len(similarities_np):,}개):")
    print(f"   - 최소값: {np.min(similarities_np):.4f}")
    print(f"   - 10% 분위: {p10:.4f}")
    print(f"   - 중앙값 (Median): {p50:.4f}")
    print(f"   - 90% 분위: {p90:.4f}")
    print(f"   - 최대값: {np.max(similarities_np):.4f}")
    print(f"   - 평균값: {np.mean(similarities_np):.4f}")
    print(f"   - 표준편차: {np.std(similarities_np):.4f}")

    # --- ROC Curve Visualization ---
    plt.figure(figsize=(15, 5))
    
    # 1. ROC 커브
    plt.subplot(1, 3, 1)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.scatter(fpr[eer_index], tpr[eer_index], color='red', s=100, zorder=5, label=f'EER = {eer:.4f}')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FAR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    
    # 2. 유사도 분포 히스토그램
    plt.subplot(1, 3, 2)
    positive_similarities = similarities_np[labels_np == 1]
    negative_similarities = similarities_np[labels_np == 0]
    
    plt.hist(positive_similarities, bins=30, alpha=0.7, color='green', label=f'Same Person (n={len(positive_similarities)})', density=True)
    plt.hist(negative_similarities, bins=30, alpha=0.7, color='red', label=f'Different Person (n={len(negative_similarities)})', density=True)
    plt.axvline(eer_threshold, color='black', linestyle='--', linewidth=2, label=f'EER Threshold: {eer_threshold:.3f}')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Density')
    plt.title('Similarity Distribution')
    plt.legend()
    plt.grid(True)
    
    # 3. Confusion Matrix 히트맵
    plt.subplot(1, 3, 3)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    im = plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Normalized Confusion Matrix')
    plt.colorbar(im)
    
    classes = ['Different Person', 'Same Person']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # 텍스트 추가
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f'{cm[i, j]}\n({cm_normalized[i, j]:.2%})',
                ha="center", va="center", color="white" if cm_normalized[i, j] > 0.5 else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    plot_filename = "face_recognition_evaluation.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"\n📊 평가 결과 그래프가 '{plot_filename}' 파일로 저장되었습니다.")
    
    plt.show()

else:
    print("⚠️ 평가할 데이터가 없습니다.")
