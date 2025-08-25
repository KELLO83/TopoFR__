import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools

far_excel = pd.read_csv('far_elfin_model.csv')
frr_excel = pd.read_csv('frr_elfin_model.csv')

print(far_excel.head(5))
print(far_excel['similarity'][:5])


threshold = 0.7
far_similarity = far_excel['similarity'].tolist()
far_similarity = np.array(far_similarity)
far_similarity = np.unique(far_similarity)
frr_similarity = frr_excel['similarity'].tolist()
frr_similarity = np.array(frr_similarity)
frr_similarity = np.unique(frr_similarity)

merged_similarity = np.concatenate((far_similarity, frr_similarity), axis=0)

print(len(far_similarity))
print(len(frr_similarity))

lebel_far = np.zeros(len(far_similarity))
lebel_frr = np.ones(len(frr_similarity))
labels = np.concatenate((lebel_far, lebel_frr), axis=0)

    
predictions = ( merged_similarity >= threshold).astype(int)
cm =  confusion_matrix( labels , predictions)

# --- Bug Fix and Performance Calculation ---
# Correctly unpack the confusion matrix: TN, FP, FN, TP
tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

# Calculate performance metrics
total_positives = tp + fn
total_negatives = tn + fp
total_samples = total_positives + total_negatives

accuracy = (tp + tn) / total_samples if total_samples > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / total_positives if total_positives > 0 else 0 # Also known as True Positive Rate (TPR)
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# False Acceptance Rate (FAR) and False Rejection Rate (FRR)
far = fp / total_negatives if total_negatives > 0 else 0
frr = fn / total_positives if total_positives > 0 else 0

# --- Print Results ---
print(f"\n--- 성능 평가 결과 (임계값: {threshold}) ---")

print("\n--- 🔢 Confusion Matrix ---")
print(f"                     | 예측: 타인 (0) | 예측: 본인 (1)")
print("---------------------|----------------|----------------")
print(f"실제: 타인 (0) |      {tn:^6} (TN) |      {fp:^6} (FP)")
print(f"실제: 본인 (1) |      {fn:^6} (FN) |      {tp:^6} (TP)")
print("\n")
print(f"True Positive (TP): {tp} (본인을 본인으로 정확히 인식)")
print(f"True Negative (TN): {tn} (타인을 타인으로 정확히 인식)")
print(f"False Positive (FP): {fp} (타인을 본인으로 잘못 인식) - FAR 오류")
print(f"False Negative (FN): {fn} (본인을 타인으로 잘못 인식) - FRR 오류")

print("\n--- 📊 주요 성능 지표 ---")
print(f"정확도 (Accuracy): {accuracy:.4f}")
print(f"정밀도 (Precision): {precision:.4f}")
print(f"재현율 (Recall/TPR): {recall:.4f}")
print(f"F1-Score: {f1_score:.4f}")
print(f"오인식률 (FAR): {far:.4f} ({fp}/{total_negatives})")
print(f"오거부율 (FRR): {frr:.4f} ({fn}/{total_positives})")

# --- Detailed Statistics Calculation ---
# Positive (Same Person) Pairs Statistics
if len(frr_similarity) > 0:
    pos_min = np.min(frr_similarity)
    pos_p10 = np.percentile(frr_similarity, 10)
    pos_median = np.median(frr_similarity)
    pos_p90 = np.percentile(frr_similarity, 90)
    pos_max = np.max(frr_similarity)
    pos_mean = np.mean(frr_similarity.astype(np.float64))
    pos_std = np.std(frr_similarity.astype(np.float64)) if len(frr_similarity) > 1 else float('nan')
else:
    pos_min, pos_p10, pos_median, pos_p90, pos_max, pos_mean, pos_std = [float('nan')] * 7

# Negative (Different Person) Pairs Statistics
if len(far_similarity) > 0:
    neg_min = np.min(far_similarity)
    neg_p10 = np.percentile(far_similarity, 10)
    neg_median = np.median(far_similarity)
    neg_p90 = np.percentile(far_similarity, 90)
    neg_max = np.max(far_similarity)
    neg_mean = np.mean(far_similarity.astype(np.float64))
    neg_std = np.std(far_similarity.astype(np.float64)) if len(far_similarity) > 1 else float('nan')
else:
    neg_min, neg_p10, neg_median, neg_p90, neg_max, neg_mean, neg_std = [float('nan')] * 7


# --- Print Detailed Statistics ---
print("\n--- 유사도 분포 상세 분석 ---")

if not np.isnan(pos_mean):
    print(f"🔵 동일 인물 쌍 유사도 (총 {len(frr_similarity):,}개):")
    print(f"   - 최소값: {pos_min:.4f}")
    print(f"   - 10% 분위: {pos_p10:.4f}")
    print(f"   - 중앙값 (Median): {pos_median:.4f}")
    print(f"   - 90% 분위: {pos_p90:.4f}")
    print(f"   - 최대값: {pos_max:.4f}")
    print(f"   - 평균값: {pos_mean:.4f}")
    print(f"   - 표준편차: {pos_std:.4f}")
else:
    print("🔵 동일 인물 쌍 데이터가 없습니다.")

if not np.isnan(neg_mean):
    print(f"\n🔴 다른 인물 쌍 유사도 (총 {len(far_similarity):,}개):")
    print(f"   - 최소값: {neg_min:.4f}")
    print(f"   - 10% 분위: {neg_p10:.4f}")
    print(f"   - 중앙값 (Median): {neg_median:.4f}")
    print(f"   - 90% 분위: {neg_p90:.4f}")
    print(f"   - 최대값: {neg_max:.4f}")
    print(f"   - 평균값: {neg_mean:.4f}")
    print(f"   - 표준편차: {neg_std:.4f}")
else:
    print("\n🔴 다른 인물 쌍 데이터가 없습니다.")


# --- Write Results to File ---
output_filename = "elfin_model_test_result.txt"
with open(output_filename, "w") as f:
    f.write(f"--- 성능 평가 결과 (임계값: {threshold}) ---\n\n")

    f.write(f"True Positive (TP): {tp} (본인을 본인으로 정확히 인식)\n")
    f.write(f"True Negative (TN): {tn} (타인을 타인으로 정확히 인식)\n")
    f.write(f"False Positive (FP): {fp} (타인을 본인으로 잘못 인식) - FAR 오류\n")
    f.write(f"False Negative (FN): {fn} (본인을 타인으로 잘못 인식) - FRR 오류\n\n")

    f.write(f"정확도 (Accuracy): {accuracy:.4f}\n")
    f.write(f"정밀도 (Precision): {precision:.4f}\n")
    f.write(f"재현율 (Recall/TPR): {recall:.4f}\n")
    f.write(f"F1-Score: {f1_score:.4f}\n")
    f.write(f"오인식률 (FAR): {far:.4f} ({fp}/{total_negatives})\n")
    f.write(f"오거부율 (FRR): {frr:.4f} ({fn}/{total_positives})\n\n")

    if not np.isnan(pos_mean):
        f.write(f"🔵 동일 인물 쌍 유사도 (총 {len(frr_similarity):,}개):\n")
        f.write(f"   - 최소값: {pos_min:.4f}\n")
        f.write(f"   - 10% 분위: {pos_p10:.4f}\n")
        f.write(f"   - 중앙값 (Median): {pos_median:.4f}\n")
        f.write(f"   - 90% 분위: {pos_p90:.4f}\n")
        f.write(f"   - 최대값: {pos_max:.4f}\n")
        f.write(f"   - 평균값: {pos_mean:.4f}\n")
        f.write(f"   - 표준편차: {pos_std:.4f}\n")

    if not np.isnan(neg_mean):
        f.write(f"🔴 다른 인물 쌍 유사도 (총 {len(far_similarity):,}개):\n")
        f.write(f"   - 최소값: {neg_min:.4f}\n")
        f.write(f"   - 10% 분위: {neg_p10:.4f}\n")
        f.write(f"   - 중앙값 (Median): {neg_median:.4f}\n")
        f.write(f"   - 90% 분위: {neg_p90:.4f}\n")
        f.write(f"   - 최대값: {neg_max:.4f}\n")
        f.write(f"   - 평균값: {neg_mean:.4f}\n")
        f.write(f"   - 표준편차: {neg_std:.4f}\n")


print(f"\n결과가 '{output_filename}' 파일에 저장되었습니다.")

# --- Visualization ---
print("\n시각화 생성 중...")
plt.figure(figsize=(12, 6))
# 1. Similarity Distribution Histogram
plt.subplot(1, 2, 1)
plt.hist(frr_similarity, bins=30, alpha=0.7, color='green', label=f'Same Person (n={len(frr_similarity)})', density=True)
plt.hist(far_similarity, bins=30, alpha=0.7, color='red', label=f'Different Person (n={len(far_similarity)})', density=True)
plt.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.2f}')
plt.xlabel('Cosine Similarity')
plt.ylabel('Density')
plt.title('Similarity Distribution')
plt.legend()
plt.grid(True)
# 2. Confusion Matrix Heatmap
plt.subplot(1, 2, 2)
cm_sum = cm.sum(axis=1)[:, np.newaxis]
cm_normalized = cm.astype('float') / cm_sum if np.all(cm_sum > 0) else cm
im = plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
plt.title('Normalized Confusion Matrix')
plt.colorbar(im)
classes = ['Different Person', 'Same Person']
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
# Add text annotations
thresh = cm_normalized.max() / 2. if cm_normalized.max() > 0 else 0.5
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, f'{cm[i, j]}\n({cm_normalized[i, j]:.2%})',
             ha="center", va="center",
             color="white" if cm_normalized[i, j] > thresh else "black")
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
# Save the plot
plot_filename = "elfin_model_test_visualization.png"
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
print(f"📊 평가 결과 그래프가 '{plot_filename}' 파일로 저장되었습니다.")
plt.show()