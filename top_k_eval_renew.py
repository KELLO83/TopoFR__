import os
import itertools
import random
import numpy as np
import pandas as pd
import torch
import cv2
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, confusion_matrix
import logging
import traceback
import pickle
import argparse
import matplotlib.pyplot as plt
import logging
import torchvision.transforms.v2 as v2
from PIL import Image
import numpy as np
from backbones.iresnet import IResNet , IBasicBlock
from multiprocessing.pool import Pool
from datetime import datetime

try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()


def init_worker(worker_embeddings):
    """워커 프로세스 초기화 함수. embeddings 딕셔너리를 전역 변수로 설정합니다."""
    global embeddings
    embeddings = worker_embeddings

transforms_v2 = v2.Compose([
    v2.ToImage(), # cv2 HWC ---> Tensor C H W
    v2.ToDtype(torch.float32, scale=True),
    v2.CenterCrop(size=(112, 112)),
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

@torch.inference_mode()
def get_all_embeddings(identity_map, backbone, device, batch_size):

    logging.info(f"임베딩 추출 배치사이즈 : {batch_size}")

    if isinstance(device, str):
        device = torch.device(device)
    
    backbone = backbone.to(device)
    backbone.eval()
    embeddings = {} # {이미지경로 : 이미지벡터}
    all_images = sorted(list(set(itertools.chain.from_iterable(identity_map.values())))) #모든이미지경로 평탄화
     
    def preprocess_image(image):
        transformed_image = transforms_v2(image)
        return transformed_image

    for i in tqdm(range(0, len(all_images), batch_size), desc='임베딩 추출'):
        batch_paths = all_images[i:i+batch_size] # 배치단위로 경로 추출
        batch_images = []
        valid_paths = []

        for img_path in batch_paths: # 배치단위로 하나씩 -> tensor값으로 변환
            try:
                image = cv2.imread(img_path)
                if image is None:
                    logging.warning(f"{img_path} 경로 이미지가 비었습니다")
                    embeddings[img_path] = None
                    continue

                if image.shape[0] != 112 or image.shape[1] != 112:
                    image = cv2.resize(image, (112, 112), interpolation=cv2.INTER_CUBIC if image.shape[0] > 112 else cv2.INTER_AREA)
                
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                processed_image = preprocess_image(image_rgb)
                batch_images.append(processed_image)
                valid_paths.append(img_path)

            except Exception as e:
                logging.warning(f"이미지 처리 실패 경로 : {img_path} 오류 : {e}")
                embeddings[img_path] = None
        
        if not batch_images:
            continue

        batch_tensor = torch.stack(batch_images).to(device) # 배치단위로 하나로만들어

        try:
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                
                output = backbone(batch_tensor)
                if isinstance(output , tuple):
                    _ , vectors  = output

                else:
                    vectors = output
        

            if vectors is None or vectors.numel() == 0:
                logging.warning(f"벡터 추출 실패 (배치 크기: {len(batch_paths)})")
                for path in valid_paths:
                    embeddings[path] = None

            else:
                vectors_cpu = vectors.cpu().numpy()
                for path, vector in zip(valid_paths, vectors_cpu):
                    embeddings[path] = vector.flatten()

        except Exception as e:
            logging.warning(f"임베딩 추출 실패 (배치 크기: {len(batch_paths)}) 오류 : {e}")
            for path in valid_paths:
                embeddings[path] = None
    try:
        sp = args.data_path
        sp = sp.split('/')[-1]
        file_name = f'{args.model}.npz'
        np.savez_compressed(f'{file_name}' , **embeddings)
        logging.info(f"임베딩 캐시 저장완료 파일이름 : {file_name}")

    except Exception as e:
        logging.info(f'{e}')
        logging.info("@@임베딩 캐시 저장 실패 코드검수.....!!@@")

    return embeddings

def collect_scores_from_embeddings(pairs, embeddings, is_positive, total_pairs=None):
    """임베딩으로 유사도를 계산합니다 (코사인 유사도 사용)."""
    similarities, labels = [], []
    label = 1 if is_positive else 0
    desc = "동일 인물 쌍 계산" if is_positive else "다른 인물 쌍 계산"

    if total_pairs is None:
        try:
            total_pairs = len(pairs)
        except TypeError:
            total_pairs = None 

    for img1_path, img2_path in tqdm(pairs, desc=desc, total=total_pairs):
        emb1, emb2 = embeddings.get(img1_path), embeddings.get(img2_path)
        if emb1 is not None and emb2 is not None:
        
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            
            if norm1 == 0 or norm2 == 0:
                logging.warning(f"Zero norm embedding found: {img1_path} or {img2_path}")
                continue
            
            emb1_norm = emb1 / norm1
            emb2_norm = emb2 / norm2
            cosine_similarity = np.dot(emb1_norm, emb2_norm)
            
            if np.isfinite(cosine_similarity):
                similarities.append(cosine_similarity)
                labels.append(label)
            else:
                logging.warning(f"Invalid similarity computed: {cosine_similarity}")
    
    return similarities, labels

def _calculate_similarity_for_pair(pair):
    """한 쌍의 이미지에 대한 유사도를 계산합니다. (워커 프로세스용)"""
    # init_worker에 의해 설정된 전역 변수 embeddings를 사용합니다.
    global embeddings
    img1_path, img2_path = pair
    
    emb1 = embeddings.get(img1_path)
    emb2 = embeddings.get(img2_path)
    
    if emb1 is not None and emb2 is not None:
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 > 0 and norm2 > 0:
            # 정규화와 내적을 한 번에 계산
            cosine_similarity = np.dot(emb1, emb2) / (norm1 * norm2)
            if np.isfinite(cosine_similarity):
                return cosine_similarity
    return None # 계산 실패 시 None 반환

def calculate_identification_metrics(identity_map, embeddings ):
    logging.info("Calculating identification metrics (Rank-k, CMC)...")

    gallery_images = {} # {identity: image_path}
    probe_images_with_labels = [] # [(image_path, identity)]

    # Split data into gallery and probe sets
    # For each identity, take one image for gallery, rest for probes
    for identity, img_paths in identity_map.items():
        if not img_paths:
            continue
        
        # Use the first image as gallery representative
        try:
            gallery_images[identity] = img_paths[2098]
            idx = 2098 
        except:
            logging.info("대표이미지 0으로 설정함 top k 부정확")
            gallery_images[identity] = img_paths[0]
            idx = 0

        for i in range(0, len(img_paths)):
            if i == idx:
                continue
            probe_images_with_labels.append((img_paths[i], identity))  # 이미지와 해당사람 클래스
    
    
    if not probe_images_with_labels:
        logging.warning("No probe images available for identification evaluation. Skipping identification metrics.")
        return None, None, None, None, None

    logging.info(f"Identities in gallery: {len(gallery_images)}")
    logging.info(f"Total probe images: {len(probe_images_with_labels)}")

    # Prepare gallery embeddings
    gallery_embeddings = [] # list of (embedding, identity)
    gallery_identities_ordered = [] # ordered list of identities corresponding to gallery_embeddings
    for identity in sorted(gallery_images.keys()): # Sort to ensure consistent order
        img_path = gallery_images[identity]
        emb = embeddings.get(img_path) # 대표이미지 임베딩값 추출
        if emb is not None:
            gallery_embeddings.append(emb / np.linalg.norm(emb)) # Normalize
            gallery_identities_ordered.append(identity)
        else:
            logging.warning(f"Gallery image embedding missing for identity {identity}: {img_path}")
    
    if not gallery_embeddings:
        logging.error("No valid gallery embeddings found. Cannot perform identification evaluation.")
        return None, None, None, None, None

    gallery_embeddings_np = np.array(gallery_embeddings)

    # Max rank for CMC curve
    max_rank = len(gallery_identities_ordered) # Max possible rank is number of identities in gallery
    if max_rank == 0: # Avoid division by zero if no gallery
        logging.error("Gallery is empty. Cannot calculate identification metrics.")
        return None, None, None, None, None
    
    cmc_hits = np.zeros(max_rank, dtype=int)
    total_probes = 0

    rank_1_correct = 0
    rank_5_correct = 0
    
    for probe_img_path, true_identity in tqdm(probe_images_with_labels, desc="Evaluating identification"):
        probe_emb = embeddings.get(probe_img_path) # 추측 임베딩 추출
        if probe_emb is None:
            logging.warning(f"Probe image embedding missing: {probe_img_path}. Skipping.")
            continue
        
        probe_emb_norm = probe_emb / np.linalg.norm(probe_emb)

        # Calculate similarities with all gallery embeddings
        similarities = np.dot(gallery_embeddings_np, probe_emb_norm)# (class , 512 )  dot (512 ,)= (class, 1) -> 에측이미지에대하여 대표이미지 전부 유사한정도 구하기 
        
        # Get ranks (indices of sorted similarities in descending order)
        # argsort returns indices that would sort an array in ascending order.
        # To get descending, we can negate the similarities and then argsort.
        ranked_indices = np.argsort(similarities)[::-1] 
        
        # Find the rank of the true identity
        true_identity_rank = -1
        for rank, idx in enumerate(ranked_indices):
            if gallery_identities_ordered[idx] == true_identity:
                true_identity_rank = rank + 1 # Rank is 1-based
                break
        
        if true_identity_rank != -1:
            # Update CMC hits
            for r in range(true_identity_rank, max_rank + 1):
                cmc_hits[r-1] += 1 # cmc_hits is 0-indexed
            
            # Update Rank-1 and Rank-5
            if true_identity_rank == 1:
                rank_1_correct += 1
            if true_identity_rank <= 5:
                rank_5_correct += 1
        
        total_probes += 1

    if total_probes == 0:
        logging.warning("No valid probes processed for identification evaluation.")
        return None, None, None, None, None

    rank_1_accuracy = rank_1_correct / total_probes
    rank_5_accuracy = rank_5_correct / total_probes
    
    cmc_curve = cmc_hits / total_probes

    logging.info(f"Rank-1 Accuracy: {rank_1_accuracy:.4f}")
    logging.info(f"Rank-5 Accuracy: {rank_5_accuracy:.4f}")
    logging.info(f"CMC Curve calculated up to rank {max_rank}")

    return rank_1_accuracy, rank_5_accuracy, cmc_curve, max_rank, total_probes

def main(args):
    LOG_FILE = os.path.join(script_dir , f'{args.model}_LOG.log')
    torch.backends.cudnn.benchmark = True
    np.random.seed(42)
    random.seed(42)
    logging.basicConfig(
        filename=LOG_FILE, level=logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s', filemode='w'
    )
    
    if not os.path.isdir(args.data_path):
        raise FileNotFoundError(f"데이터셋 경로를 찾을 수 없습니다: {args.data_path}")

    if args.model =='Glint360K_R200_TopoFR':
        Weight_path = 'Glint360K_R200_TopoFR_9784.pt'
        backbone = IResNet(IBasicBlock, [6, 26, 60, 6] , num_classes=360232)

    elif args.model == 'MS1MV2_R200_TopoFR':
        Weight_path = 'MS1MV2_R200_TopoFR_9712_cosface.pt'
        backbone = IResNet(IBasicBlock, [6, 26, 60, 6] , num_classes=85742)

    elif args.model == 'Glint360K_R100_TopoFR_9760':
        Weight_path = 'Glint360K_R100_TopoFR_9760.pt'
        backbone = IResNet(IBasicBlock, [3, 13, 30, 3] , num_classes=360232)

    elif args.model == 'Glint360K_R50_TopoFR_9727':
        Weight_path = 'Glint360K_R50_TopoFR_9727.pt'
        backbone = IResNet(IBasicBlock , [3,4,14,3] , num_classes= 360232)

    else:
        logging.info(f"Select Model {args.model}")
        exit(0)



    load_result = backbone.load_state_dict(torch.load(Weight_path, map_location='cpu'), strict=False)
    print("누락된 가중치 : {}".format(load_result.missing_keys))
    print("예상치못한 가중치 : {}".format(load_result.unexpected_keys))

    if not load_result.missing_keys and not load_result.unexpected_keys:
        print("모델 가중치가 성공적으로 로드되었습니다.")

    flag = str(input("진행시 아무키... (1)종료..."))
    if flag == '1':
        logging.info("종료../")
        exit(0)

    backbone = torch.compile(backbone)

    identity_map = {} # 사람폴더라벨 : 해당 폴더 사람 이미지 경로
    for person_folder in sorted(os.listdir(args.data_path)):
        person_path = os.path.join(args.data_path, person_folder) # 각 사람폴더의 경로
        if os.path.isdir(person_path):
            images = [os.path.join(person_path, f) for f in os.listdir(person_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))] # 각사람폴더에 들어있는 모든 jpg
            if len(images) > 1:
                identity_map[person_folder] = images
    
    if not identity_map:
        raise ValueError("데이터셋에서 2개 이상의 이미지를 가진 인물을 찾지 못했습니다.")
    print(f"총 {len(identity_map)}명의 인물, {sum(len(v) for v in identity_map.values())}개의 이미지를 찾았습니다.")

    print("\n평가에 사용할 동일 인물/다른 인물 쌍을 생성합니다...")
    
    positive_pairs = []
    for imgs in tqdm(identity_map.values(), desc="동일 인물 쌍 생성"):
        positive_pairs.extend(itertools.combinations(imgs, 2))

    num_positive_pairs = len(positive_pairs)


    identities = list(identity_map.keys())
    negative_pairs_set = set()
    if len(identities) > 1:
        with tqdm(total=num_positive_pairs, desc="다른 인물 쌍 생성") as pbar:
            while len(negative_pairs_set) < num_positive_pairs:
                id1, id2 = random.sample(identities, 2)
                pair = (random.choice(identity_map[id1]), random.choice(identity_map[id2]))
                sorted_pair = tuple(sorted(pair))
                if sorted_pair not in negative_pairs_set:
                    negative_pairs_set.add(sorted_pair)
                    pbar.update(1)
    negative_pairs = list(negative_pairs_set)

    print(f"- 동일 인물 쌍: {len(positive_pairs)}개, 다른 인물 쌍: {len(negative_pairs)}개")


    if args.load_cache is not None :
        cache_path = args.load_cache
        loaded_npz = np.load(cache_path)
        embeddings = {key: loaded_npz[key] for key in tqdm(loaded_npz.files , desc='임베딩 캐시 로딩..')}
        embeddings = loaded_npz

    else:
        embeddings = get_all_embeddings(
            identity_map, backbone, args.device,args.batch_size
        )

    with Pool(initializer=init_worker, initargs=(embeddings,)) as pool:
        # 1. 동일 인물 쌍 계산
        pos_results = list(tqdm(pool.imap_unordered(_calculate_similarity_for_pair, positive_pairs , chunksize= 1000), 
                                total=len(positive_pairs), 
                                desc="동일 인물 쌍 계산"))
        pos_similarities = [r for r in pos_results if r is not None]
        pos_labels = [1] * len(pos_similarities)

        # 2. 다른 인물 쌍 계산
        neg_results = list(tqdm(pool.imap_unordered(_calculate_similarity_for_pair, negative_pairs , chunksize = 1000), 
                                total=len(negative_pairs), 
                                desc="다른 인물 쌍 계산"))
        neg_similarities = [r for r in neg_results if r is not None]
        neg_labels = [0] * len(neg_similarities)

    print(f"🔍 디버깅 정보:")
    print(f"   - 전체 임베딩 수: {len(embeddings)}")
    print(f"   - 유효한 임베딩 수: {sum(1 for v in embeddings.values() if v is not None)}")
    print(f"   - None 임베딩 수: {sum(1 for v in embeddings.values() if v is None)}")
    print(f"   - 양성 쌍 유사도 수 (변환 전): {len(pos_similarities)}")
    print(f"   - 음성 쌍 유사도 수 (변환 전): {len(neg_similarities)}")
    
    pos_similarities_array = np.array(pos_similarities)
    neg_similarities_array = np.array(neg_similarities)
    
    print(f"   - NaN 개수 (양성/음성): {np.isnan(pos_similarities_array).sum()} / {np.isnan(neg_similarities_array).sum()}")
    print(f"   - Inf 개수 (양성/음성): {np.isinf(pos_similarities_array).sum()} / {np.isinf(neg_similarities_array).sum()}")

    pos_similarities = pos_similarities_array
    neg_similarities = neg_similarities_array
    pos_labels = np.array(pos_labels)
    neg_labels = np.array(neg_labels)

    pos_finite_mask = np.isfinite(pos_similarities)
    neg_finite_mask = np.isfinite(neg_similarities)

    pos_similarities = pos_similarities[pos_finite_mask]
    neg_similarities = neg_similarities[neg_finite_mask]
    pos_labels = pos_labels[pos_finite_mask]
    neg_labels = neg_labels[neg_finite_mask]


    with open(LOG_FILE, 'a') as log_file:
        log_file.write(f"현재시각 : {datetime.now().strftime('%Y.%m.%d - %H:%M:%S')}\n")
        log_file.write(f"\n🔍 디버깅 정보:\n")
        log_file.write(f"   - 전체 임베딩 수: {len(embeddings)}\n")
        log_file.write(f"   - 유효한 임베딩 수: {sum(1 for v in embeddings.values() if v is not None)}\n")
        log_file.write(f"   - None 임베딩 수: {sum(1 for v in embeddings.values() if v is None)}\n")
        log_file.write(f"   - 양성 쌍 유사도 수 (필터링 후): {len(pos_similarities)}\n")
        log_file.write(f"   - 음성 쌍 유사도 수 (필터링 후): {len(neg_similarities)}\n")

    print(f"\n--- 유사도 분포 분석 ---")
    if len(pos_similarities) > 0 and len(neg_similarities) > 0:
        def safe_std(arr):
            try:
                arr_f64 = arr.astype(np.float64)
                std_val = np.std(arr_f64, dtype=np.float64)
                return std_val if np.isfinite(std_val) else "overflow"
            except:
                return "계산 불가"
        
        pos_std = safe_std(pos_similarities)
        neg_std = safe_std(neg_similarities)
        
        print(f"🔵 동일 인물 쌍 유사도 (총 {len(pos_similarities):,}개):")
        print(f"   - 최소값: {np.min(pos_similarities):.4f}")
        print(f"   - 최대값: {np.max(pos_similarities):.4f}")
        print(f"   - 평균값: {np.mean(pos_similarities.astype(np.float64)):.4f}")
        print(f"   - 표준편차: {pos_std:.4f}" if isinstance(pos_std, (int, float)) else f"   - 표준편차: {pos_std}")
        
        print(f"🔴 다른 인물 쌍 유사도 (총 {len(neg_similarities):,}개):")
        print(f"   - 최소값: {np.min(neg_similarities):.4f}")
        print(f"   - 최대값: {np.max(neg_similarities):.4f}")
        print(f"   - 평균값: {np.mean(neg_similarities.astype(np.float64)):.4f}")
        print(f"   - 표준편차: {neg_std:.4f}" if isinstance(neg_std, (int, float)) else f"   - 표준편차: {neg_std}")


        with open(LOG_FILE, 'a') as log_file:
            log_file.write(f"\n--- 유사도 분포 분석 ---\n")
            log_file.write(f"🔵 동일 인물 쌍 유사도 (총 {len(pos_similarities):,}개):\n")
            log_file.write(f"   - 최소값: {np.min(pos_similarities):.4f}\n")
            log_file.write(f"   - 최대값: {np.max(pos_similarities):.4f}\n")
            log_file.write(f"   - 평균값: {np.mean(pos_similarities.astype(np.float64)):.4f}\n")
            log_file.write(f"   - 표준편차: {pos_std:.4f}\n" if isinstance(pos_std, (int, float)) else f"   - 표준편차: {pos_std}\n")
            
            log_file.write(f"🔴 다른 인물 쌍 유사도 (총 {len(neg_similarities):,}개):\n")
            log_file.write(f"   - 최소값: {np.min(neg_similarities):.4f}\n")
            log_file.write(f"   - 최대값: {np.max(neg_similarities):.4f}\n")
            log_file.write(f"   - 평균값: {np.mean(neg_similarities.astype(np.float64)):.4f}\n")
            log_file.write(f"   - 표준편차: {neg_std:.4f}\n" if isinstance(neg_std, (int, float)) else f"   - 표준편차: {neg_std}\n")
    else:
        print("유사도 데이터가 충분하지 않아 분포 분석을 수행할 수 없습니다.")
        print(f"len pos : {len(pos_similarities)}, len neg: {len(neg_similarities)}")
        exit(0)
    
    scores = np.concatenate([pos_similarities, neg_similarities])
    labels = np.concatenate([pos_labels, neg_labels])

    print("\n--- 최종 평가 결과 ---")
    if labels.size > 0:
        fpr, tpr, thresholds = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        
        frr = 1 - tpr
        eer_index = np.nanargmin(np.abs(fpr - frr))
        eer = fpr[eer_index]
        eer_threshold = thresholds[eer_index]

        tar_at_far_results = {far: np.interp(far, fpr, tpr) for far in args.target_fars}
        
        predictions = (scores >= eer_threshold).astype(int)
        cm = confusion_matrix(labels, predictions)
        
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)

        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        metrics = {"accuracy": accuracy, "recall": recall, "f1_score": f1_score, "tp": tp, "tn": tn, "fp": fp, "fn": fn}

        print(f"전체 평가 쌍: {len(labels)} 개")
        print(f"[주요 성능] ROC-AUC: {roc_auc:.4f}, EER: {eer:.4f} (유사도 임계값: {eer_threshold:.4f})")
        print(f"[상세 지표] Accuracy: {metrics['accuracy']:.4f}, Recall: {metrics['recall']:.4f}, F1-Score: {metrics['f1_score']:.4f}")
        for far, tar in tar_at_far_results.items():
            print(f"  - TAR @ FAR {far*100:g}%: {tar:.4f}")
        
        with open(LOG_FILE, 'a') as log_file:
            log_file.write(f"\n평가 결과:\n")
            log_file.write(f"ROC-AUC: {roc_auc:.4f}, EER: {eer:.4f} (Threshold: {eer_threshold:.4f})\n")
            log_file.write(f"Accuracy: {metrics['accuracy']:.4f}, Recall: {metrics['recall']:.4f}, F1-Score: {metrics['f1_score']:.4f}\n")
            for far, tar in tar_at_far_results.items():
                log_file.write(f"TAR @ FAR {far*100:g}%: {tar:.4f}\n")
            log_file.write("\n")  # 빈 줄 추가

        excel_path = os.path.join(script_dir, args.excel_path)
        total_dataset_img_len = sum(len(v) for v in identity_map.values())
        total_class = len(identity_map)
        
        # Calculate identification metrics
        rank_1_accuracy, rank_5_accuracy, cmc_curve, max_rank, total_probes = calculate_identification_metrics(identity_map, embeddings)

        if rank_1_accuracy is not None:
            print(f"\n--- 얼굴 식별 성능 ---")
            print(f"Rank-1 Accuracy: {rank_1_accuracy:.4f}")
            print(f"Rank-5 Accuracy: {rank_5_accuracy:.4f}")
            print(f"총 프로브 이미지 수: {total_probes}")

            with open(LOG_FILE, 'a') as log_file:
                log_file.write(f"\n얼굴 식별 성능:\n")
                log_file.write(f"Rank-1 Accuracy: {rank_1_accuracy:.4f}\n")
                log_file.write(f"Rank-5 Accuracy: {rank_5_accuracy:.4f}\n")
                log_file.write(f"총 프로브 이미지 수: {total_probes}\n")
                if cmc_curve is not None:
                    log_file.write(f"CMC Curve (first 10 ranks): {cmc_curve[:10].tolist()}\n")
                log_file.write("\n")

            # Plot CMC Curve
            if cmc_curve is not None and max_rank > 0:
                plt.figure(figsize=(8, 6))
                plt.plot(np.arange(1, max_rank + 1), cmc_curve, marker='o', linestyle='-', markersize=4)
                plt.xlim([1, min(max_rank, 20)]) # Show up to rank 20 or max_rank
                plt.ylim([0.0, 1.05])
                plt.xlabel('Rank (k)')
                plt.ylabel('Accuracy')
                plt.title(f'CMC Curve for {args.model}')
                plt.grid(True)
                cmc_plot_filename = os.path.splitext(excel_path)[0] + f"_{args.model}_cmc_curve.png"
                plt.savefig(cmc_plot_filename)
                print(f"CMC 커브 그래프가 '{cmc_plot_filename}' 파일로 저장되었습니다.")
        else:
            print("\n--- 얼굴 식별 성능 ---")
            print("얼굴 식별 성능을 계산할 수 없습니다 (유효한 프로브 이미지 부족).")
            with open("LOG_FILE", 'a') as log_file:
                log_file.write("\n얼굴 식별 성능:\n")
                log_file.write("얼굴 식별 성능을 계산할 수 없습니다 (유효한 프로브 이미지 부족).\n")
                log_file.write("\n")

        save_results_to_excel(excel_path, args.model, roc_auc, eer, tar_at_far_results, \
                              args.target_fars, metrics, total_dataset_img_len, total_class, args.data_path, args.model,
                              rank_1_accuracy, rank_5_accuracy)

        plot_roc_curve(fpr, tpr, roc_auc, args.model, excel_path)
    else:
        msg = "평가를 위한 유효한 점수를 수집하지 못했습니다."
        print(msg)
        logging.error(msg)

def save_results_to_excel(excel_path, model_name, roc_auc, eer, tar_at_far_results, target_fars, metrics, total_dataset_img_len, total_class,
                           data_path, model_attr_value, rank_1_accuracy=None, rank_5_accuracy=None):
    """결과를 Excel 파일에 저장합니다."""
    new_data = {
        "model_name": [model_name],
        "roc_auc": [f"{roc_auc:.4f}"], "eer": [f"{eer:.4f}"],
        "accuracy": [f"{metrics['accuracy']:.4f}"], "recall": [f"{metrics['recall']:.4f}"],
        "f1_score": [f"{metrics['f1_score']:.4f}"], "tp": [metrics['tp']],
        "tn": [metrics['tn']], "fp": [metrics['fp']], "fn": [metrics['fn']]
    }

    for far in target_fars:
        new_data[f"tar_at_far_{far*100:g}%"] = [f"{tar_at_far_results.get(far, 0):.4f}"]

    if rank_1_accuracy is not None:
        new_data["rank_1_accuracy"] = [f"{rank_1_accuracy:.4f}"]
    if rank_5_accuracy is not None:
        new_data["rank_5_accuracy"] = [f"{rank_5_accuracy:.4f}"]

    new_data.update({
        'total_dataset_img_len': [total_dataset_img_len],
        'total_class': [total_class],
        'data_path': [data_path],
        'model_attr': [model_attr_value]
    })
    
    new_df = pd.DataFrame(new_data)
    try:
        df = pd.read_excel(excel_path)
    except FileNotFoundError:
        df = pd.DataFrame()

    updated_df = pd.concat([df, new_df], ignore_index=True)
    updated_df.to_excel(excel_path, index=False)
    print(f"\n평가 결과가 '{excel_path}' 파일에 저장되었습니다.")

def plot_roc_curve(fpr, tpr, roc_auc, model_name, excel_path):
    """ROC 커브를 그리고 파일로 저장합니다."""

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FAR)')
    plt.ylabel('True Positive Rate (TAR)')
    plt.title(f'ROC Curve {model_name}')
    plt.legend(loc="lower right")
    plt.grid(True)
    plot_filename = os.path.splitext(excel_path)[0] + f"_{model_name}_roc_curve.png"
    plt.savefig(plot_filename)
    print(f"ROC 커브 그래프가 '{plot_filename}' 파일로 저장되었습니다.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SEvaluation Script")
    parser.add_argument('--model',type=str , default='Glint360K_R50_TopoFR_9727', choices=['Glint360K_R50_TopoFR_9727, Glint360K_R200_TopoFR', 'MS1MV2_R200_TopoFR', 'Glint360K_R100_TopoFR_9760'],)
    parser.add_argument("--data_path", type=str, default="/home/ubuntu/KOR_DATA/일반/kor_data_sorting", help="평가할 데이터셋의 루트 폴더")
    parser.add_argument("--excel_path", type=str, default="evaluation_results.xlsx", help="결과를 저장할 Excel 파일 이름")
    parser.add_argument("--target_fars", nargs='+', type=float, default=[0.01, 0.001, 0.0001], help="TAR을 계산할 FAR 목표값들")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="사용할 장치 (예: cpu, cuda, cuda:0)")
    parser.add_argument("--batch_size", type=int, default=512, help="임베딩 추출 시 배치 크기")
    parser.add_argument('--load_cache' , type=str , default = None ,help="임베딩 캐시경로")
    args = parser.parse_args()

    for key , values in args.__dict__.items():
        print(f"key {key}  :  {values}")

    main(args)