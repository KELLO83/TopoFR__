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
import argparse
import matplotlib.pyplot as plt
import logging
import torchvision.transforms.v2 as v2
import numpy as np
from backbones.iresnet import IResNet , IBasicBlock
from multiprocessing.pool import Pool



import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

def setup_ddp(rank , world_size):

    """rank gpu번호  wolrd --> 프로세스 수 """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12355'

    torch.distributed.init_process_group(
        backend='nccl',
        rank=rank,
        world_size=world_size
    )

    torch.cuda.set_device(rank)

    print(f"Multi GPU SETUP - RANK: {rank}, WORLD_SIZE: {world_size}")
    return rank

def cleanup_ddp():
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()





try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    script_dir = os.getcwd()


transforms_v2 = v2.Compose([
    v2.ToImage(), # cv2 HWC ---> Tensor C H W
    v2.ToDtype(torch.float32, scale=True),
    v2.CenterCrop(size=(112, 112)),
    v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

@torch.inference_mode()
def get_all_embeddings(rank, world_size, identity_map, backbone, batch_size, ):
    """
    Performs distributed embedding extraction using DDP.
    Each rank processes a shard of the data, and rank 0 gathers all results
    and saves them to a temporary file.
    """
    setup_ddp(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    logging.info(f"Starting embedding extraction on RANK {rank} for {world_size} GPUs.")

    backbone = backbone.to(device)
    backbone.eval()
    
    local_embeddings = {}  # Embeddings for the current rank's data shard
    all_images = sorted(list(set(itertools.chain.from_iterable(identity_map.values()))))

    # --- Data Sharding: Each rank gets a unique portion of the data ---
    my_images = np.array_split(all_images, world_size)[rank]
    
    pbar = None
    if rank == 0:
        pbar = tqdm(total=len(my_images), desc=f'Embedding Extraction (GPUs: {world_size})')

    def preprocess_image(image):
        return transforms_v2(image)

    for i in range(0, len(my_images), batch_size):
        batch_paths = my_images[i:i+batch_size].tolist()
        batch_images, valid_paths = [], []

        for img_path in batch_paths:
            try:
                image = cv2.imread(img_path)
                if image is None:
                    logging.warning(f"[{rank}] Image is empty: {img_path}")
                    local_embeddings[img_path] = None
                    continue
                
                if image.shape[0] != 112 or image.shape[1] != 112:
                    image = cv2.resize(image, (112, 112), interpolation=cv2.INTER_CUBIC if image.shape[0] > 112 else cv2.INTER_AREA)
                
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                processed_image = preprocess_image(image_rgb)
                batch_images.append(processed_image)
                valid_paths.append(img_path)
            except Exception as e:
                logging.warning(f"[{rank}] Image processing failed: {img_path}, Error: {e}")
                local_embeddings[img_path] = None

        if not batch_images:
            if pbar: pbar.update(len(batch_paths))
            continue

        batch_tensor = torch.stack(batch_images).to(device)

        try:
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                output = backbone(batch_tensor)
                vectors = output[1] if isinstance(output, tuple) else output

            if vectors is None or vectors.numel() == 0:
                logging.warning(f"[{rank}] Vector extraction failed for batch size: {len(batch_paths)}")
                for path in valid_paths: local_embeddings[path] = None
            else:
                vectors_cpu = vectors.cpu().numpy()
                for path, vector in zip(valid_paths, vectors_cpu):
                    local_embeddings[path] = vector.flatten()
        except Exception as e:
            logging.warning(f"[{rank}] Embedding extraction failed: {e}")
            for path in valid_paths: local_embeddings[path] = None
        
        if pbar: 
            pbar.update(len(batch_paths))

    if pbar: pbar.close()

    # --- Gather results from all GPUs to Rank 0 ---
    dist.barrier()

    if rank == 0:
        logging.info("Gathering embeddings from all GPUs...")
        gathered_list = [None] * world_size
        dist.gather_object(local_embeddings, gathered_list, dst=0)
        
        logging.info("Combining embeddings...")
        final_embeddings = {}
        for d in gathered_list:
            if d: final_embeddings.update(d)
        
        logging.info(f"Total embeddings combined: {len(final_embeddings)}. Saving to temp file...")
        np.savez_compressed('emb.npz', **final_embeddings)


    else: # Other ranks send their data
        dist.gather_object(local_embeddings, None, dst=0)

    cleanup_ddp()
    

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


def init_worker(worker_embeddings):
    """워커 프로세스 초기화 함수. embeddings 딕셔너리를 전역 변수로 설정합니다."""
    global embeddings
    embeddings = worker_embeddings


def collect_scores_from_embeddings(pairs, is_positive, total_pairs=None):
    """임베딩으로 유사도를 계산합니다 (코사인 유사도 사용). 워커 프로세스에서 실행됩니다."""
    global embeddings  # 워커의 전역 변수인 embeddings에 접근합니다.
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


def init_identification_worker(worker_embeddings, gallery_embs_np, gallery_ids):
    """워커 프로세스 초기화 함수. 식별 평가에 필요한 모든 데이터를 전역 변수로 설정합니다."""
    global g_embeddings, g_gallery_embeddings_np, g_gallery_identities_ordered
    g_embeddings = worker_embeddings
    g_gallery_embeddings_np = gallery_embs_np
    g_gallery_identities_ordered = gallery_ids


def _evaluate_probe_worker(probe_data):
    """단일 프로브를 처리하는 워커 함수 (전역 데이터를 사용)."""
    global g_embeddings, g_gallery_embeddings_np, g_gallery_identities_ordered
    probe_img_path, true_identity = probe_data

    probe_emb = g_embeddings.get(probe_img_path)
    if probe_emb is None:
        return -1  # Indicate skipped probe

    norm_val = np.linalg.norm(probe_emb)
    if norm_val == 0:
        return -1
    probe_emb_norm = probe_emb / norm_val

    if not np.all(np.isfinite(probe_emb_norm)):
        return -1

    # Calculate similarities and rank
    similarities = np.dot(g_gallery_embeddings_np, probe_emb_norm)
    ranked_indices = np.argsort(similarities)[::-1]
    
    ranked_identities = np.array(g_gallery_identities_ordered)[ranked_indices]
    match_indices = np.where(ranked_identities == true_identity)[0]

    if len(match_indices) > 0:
        return match_indices[0] + 1  # Return 1-based rank
    else:
        return -1 # Not found


def calculate_identification_metrics(identity_map, embeddings):
    """
    Calculates identification metrics (Rank-k, CMC) in parallel.
    """
    logging.info("Calculating identification metrics (Rank-k, CMC) using multiprocessing...")

    gallery_images = {}  # {identity: image_path}
    probe_images_with_labels = []  # [(image_path, identity)]

    # Split data into gallery and probe sets
    for identity, img_paths in identity_map.items():
        if not img_paths:
            continue
        
        try:
            # Use a specific image for the gallery, fallback to the first one
            gallery_images[identity] = img_paths[2098]
        except IndexError:
            logging.info(f"Identity {identity} has fewer than 2099 images; using the first image for the gallery.")
            gallery_images[identity] = img_paths[0]

        # Remaining images are probes
        for i in range(1, len(img_paths)):
            probe_images_with_labels.append((img_paths[i], identity))

    if not probe_images_with_labels:
        logging.warning("No probe images available for identification evaluation.")
        return None, None, None, None, None

    logging.info(f"Identities in gallery: {len(gallery_images)}")
    logging.info(f"Total probe images: {len(probe_images_with_labels)}")

    # Prepare gallery embeddings
    gallery_embeddings = []
    gallery_identities_ordered = []
    for identity in sorted(gallery_images.keys()):
        img_path = gallery_images[identity]
        emb = embeddings.get(img_path)
        if emb is not None:
            norm = np.linalg.norm(emb)
            if norm > 0:
                norm_emb = emb / norm
                if np.all(np.isfinite(norm_emb)):
                    gallery_embeddings.append(norm_emb)
                    gallery_identities_ordered.append(identity)
                else:
                    logging.warning(f"Non-finite gallery embedding for identity {identity}: {img_path}")
            else:
                logging.warning(f"Zero-norm gallery embedding for identity {identity}: {img_path}")
        else:
            logging.warning(f"Gallery image embedding missing for identity {identity}: {img_path}")

    if not gallery_embeddings:
        logging.error("No valid gallery embeddings found.")
        return None, None, None, None, None

    gallery_embeddings_np = np.array(gallery_embeddings)
    max_rank = len(gallery_identities_ordered)
    if max_rank == 0:
        logging.error("Gallery is empty.")
        return None, None, None, None, None

    # --- Parallel Evaluation ---
    from multiprocessing import Pool, cpu_count

    # Set up arguments for the worker initializer
    init_args = (embeddings, gallery_embeddings_np, gallery_identities_ordered)

    all_ranks = []
    with Pool(initializer=init_identification_worker, initargs=init_args, processes=cpu_count()) as pool:
        results_iterator = pool.imap(_evaluate_probe_worker, probe_images_with_labels)
        all_ranks = list(tqdm(results_iterator, total=len(probe_images_with_labels), desc="Evaluating identification (multi-process)"))

    # --- Aggregate Results ---
    valid_ranks = [r for r in all_ranks if r > 0]
    total_probes = len(valid_ranks)

    if total_probes == 0:
        logging.warning("No valid probes were processed.")
        return None, None, None, None, None

    valid_ranks_np = np.array(valid_ranks)
    
    # Efficiently calculate metrics using numpy
    rank_counts = np.bincount(valid_ranks_np, minlength=max_rank + 1)
    
    rank_1_accuracy = rank_counts[1] / total_probes
    rank_5_accuracy = np.sum(rank_counts[1:6]) / total_probes
    
    cmc_hits = np.cumsum(rank_counts[1:max_rank + 1])
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
    ngpus_per_node = torch.cuda.device_count()

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
    for person_folder in os.listdir(args.data_path):
        person_path = os.path.join(args.data_path, person_folder) # 각 사람폴더의 경로
        if os.path.isdir(person_path):
            images = [os.path.join(person_path, f) for f in os.listdir(person_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))] # 각사람폴더에 들어있는 모든 jpg
            if len(images) > 1:
                identity_map[person_folder] = images
    
    if not identity_map:
        raise ValueError("데이터셋에서 2개 이상의 이미지를 가진 인물을 찾지 못했습니다.")
    print(f"총 {len(identity_map)}명의 인물, {sum(len(v) for v in identity_map.values())}개의 이미지를 찾았습니다.")

    print("\n평가에 사용할 동일 인물/다른 인물 쌍을 생성합니다...")

    num_positive_pairs = sum(len(imgs) * (len(imgs) - 1) // 2 for imgs in identity_map.values())
    positive_pairs_generator = itertools.chain.from_iterable(
        itertools.combinations(imgs, 2) for imgs in identity_map.values()
    )

    num_negative_pairs = num_positive_pairs
    
    def negative_pairs_generator_func(identity_map, num_pairs_to_generate):
        identities = list(identity_map.keys())
        if len(identities) < 2:
            return

        seen_pairs = set()
        
        for _ in range(num_pairs_to_generate):
            while True:
                id1, id2 = random.sample(identities, 2)
                img1_path = random.choice(identity_map[id1])
                img2_path = random.choice(identity_map[id2])

                sorted_pair = tuple(sorted((img1_path, img2_path)))
                
                if sorted_pair not in seen_pairs:
                    seen_pairs.add(sorted_pair)

                    if len(seen_pairs) > 2000000: # 메모리 상황에 따라 조절 가능
                        seen_pairs.clear()
                    yield (img1_path, img2_path)
                    break

    negative_pairs_generator = negative_pairs_generator_func(identity_map, num_negative_pairs)
    
    print(f"- 동일 인물 쌍: {num_positive_pairs}개, 다른 인물 쌍: {num_negative_pairs}개 (생성 예정)")


    if args.load_cache is not None :
        cache_path = args.load_cache
        loaded_npz = np.load(cache_path)
        embeddings = {key: loaded_npz[key] for key in loaded_npz.files}
        embeddings = loaded_npz

    else:
        temp_filepath = os.path.join(script_dir, f'temp_embeddings_{os.getpid()}.npz')
        if ngpus_per_node <= 1:
            # Fallback for single GPU, though spawn is preferred
            get_all_embeddings(0, 1, identity_map, backbone, args.batch_size, temp_filepath)
        else:
            logging.info(f"Spawning {ngpus_per_node} processes for distributed inference.")
            # Note: DDP model wrapping is not needed for inference if each process has its own model copy
            torch.multiprocessing.spawn(
                get_all_embeddings,
                nprocs=ngpus_per_node,
                args=(ngpus_per_node, identity_map, backbone, args.batch_size, temp_filepath),
                join=True
            )

        print(f"Loading combined embeddings from {temp_filepath}")
        try:
            embeddings_npz = np.load(temp_filepath)
            embeddings = {key: embeddings_npz[key] for key in embeddings_npz.files}
            os.remove(temp_filepath) # Clean up the temporary file
            print("Successfully loaded and cleaned up temporary embeddings file.")
        except FileNotFoundError:
            print("Error: Temporary embeddings file not found. Inference might have failed.")
            exit(1)
        except Exception as e:
            print(f"Error loading temporary embeddings file: {e}")
            exit(1)

    def chunk_iterable(iterable, chunk_size):
        it = iter(iterable)
        while True:
            chunk = list(itertools.islice(it, chunk_size))
            if not chunk:
                break
            yield chunk

    # 각 워커에게 한 번에 보낼 쌍의 수입니다. 시스템의 메모리와 CPU 코어 수에 따라 조절할 수 있습니다.
    CHUNK_SIZE = 200000  
    
    pos_similarities, pos_labels = [], []
    neg_similarities, neg_labels = [], []

    # Pool을 생성하고, init_worker 함수를 initializer로 설정합니다.
    # initargs에는 init_worker에 전달할 인자(embeddings 딕셔너리)를 튜플로 제공합니다.
    with Pool(initializer=init_worker, initargs=(embeddings,)) as pool:
        # 1. 동일 인물 쌍 처리
        print("동일 인물 쌍 처리 중...")
        # positive_pairs_generator를 청크 단위로 나눕니다.
        positive_chunks = chunk_iterable(positive_pairs_generator, CHUNK_SIZE)
        
        # starmap을 사용하여 각 청크와 is_positive=True 인자를 워커의 collect_scores_from_embeddings 함수에 전달합니다.
        pos_results = pool.starmap(
            collect_scores_from_embeddings,
            [(chunk, True) for chunk in positive_chunks]
        )
        
        # 모든 워커로부터 반환된 결과를 취합합니다.
        for sims, labs in pos_results:
            pos_similarities.extend(sims)
            pos_labels.extend(labs)

        # 2. 다른 인물 쌍 처리
        print("다른 인물 쌍 처리 중...")
        negative_chunks = chunk_iterable(negative_pairs_generator, CHUNK_SIZE)
        neg_results = pool.starmap(
            collect_scores_from_embeddings,
            [(chunk, False) for chunk in negative_chunks]
        )
        
        # 결과 취합
        for sims, labs in neg_results:
            neg_similarities.extend(sims)
            neg_labels.extend(labs)


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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SEvaluation Script")
    parser.add_argument('--model',type=str , default='Glint360K_R100_TopoFR_9760', choices=['Glint360K_R200_TopoFR', 'MS1MV2_R200_TopoFR', 'Glint360K_R100_TopoFR_9760'],)
    parser.add_argument("--data_path", type=str, default="/home/ubuntu/KOR_DATA/kor_data_full_Middle_aligend", help="평가할 데이터셋의 루트 폴더")
    parser.add_argument("--excel_path", type=str, default="evaluation_results.xlsx", help="결과를 저장할 Excel 파일 이름")
    parser.add_argument("--target_fars", nargs='+', type=float, default=[0.01, 0.001, 0.0001], help="TAR을 계산할 FAR 목표값들")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="사용할 장치 (예: cpu, cuda, cuda:0)")
    parser.add_argument("--batch_size", type=int, default=256, help="임베딩 추출 시 배치 크기")
    parser.add_argument('--load_cache' , type=str , default = None ,help="임베딩 캐시경로")
    args = parser.parse_args()
    #args.data_path = '/home/ubuntu/arcface-pytorch/insight_face_package_model/split_pair/aligned'

    for key , values in args.__dict__.items():
        print(f"key {key}  :  {values}")

    main(args)