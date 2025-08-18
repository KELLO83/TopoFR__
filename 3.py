import torch
import lightning as L
from torch.utils.data import DataLoader, TensorDataset

# backbones.iresnet 모듈과 클래스 정의가 있다고 가정
from backbones.iresnet import IResNet, IBasicBlock

# 1. LightningModule Wrapper 클래스 (이전과 동일)
class ModelWrapper(L.LightningModule):
    def __init__(self, model, dataset, batch_size=1):
        super().__init__()
        # self.save_hyperparameters() # Lightning 2.0 이상에서는 이 줄을 추가하는 것이 좋습니다.
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size 

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        inputs, labels = batch
        if inputs.shape[0] < 2:
            return None
        # IResNet의 forward는 (logits, embedding) 튜플을 반환
        logits, _ = self(inputs)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def train_dataloader(self):
        # self.batch_size는 Tuner에 의해 최적값으로 업데이트됩니다.
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=4)
        
    def val_dataloader(self):
        # .fit()은 검증 데이터로더도 필요로 하므로 추가합니다.
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers=4)

# ===================================================================
# --- 1단계: 최대 배치 사이즈 탐색 ---
# ===================================================================

# 1-1. 모델 및 더미 데이터 준비
iresnet_model = IResNet(IBasicBlock, [2, 2, 2, 2]).to('cuda') # 간단한 iresnet18로 변경
dummy_dataset = TensorDataset(
    torch.randn(1024, 3, 112, 112), 
    torch.randint(0, 512, (1024,))
)

# 1-2. 탐색을 위한 Trainer 및 Tuner 준비 (단일 GPU 기준)
lightning_model = ModelWrapper(model=iresnet_model, dataset=dummy_dataset)
trainer_tune = L.Trainer(accelerator="gpu", devices=1)
tuner = L.pytorch.tuner.Tuner(trainer_tune)

print("1단계: 최대 배치 사이즈 탐색을 시작합니다...")
tuner.scale_batch_size(lightning_model, mode="power", init_val=2)

found_batch_size = lightning_model.batch_size
print("-" * 20)
print(f"탐색 완료! GPU당 최대 배치 사이즈: [ {found_batch_size} ]")
print("-" * 20)


# ===================================================================
# --- 2단계: 찾은 배치 사이즈로 분산 훈련 (동작 확인) ---
# ===================================================================

# 2-1. 훈련을 위한 새로운 Trainer 생성 (분산 설정 적용)
# fast_dev_run=True: 1개의 배치만 학습/검증하고 종료하여 동작만 빠르게 확인
trainer_fit = L.Trainer(
    accelerator="gpu", 
    devices=2,  # 사용할 GPU 개수 지정
    strategy="ddp", # 분산 훈련 전략
    max_steps=100
)

# 2-2. .fit() 호출
# lightning_model은 이미 최적의 batch_size 값을 알고 있습니다.
print("2단계: 찾은 배치 사이즈로 분산 훈련 동작을 확인합니다...")
trainer_fit.fit(model=lightning_model)

print("\n분산 훈련 테스트가 성공적으로 완료되었습니다.")