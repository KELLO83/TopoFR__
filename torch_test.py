
import torch



batch_images = []


for i in range(10):
    sample = torch.randn(size=(3,112,112))
    batch_images.append(sample)


batch_tensor = torch.stack(batch_images)


print(batch_tensor.shape)
