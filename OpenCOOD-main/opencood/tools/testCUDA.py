import torch

print(torch.__version__)              # PyTorch 版本
print(torch.cuda.is_available())      # 是否检测到 CUDA
print(torch.cuda.get_device_name(0))  # GPU 型号（如 NVIDIA RTX 3060）