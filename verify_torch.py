import torch

print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"系统CUDA版本: {torch.cuda.get_device_properties(0).cuda_version_major}.{torch.cuda.get_device_properties(0).cuda_version_minor}")
    print(f"当前CUDA设备: {torch.cuda.get_device_name(0)}") 