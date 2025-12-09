#!/usr/bin/env python3
"""
将 PyTorch 模型 (pytorch_model.bin) 转换为 safetensors 格式
用于 candle 库加载

使用方法:
    python convert_pytorch_to_safetensors.py <model_directory>
"""
import sys
import json
from pathlib import Path
from safetensors.torch import save_file
import torch

def convert_pytorch_to_safetensors(model_dir: str):
    """将 pytorch_model.bin 转换为 model.safetensors"""
    model_path = Path(model_dir)
    pytorch_file = model_path / "pytorch_model.bin"
    safetensors_file = model_path / "model.safetensors"
    config_file = model_path / "config.json"
    
    if not pytorch_file.exists():
        print(f"Error: {pytorch_file} not found")
        return False
    
    if safetensors_file.exists():
        print(f"Info: {safetensors_file} already exists, skipping conversion")
        return True
    
    print(f"Loading PyTorch model from {pytorch_file}...")
    try:
        # 加载 PyTorch 模型权重
        state_dict = torch.load(pytorch_file, map_location="cpu")
        
        # 转换为 safetensors 格式
        print(f"Converting to safetensors format...")
        save_file(state_dict, safetensors_file)
        
        print(f"✅ Successfully converted to {safetensors_file}")
        return True
    except Exception as e:
        print(f"❌ Error converting model: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_pytorch_to_safetensors.py <model_directory>")
        sys.exit(1)
    
    model_dir = sys.argv[1]
    success = convert_pytorch_to_safetensors(model_dir)
    sys.exit(0 if success else 1)

