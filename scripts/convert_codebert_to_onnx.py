#!/usr/bin/env python3
"""
将 CodeBERT 模型从 PyTorch 转换为 ONNX 格式
用于在 Rust 项目中使用 ONNX Runtime 运行

使用方法:
    python convert_codebert_to_onnx.py --model microsoft/codebert-base --output ./model
"""

import argparse
import torch
from pathlib import Path
from transformers import AutoModel, AutoTokenizer

def convert_to_onnx(model_name: str, output_dir: str, max_length: int = 512):
    """将 CodeBERT 模型转换为 ONNX 格式"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    
    # 保存分词器（Rust tokenizers 库需要）
    print("Saving tokenizer...")
    tokenizer.save_pretrained(str(output_path))
    
    # 创建示例输入
    dummy_code = "def hello(): pass"
    dummy_input = tokenizer(
        dummy_code,
        return_tensors="pt",
        padding="max_length",
        max_length=max_length,
        truncation=True
    )
    
    # 导出为 ONNX
    onnx_path = output_path / "model.onnx"
    print(f"Exporting to ONNX: {onnx_path}")
    
    torch.onnx.export(
        model,
        (dummy_input['input_ids'], dummy_input['attention_mask']),
        str(onnx_path),
        input_names=['input_ids', 'attention_mask'],
        output_names=['last_hidden_state', 'pooler_output'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
            'last_hidden_state': {0: 'batch_size', 1: 'sequence_length'},
            'pooler_output': {0: 'batch_size'},
        },
        opset_version=14,
        do_constant_folding=True,
    )
    
    print(f"✅ Model exported successfully to {onnx_path}")
    print(f"✅ Tokenizer saved to {output_path}")
    print(f"\nNext steps:")
    print(f"1. Update browser-mcp.toml:")
    print(f"   [embedding]")
    print(f"   model_path = \"{onnx_path.absolute()}\"")
    print(f"   tokenizer_path = \"{output_path.absolute()}/tokenizer.json\"")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CodeBERT to ONNX")
    parser.add_argument(
        "--model",
        type=str,
        default="microsoft/codebert-base",
        help="Hugging Face model name or path"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./model",
        help="Output directory for ONNX model and tokenizer"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )
    
    args = parser.parse_args()
    convert_to_onnx(args.model, args.output, args.max_length)

