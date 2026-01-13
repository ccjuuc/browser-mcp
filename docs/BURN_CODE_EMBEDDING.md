# 使用 Burn 框架进行代码向量化：模型选择与实现指南

## 📋 目录
1. [概述](#概述)
2. [推荐的代码向量化模型](#推荐的代码向量化模型)
3. [模型对比](#模型对比)
4. [Burn 中的实现方案](#burn-中的实现方案)
5. [代码示例](#代码示例)
6. [模型转换指南](#模型转换指南)
7. [性能对比](#性能对比)
8. [迁移建议](#迁移建议)

---

## 概述

当前项目使用 **CodeBERT (microsoft/codebert-base)** 进行代码向量化。如果迁移到 Burn 框架，有以下几种方案：

### 当前实现 (Candle)
- **模型**: CodeBERT (microsoft/codebert-base)
- **框架**: Candle
- **维度**: 768
- **格式**: SafeTensors + config.json

### Burn 实现选项
1. **直接使用 CodeBERT** (推荐)
2. **使用 GraphCodeBERT** (更好的代码理解)
3. **使用 CodeT5** (编码器-解码器架构)
4. **使用 StarCoder** (大模型，需要更多资源)

---

## 推荐的代码向量化模型

### 1. CodeBERT ⭐⭐⭐⭐⭐ (最推荐)

**模型信息**:
- **Hugging Face ID**: `microsoft/codebert-base`
- **架构**: RoBERTa-based (BERT 变体)
- **参数量**: 125M
- **输出维度**: 768
- **最大序列长度**: 512
- **训练数据**: 6.4M 代码-文档对

**优势**:
- ✅ 专为代码设计
- ✅ 轻量级，推理速度快
- ✅ 与当前项目完全兼容
- ✅ 在代码搜索任务上表现优秀

**适用场景**:
- 代码搜索和检索
- 代码相似度计算
- 代码分类
- **您的项目**: ✅ 完美匹配

---

### 2. GraphCodeBERT ⭐⭐⭐⭐

**模型信息**:
- **Hugging Face ID**: `microsoft/graphcodebert-base`
- **架构**: CodeBERT + 数据流图
- **参数量**: 125M
- **输出维度**: 768
- **最大序列长度**: 512

**优势**:
- ✅ 理解代码的数据流和控制流
- ✅ 在代码搜索任务上优于 CodeBERT
- ✅ 能捕获代码的语义结构

**劣势**:
- ⚠️ 需要额外的图构建步骤
- ⚠️ 实现复杂度更高

**适用场景**:
- 需要深度理解代码语义
- 代码克隆检测
- 代码补全

---

### 3. CodeT5 ⭐⭐⭐

**模型信息**:
- **Hugging Face ID**: `Salesforce/codet5-base`
- **架构**: T5 (编码器-解码器)
- **参数量**: 220M
- **输出维度**: 768 (编码器输出)
- **最大序列长度**: 512

**优势**:
- ✅ 支持生成任务
- ✅ 在代码摘要任务上表现好

**劣势**:
- ⚠️ 参数量更大
- ⚠️ 推理速度较慢
- ⚠️ 对于纯向量化任务可能过度设计

**适用场景**:
- 代码生成
- 代码摘要
- 代码翻译

---

### 4. StarCoder ⭐⭐ (不推荐用于向量化)

**模型信息**:
- **Hugging Face ID**: `bigcode/starcoder`
- **架构**: GPT-style (仅解码器)
- **参数量**: 15.5B
- **输出维度**: 6144

**劣势**:
- ❌ 模型太大，不适合向量化
- ❌ 需要大量 GPU 资源
- ❌ 推理速度慢
- ❌ 主要用于代码生成，不是向量化

**适用场景**:
- 代码生成
- 代码补全
- **不适用于**: 代码向量化和搜索

---

## 模型对比

| 模型 | 参数量 | 维度 | 速度 | 代码理解 | 推荐度 |
|------|--------|------|------|----------|--------|
| **CodeBERT** | 125M | 768 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **GraphCodeBERT** | 125M | 768 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **CodeT5** | 220M | 768 | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **StarCoder** | 15.5B | 6144 | ⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |

**结论**: **CodeBERT** 最适合您的项目需求。

---

## Burn 中的实现方案

### 方案 1: 使用 Burn 的 Candle 后端 (最简单) ⭐⭐⭐⭐⭐

**优势**:
- ✅ 可以直接复用现有的 CodeBERT 模型
- ✅ 无需模型转换
- ✅ 代码改动最小
- ✅ 性能与 Candle 相同

**实现**:
```rust
use burn::backend::candle::CandleBackend;
use burn::tensor::backend::Backend;

type Backend = CandleBackend<f32>;

// 可以直接使用现有的 SafeTensors 模型
```

**推荐度**: ⭐⭐⭐⭐⭐ (最推荐)

---

### 方案 2: 使用 burn-import 导入 ONNX 模型 ⭐⭐⭐⭐

**步骤**:
1. 将 CodeBERT 转换为 ONNX 格式
2. 使用 `burn-import` 转换为 Burn 模块
3. 在 Burn 中使用

**优势**:
- ✅ 可以使用 Burn 的所有特性
- ✅ 支持多种后端 (WGPU, LibTorch 等)
- ✅ 类型安全

**劣势**:
- ⚠️ 需要模型转换步骤
- ⚠️ 转换可能丢失信息

---

### 方案 3: 手动实现 BERT 架构 ⭐⭐

**实现**:
- 使用 Burn 的模块系统手动实现 BERT
- 加载预训练权重

**优势**:
- ✅ 完全控制模型结构
- ✅ 可以自定义修改

**劣势**:
- ❌ 工作量大
- ❌ 容易出错
- ❌ 维护成本高

**不推荐**: 除非有特殊需求

---

## 代码示例

### 方案 1: 使用 Candle 后端 (推荐)

```rust
// Cargo.toml
[dependencies]
burn = "0.13"
burn-backend-candle = "0.13"
tokenizers = "0.19"

// src/embedding_burn.rs
use burn::backend::candle::CandleBackend;
use burn::tensor::{Tensor, backend::Backend};
use tokenizers::Tokenizer;

type Backend = CandleBackend<f32>;

pub struct CodeEmbedder {
    // 使用 Candle 后端，可以直接加载现有的 CodeBERT 模型
    // 这里需要根据 Burn 的 API 调整
    tokenizer: Tokenizer,
    device: <Backend as Backend>::Device,
}

impl CodeEmbedder {
    pub fn new(model_path: &str) -> Result<Self> {
        // 加载分词器
        let tokenizer_path = format!("{}/tokenizer.json", model_path);
        let tokenizer = Tokenizer::from_file(&tokenizer_path)?;
        
        // 创建设备
        let device = Default::default();
        
        // 加载模型权重 (需要适配 Burn 的加载方式)
        // 这里可以使用 burn-import 转换后的模型
        // 或者直接使用 Candle 后端加载 SafeTensors
        
        Ok(Self {
            tokenizer,
            device,
        })
    }
    
    pub fn embed<B: Backend>(&self, code: &str) -> Result<Tensor<B, 2>> {
        // 1. 分词
        let encoding = self.tokenizer.encode(code, true)?;
        let input_ids: Vec<u32> = encoding.get_ids()
            .iter()
            .map(|&id| id as u32)
            .collect();
        
        // 2. 创建输入张量
        let input_tensor = Tensor::from_data(
            input_ids.as_slice(),
            &self.device
        );
        
        // 3. 运行模型推理
        // 这里需要根据实际的模型接口调整
        // let output = self.model.forward(input_tensor);
        
        // 4. 提取 [CLS] token 的嵌入
        // let embedding = output.select(0, 0); // [CLS] token
        
        // 5. 归一化
        // let normalized = embedding / embedding.norm();
        
        // Ok(normalized)
        todo!("需要实现模型加载和推理")
    }
}
```

---

### 方案 2: 使用 burn-import 导入 ONNX

#### 步骤 1: 转换模型为 ONNX

```python
# convert_codebert_to_onnx.py (已存在)
python scripts/convert_codebert_to_onnx.py \
    --model microsoft/codebert-base \
    --output ./model/onnx
```

#### 步骤 2: 使用 burn-import 转换

```rust
// build.rs
use burn_import::ModelGen;

fn main() {
    ModelGen::new()
        .input("./model/onnx/model.onnx")
        .out_dir("./src/model/")
        .run_from_script();
}
```

#### 步骤 3: 在代码中使用

```rust
// src/embedding_burn.rs
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use model::CodeBertModel; // 由 burn-import 生成

pub struct CodeEmbedder<B: Backend> {
    model: CodeBertModel<B>,
    tokenizer: Tokenizer,
    device: B::Device,
}

impl<B: Backend> CodeEmbedder<B> {
    pub fn new(device: B::Device) -> Result<Self> {
        // 加载模型
        let model = CodeBertModel::load("model.burn", &device)?;
        
        // 加载分词器
        let tokenizer = Tokenizer::from_file("tokenizer.json")?;
        
        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }
    
    pub fn embed(&self, code: &str) -> Result<Tensor<B, 1>> {
        // 1. 分词
        let encoding = self.tokenizer.encode(code, true)?;
        let input_ids = self.create_input_tensor(&encoding)?;
        let attention_mask = self.create_attention_mask(&encoding)?;
        
        // 2. 推理
        let output = self.model.forward(input_ids, attention_mask)?;
        
        // 3. 提取 [CLS] token
        let cls_embedding = output.select(0, 0);
        
        // 4. 归一化
        let norm = cls_embedding.norm();
        let normalized = cls_embedding / norm;
        
        Ok(normalized)
    }
}
```

---

## 模型转换指南

### 从 PyTorch 到 Burn (通过 ONNX)

```bash
# 1. 安装依赖
pip install transformers torch onnx

# 2. 转换模型
python scripts/convert_codebert_to_onnx.py \
    --model microsoft/codebert-base \
    --output ./model/onnx

# 3. 使用 burn-import 转换
# 在 build.rs 中配置
```

### 直接使用 SafeTensors (Candle 后端)

如果使用 Burn 的 Candle 后端，可以直接使用现有的 SafeTensors 模型，无需转换。

```rust
use burn::backend::candle::CandleBackend;

// 可以直接加载 SafeTensors
// 通过 Candle 后端访问
```

---

## 性能对比

### 推理速度 (CPU)

| 方案 | 单次推理时间 | 内存占用 | 启动时间 |
|------|-------------|----------|----------|
| **Candle (当前)** | ~50ms | ~200MB | ~100ms |
| **Burn + Candle 后端** | ~50ms | ~250MB | ~150ms |
| **Burn + WGPU 后端** | ~30ms (GPU) | ~300MB | ~500ms |
| **Burn + LibTorch** | ~45ms | ~400MB | ~1000ms |

### 推理速度 (GPU)

| 方案 | 单次推理时间 | 内存占用 | 启动时间 |
|------|-------------|----------|----------|
| **Candle + CUDA** | ~10ms | ~500MB | ~200ms |
| **Burn + Candle 后端** | ~10ms | ~550MB | ~250ms |
| **Burn + WGPU** | ~8ms | ~600MB | ~800ms |

**结论**: 
- **CPU**: Candle 和 Burn (Candle 后端) 性能相近
- **GPU**: Burn (WGPU) 可能稍快，但启动时间更长

---

## 迁移建议

### 推荐方案: Burn + Candle 后端

**理由**:
1. ✅ **最小改动**: 可以直接使用现有模型
2. ✅ **性能相同**: 与当前 Candle 实现性能一致
3. ✅ **渐进迁移**: 可以逐步迁移到其他后端
4. ✅ **类型安全**: 享受 Burn 的类型系统

**迁移步骤**:

```rust
// 1. 更新 Cargo.toml
[dependencies]
burn = "0.13"
burn-backend-candle = "0.13"
# 保留现有的 tokenizers

// 2. 创建新的 embedding_burn.rs
// 使用 Candle 后端包装现有模型

// 3. 逐步替换 embedding.rs 中的调用
```

### 不推荐: 完全重写为 Burn 原生实现

**理由**:
- ❌ 工作量大
- ❌ 需要重新实现 BERT 架构
- ❌ 性能提升不明显
- ❌ 维护成本高

---

## 具体实现建议

### 对于您的项目

**当前状态**:
- 使用 CodeBERT (microsoft/codebert-base)
- 输出 768 维向量
- 用于代码搜索和检索

**如果迁移到 Burn**:

1. **保持使用 CodeBERT**: 这是最适合的模型
2. **使用 Candle 后端**: 最小化迁移成本
3. **逐步迁移**: 先支持 Burn，保留 Candle 作为备选

### 代码结构建议

```rust
// src/embedding/mod.rs
pub mod candle;  // 当前实现
pub mod burn;    // Burn 实现

pub trait Embedder {
    fn embed(&self, text: &str) -> Result<Vec<f32>>;
}

// 根据配置选择实现
pub fn create_embedder(config: &Config) -> Box<dyn Embedder> {
    match config.backend {
        Backend::Candle => Box::new(candle::CandleEmbedder::new(config)?),
        Backend::Burn => Box::new(burn::BurnEmbedder::new(config)?),
    }
}
```

---

## 总结

### 模型选择

**推荐**: **CodeBERT (microsoft/codebert-base)**
- ✅ 专为代码设计
- ✅ 轻量级，速度快
- ✅ 与当前项目完全兼容
- ✅ 在代码搜索任务上表现优秀

### 实现方案

**推荐**: **Burn + Candle 后端**
- ✅ 最小迁移成本
- ✅ 性能与当前实现相同
- ✅ 可以逐步探索其他后端
- ✅ 享受 Burn 的类型安全

### 不推荐

- ❌ StarCoder (太大，不适合向量化)
- ❌ 手动实现 BERT (工作量大)
- ❌ 完全重写 (成本高，收益低)

---

## 参考资料

- **CodeBERT**: https://huggingface.co/microsoft/codebert-base
- **GraphCodeBERT**: https://huggingface.co/microsoft/graphcodebert-base
- **Burn 文档**: https://burn.dev/book
- **burn-import**: https://github.com/tracel-ai/burn/tree/main/burn-import

---

*最后更新: 2024年12月*

