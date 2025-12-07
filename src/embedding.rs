use anyhow::Result;
#[cfg(feature = "model-embedding")]
use candle_core::{Device, Tensor};
#[cfg(feature = "model-embedding")]
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use tokenizers::Tokenizer;

/// 向量化配置
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingConfig {
    /// 向量维度
    pub dimension: usize,
    /// 代码向量化模型路径（可选，如果提供则使用模型，否则使用 TF-IDF）
    /// 可以是本地目录路径（包含 model.safetensors 和 config.json）或 Hugging Face 模型 ID
    #[serde(default)]
    pub model_path: Option<String>,
    /// 分词器路径（可选，如果提供模型路径则需要）
    #[serde(default)]
    pub tokenizer_path: Option<String>,
    /// 最大序列长度
    #[serde(default = "default_max_length")]
    pub max_length: usize,
}

fn default_max_length() -> usize {
    512
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        // 默认尝试从常见路径加载 CodeBERT 模型
        let default_model_paths = vec![
            "./model",
            "./codebert",
            "~/.browser-mcp/model",
        ];
        
        let model_path = default_model_paths
            .iter()
            .find(|path| {
                let expanded = shellexpand::full(path).unwrap_or_else(|_| path.to_string().into());
                let path_buf = PathBuf::from(expanded.as_ref());
                // 检查是否存在模型文件（safetensors 或 bin）
                path_buf.join("model.safetensors").exists() 
                    || path_buf.join("pytorch_model.bin").exists()
                    || path_buf.join("model.bin").exists()
            })
            .map(|p| p.to_string());
        
        Self {
            dimension: 768, // CodeBERT 输出维度
            model_path,
            tokenizer_path: None, // 将从 model_path 推断
            max_length: default_max_length(),
        }
    }
}

/// 向量化器
pub struct Embedder {
    config: EmbeddingConfig,
    #[cfg(feature = "model-embedding")]
    // Candle BERT 模型（如果使用模型）
    model: Option<Arc<BertModel>>,
    // 分词器（如果使用模型）
    tokenizer: Option<Arc<Tokenizer>>,
    #[cfg(feature = "model-embedding")]
    // 设备（CPU 或 GPU）
    device: Device,
}

impl Embedder {
    pub fn new(config: EmbeddingConfig) -> Result<Self> {
        #[cfg(feature = "model-embedding")]
        let device = Device::Cpu; // 可以后续支持 GPU
        
        #[cfg(feature = "model-embedding")]
        let (model, tokenizer) = if let Some(ref model_path) = config.model_path {
            // 尝试加载模型
            match Self::load_model(model_path, &config.tokenizer_path, &device) {
                Ok((m, t)) => {
                    tracing::info!("Successfully loaded CodeBERT model from: {}", model_path);
                    (Some(Arc::new(m)), Some(Arc::new(t)))
                }
                Err(e) => {
                    tracing::warn!(
                        "Failed to load model from {}: {}. Falling back to TF-IDF.",
                        model_path,
                        e
                    );
                    (None, None)
                }
            }
        } else {
            (None, None)
        };
        
        #[cfg(not(feature = "model-embedding"))]
        {
            if config.model_path.is_some() {
                tracing::warn!(
                    "Model embedding is disabled: compile with --features model-embedding to enable. Using TF-IDF fallback."
                );
            }
            return Ok(Self {
                config,
                tokenizer: None,
            });
        }
        
        #[cfg(feature = "model-embedding")]
        Ok(Self {
            config,
            model,
            tokenizer,
            device,
        })
    }

    /// 加载 BERT 模型和分词器
    #[cfg(feature = "model-embedding")]
    fn load_model(
        model_path: &str,
        tokenizer_path: &Option<String>,
        device: &Device,
    ) -> Result<(BertModel, Tokenizer)> {
        // 展开路径（处理 ~ 等）
        let expanded_model_path = shellexpand::full(model_path)
            .map(|s| s.to_string())
            .unwrap_or_else(|_| model_path.to_string());
        
        let model_path_buf = PathBuf::from(&expanded_model_path);
        
        // 加载分词器
        let tokenizer_path_buf = if let Some(ref tp) = tokenizer_path {
            PathBuf::from(shellexpand::full(tp)?.as_ref())
        } else {
            // 尝试从模型路径推断分词器路径
            let tokenizer_file = model_path_buf.join("tokenizer.json");
            if tokenizer_file.exists() {
                tokenizer_file
            } else {
                // 尝试其他常见位置
                model_path_buf.join("vocab.json")
                    .parent()
                    .map(|p| p.join("tokenizer.json"))
                    .unwrap_or_else(|| model_path_buf.join("tokenizer.json"))
            }
        };

        let tokenizer = if tokenizer_path_buf.exists() {
            Tokenizer::from_file(&tokenizer_path_buf)
                .map_err(|e| anyhow::anyhow!("Failed to load tokenizer from {}: {}", tokenizer_path_buf.display(), e))?
        } else {
            // 如果本地没有，尝试从 Hugging Face 加载
            // 对于 CodeBERT，使用 microsoft/codebert-base
            tracing::info!("Tokenizer not found locally, attempting to load from Hugging Face");
            return Err(anyhow::anyhow!(
                "Tokenizer file not found: {}. Please download the model and tokenizer first.",
                tokenizer_path_buf.display()
            ));
        };

        // 加载模型配置
        let config_path = model_path_buf.join("config.json");
        if !config_path.exists() {
            return Err(anyhow::anyhow!(
                "Model config not found: {}. Please ensure config.json exists in the model directory.",
                config_path.display()
            ));
        }

        let config_str = std::fs::read_to_string(&config_path)
            .context("Failed to read config.json")?;
        let bert_config: BertConfig = serde_json::from_str(&config_str)
            .context("Failed to parse config.json")?;

        // 加载模型权重
        // 优先尝试 safetensors
        let weights_path = if model_path_buf.join("model.safetensors").exists() {
            model_path_buf.join("model.safetensors")
        } else {
            return Err(anyhow::anyhow!(
                "Model weights not found. Please ensure model.safetensors exists in: {} (pytorch_model.bin support temporarily removed)",
                model_path_buf.display()
            ));
        };

        // 使用 candle 加载模型
        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(
                &[weights_path],
                candle_core::DType::F32,
                device,
            )?
        };

        let model = BertModel::load(vb, &bert_config)
            .context("Failed to load BERT model weights")?;

        Ok((model, tokenizer))
    }

    /// 生成文本的向量嵌入
    pub async fn embed(&self, text: &str) -> Result<Vec<f32>> {
        #[cfg(feature = "model-embedding")]
        if let (Some(ref model), Some(ref tokenizer)) = (&self.model, &self.tokenizer) {
            return self.embed_with_model(text, Arc::clone(model), tokenizer).await;
        }
        
        // 降级到 TF-IDF
        self.embed_tfidf(text).await
    }

    /// 批量生成向量嵌入
    pub async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        #[cfg(feature = "model-embedding")]
        if let (Some(ref model), Some(ref tokenizer)) = (&self.model, &self.tokenizer) {
            // 使用模型批量处理
            let mut embeddings = Vec::new();
            for text in texts {
                embeddings.push(self.embed_with_model(text, Arc::clone(model), tokenizer).await?);
            }
            return Ok(embeddings);
        }
        
        // 使用 TF-IDF
        let mut embeddings = Vec::new();
        for text in texts {
            embeddings.push(self.embed_tfidf(text).await?);
        }
        Ok(embeddings)
    }

    /// 使用模型生成向量嵌入
    #[cfg(feature = "model-embedding")]
    async fn embed_with_model(
        &self,
        text: &str,
        model: Arc<BertModel>,
        tokenizer: &Arc<Tokenizer>,
    ) -> Result<Vec<f32>> {
        // candle 操作是同步的，但可以在 async 上下文中使用
        // 由于模型只需要 &self，我们可以直接调用，无需 spawn_blocking
        let text = text.to_string();
        let config = self.config.clone();
        let device = self.device.clone();
        let tokenizer = Arc::clone(tokenizer);
        
        // 在 blocking 线程中运行，避免阻塞 async runtime
        tokio::task::spawn_blocking(move || {
            // 分词
            let encoding = tokenizer
                .encode(text, true)
                .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;
            
            let input_ids: Vec<u32> = encoding
                .get_ids()
                .iter()
                .map(|&id| id as u32)
                .collect();
            
            // 填充或截断到最大长度
            let mut padded_ids = vec![0u32; config.max_length];
            let len = input_ids.len().min(config.max_length);
            padded_ids[..len].copy_from_slice(&input_ids[..len]);
            
            // 创建 attention mask
            let attention_mask: Vec<u32> = (0..len)
                .map(|_| 1u32)
                .chain((len..config.max_length).map(|_| 0u32))
                .collect();
            
            // 创建输入张量
            // shape: [batch_size, seq_len] = [1, max_length]
            let input_ids_tensor = Tensor::new(
                padded_ids.as_slice(),
                &device
            )
            .map_err(|e| anyhow::anyhow!("Failed to create input_ids tensor: {}", e))?
            .unsqueeze(0)?; // 添加 batch 维度
            
            let attention_mask_tensor = Tensor::new(
                attention_mask.as_slice(),
                &device
            )
            .map_err(|e| anyhow::anyhow!("Failed to create attention_mask tensor: {}", e))?
            .unsqueeze(0)?; // 添加 batch 维度
            
            // 创建 token_type_ids (全 0)
            let token_type_ids_tensor = Tensor::zeros(
                (1, config.max_length),
                candle_core::DType::U32,
                &device
            ).map_err(|e| anyhow::anyhow!("Failed to create token_type_ids tensor: {}", e))?;

            // 运行推理
            // BertModel::forward 返回 (hidden_states, optional_pooler_output) or just hidden_states depending on version
            // In 0.8+ it is forward(&self, input_ids: &Tensor, token_type_ids: &Tensor, attention_mask: Option<&Tensor>)
            let hidden_states = model
                .forward(&input_ids_tensor, &token_type_ids_tensor, Some(&attention_mask_tensor))
                .map_err(|e| anyhow::anyhow!("Model inference failed: {}", e))?;
            
            // 使用 [CLS] token 的嵌入（第一个 token）作为句子嵌入
            // hidden_states shape: [batch_size, seq_len, hidden_size]
            let cls_embedding = hidden_states
                .get(0)
                .and_then(|batch| batch.get(0)) // 获取第一个 token ([CLS])
                .context("Failed to extract [CLS] token embedding")?;
            
            // 转换为 Vec<f32>
            let embedding_vec: Vec<f32> = cls_embedding
                .to_vec1::<f32>()
                .map_err(|e| anyhow::anyhow!("Failed to convert tensor to vec: {}", e))?;
            
            // 归一化
            let norm: f32 = embedding_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            let normalized: Vec<f32> = if norm > 0.0 {
                embedding_vec.iter().map(|x| x / norm).collect()
            } else {
                embedding_vec
            };
            
            Ok(normalized)
        })
        .await
        .context("Failed to join model inference task")?
    }

    /// 使用 TF-IDF 生成向量
    async fn embed_tfidf(&self, text: &str) -> Result<Vec<f32>> {
        // 简单的 TF-IDF 实现
        let tokens = Self::tokenize(text);
        let mut vector = vec![0.0; self.config.dimension];
        
        // 使用哈希函数将 token 映射到向量维度
        for token in tokens {
            let hash = Self::hash_token(&token);
            let index = hash % self.config.dimension;
            vector[index] += 1.0;
        }
        
        // 归一化
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut vector {
                *v /= norm;
            }
        }
        
        Ok(vector)
    }

    /// 分词（简单的实现）
    fn tokenize(text: &str) -> Vec<String> {
        // 移除标点符号，转换为小写，按空白分割
        text.to_lowercase()
            .chars()
            .map(|c| if c.is_alphanumeric() || c.is_whitespace() {
                c
            } else {
                ' '
            })
            .collect::<String>()
            .split_whitespace()
            .filter(|s| s.len() > 2) // 过滤太短的词
            .map(|s| s.to_string())
            .collect()
    }

    /// 哈希函数，将 token 映射到向量索引
    fn hash_token(token: &str) -> usize {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        token.hash(&mut hasher);
        hasher.finish() as usize
    }

    /// 计算余弦相似度
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }
        
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        
        dot_product / (norm_a * norm_b)
    }
}

impl Default for Embedder {
    fn default() -> Self {
        Self::new(EmbeddingConfig::default()).expect("Failed to create default embedder")
    }
}
