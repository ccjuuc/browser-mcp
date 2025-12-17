use anyhow::Result;
#[cfg(feature = "model-embedding")]
use anyhow::Context;
#[cfg(feature = "model-embedding")]
use candle_core::{Device, Tensor};
#[cfg(feature = "model-embedding")]
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use serde::{Deserialize, Serialize};
#[cfg(feature = "model-embedding")]
use std::path::Path;
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
        // 优先查找已存在的模型，如果找到则使用
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
            .map(|p| p.to_string())
            // 如果没有找到已存在的模型，默认使用 "./model" 路径尝试加载
            // 如果加载失败，会自动回退到 TF-IDF
            .or_else(|| Some("./model".to_string()));
        
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
        let (model, tokenizer, device) = if let Some(ref model_path) = config.model_path {
            // 创建设备：
            // - 默认优先尝试使用 CUDA GPU（device 0）
            // - 如果 CUDA 不可用或失败，则自动回退到 CPU
            let device = match Device::cuda_if_available(0) {
                Ok(dev) => {
                    tracing::info!("Using CUDA device 0 for model embeddings");
                    dev
                }
                Err(e) => {
                    tracing::warn!(
                        "CUDA device not available for model embeddings, falling back to CPU: {}",
                        e
                    );
                    Device::Cpu
                }
            };
            // 尝试加载模型
            match Self::load_model(model_path, &config.tokenizer_path, &device) {
                Ok((m, t)) => {
                    tracing::info!("Successfully loaded CodeBERT model from: {}", model_path);
                    (Some(Arc::new(m)), Some(Arc::new(t)), device)
                }
                Err(e) => {
                    tracing::warn!(
                        "Failed to load model from {}: {}. Falling back to TF-IDF.",
                        model_path,
                        e
                    );
                    (None, None, device) // device 仍然保留，但不会被使用
                }
            }
        } else {
            // 如果没有配置模型路径，不需要创建 device
            (None, None, Device::Cpu) // 使用默认值，但不会被使用
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

    /// 将旧格式 tokenizer (vocab.json + merges.txt) 转换为 tokenizer.json
    #[cfg(feature = "model-embedding")]
    fn convert_legacy_tokenizer(model_dir: &Path) -> Result<()> {
        // 使用 Python 的 transformers 库来转换
        // CodeBERT 使用 RoBERTa tokenizer，需要从 Hugging Face 下载 fast tokenizer
        // 或者使用 AutoTokenizer 自动处理
        let script_content = r#"
import sys
from pathlib import Path
from transformers import AutoTokenizer

model_dir = sys.argv[1]
model_dir_path = Path(model_dir)

try:
    # 方法1: 如果有 tokenizer_config.json，尝试从本地加载
    if (model_dir_path / "tokenizer_config.json").exists():
        try:
            # 尝试从本地文件加载（使用 fast tokenizer）
            tokenizer = AutoTokenizer.from_pretrained(
                str(model_dir_path),
                local_files_only=True,
                use_fast=True
            )
            # 如果成功，保存为 tokenizer.json
            tokenizer.save_pretrained(str(model_dir_path))
            print("Successfully converted tokenizer to tokenizer.json (from local files)")
            sys.exit(0)
        except Exception as e:
            print(f"Local load failed: {e}, trying from Hugging Face...", file=sys.stderr)
    
    # 方法2: 从 Hugging Face 下载完整的 fast tokenizer
    # 对于 CodeBERT，模型 ID 是 microsoft/codebert-base
    model_id = "microsoft/codebert-base"
    print(f"Downloading fast tokenizer from {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    
    # 保存到本地目录
    tokenizer.save_pretrained(str(model_dir_path))
    
    # 验证文件是否创建
    tokenizer_json = model_dir_path / "tokenizer.json"
    if not tokenizer_json.exists():
        print(f"Error: tokenizer.json was not created after save_pretrained", file=sys.stderr)
        print(f"Files in directory: {list(model_dir_path.glob('*'))}", file=sys.stderr)
        sys.exit(1)
    
    print("Successfully converted tokenizer to tokenizer.json")
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
"#;
        
        // 创建临时 Python 脚本
        let temp_script = std::env::temp_dir().join("convert_tokenizer.py");
        std::fs::write(&temp_script, script_content)
            .context("Failed to create temporary conversion script")?;
        
        let output = std::process::Command::new("python3")
            .arg(&temp_script)
            .arg(model_dir)
            .output()
            .context("Failed to execute tokenizer conversion script")?;
        
        // 清理临时脚本
        let _ = std::fs::remove_file(&temp_script);
        
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            return Err(anyhow::anyhow!(
                "Tokenizer conversion failed. stderr: {}, stdout: {}",
                stderr,
                stdout
            ));
        }
        
        // 验证转换是否成功
        let tokenizer_json_path = model_dir.join("tokenizer.json");
        if !tokenizer_json_path.exists() {
            return Err(anyhow::anyhow!(
                "Tokenizer conversion completed but tokenizer.json not found in {}",
                model_dir.display()
            ));
        }
        
        Ok(())
    }

    /// 从旧格式（vocab.json + merges.txt）构建 tokenizer
    #[cfg(feature = "model-embedding")]
    fn build_tokenizer_from_legacy(vocab_path: &Path, merges_path: &Path) -> Result<Tokenizer> {
        use tokenizers::models::bpe::BPE;
        use tokenizers::pre_tokenizers::byte_level::ByteLevel;
        use tokenizers::normalizers::NFC;
        use tokenizers::processors::bert::BertProcessing;
        use std::collections::HashMap;

        tracing::info!("Building BPE tokenizer from vocab.json and merges.txt...");

        // 使用 BPE::from_file 直接从文件构建
        let vocab_str = vocab_path.to_string_lossy().to_string();
        let merges_str = merges_path.to_string_lossy().to_string();
        
        let bpe = BPE::from_file(&vocab_str, &merges_str)
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to build BPE model from vocab.json and merges.txt: {}", e))?;

        // 读取 vocab.json 以查找特殊 token
        let vocab_content = std::fs::read_to_string(vocab_path)
            .with_context(|| format!("Failed to read vocab.json from {}", vocab_path.display()))?;
        let vocab: HashMap<String, u32> = serde_json::from_str(&vocab_content)
            .context("Failed to parse vocab.json")?;

        // 查找特殊 token（CodeBERT 使用 RoBERTa 格式，通常是 <s> 和 </s>）
        let cls_token = vocab.iter()
            .find(|(k, _)| k.as_str() == "<s>" || k.as_str() == "[CLS]")
            .map(|(k, _)| k.clone())
            .unwrap_or_else(|| "<s>".to_string());
        
        let sep_token = vocab.iter()
            .find(|(k, _)| k.as_str() == "</s>" || k.as_str() == "[SEP]")
            .map(|(k, _)| k.clone())
            .unwrap_or_else(|| "</s>".to_string());

        let cls_id = *vocab.get(&cls_token).unwrap_or(&0);
        let sep_id = *vocab.get(&sep_token).unwrap_or(&1);

        // 构建 tokenizer
        let mut tokenizer = Tokenizer::new(bpe);
        
        // 设置 pre-tokenizer（ByteLevel 用于 RoBERTa/CodeBERT）
        tokenizer.with_pre_tokenizer(ByteLevel::default());
        
        // 设置 normalizer（NFC 用于 Unicode 规范化）
        tokenizer.with_normalizer(NFC);
        
        // 设置 post-processor（BERT 处理器，但使用 RoBERTa 的特殊 token）
        tokenizer.with_post_processor(
            BertProcessing::new(
                (sep_token.clone(), sep_id),
                (cls_token.clone(), cls_id),
            )
        );

        tracing::info!("Successfully built tokenizer from legacy format (vocab.json + merges.txt)");
        Ok(tokenizer)
    }

    /// 将 PyTorch 模型转换为 safetensors 格式
    #[cfg(feature = "model-embedding")]
    fn convert_pytorch_to_safetensors(model_dir: &Path) -> Result<()> {
        // 查找转换脚本，按优先级：
        // 1. 当前工作目录的 scripts/
        // 2. 可执行文件所在目录的 scripts/
        // 3. 项目根目录（通过 CARGO_MANIFEST_DIR 环境变量）
        let script_path = std::env::current_dir()
            .ok()
            .and_then(|d| {
                let path = d.join("scripts/convert_pytorch_to_safetensors.py");
                if path.exists() { Some(path) } else { None }
            })
            .or_else(|| {
                std::env::current_exe()
                    .ok()
                    .and_then(|exe| {
                        let path = exe.parent().map(|p| p.join("scripts/convert_pytorch_to_safetensors.py"))?;
                        if path.exists() { Some(path) } else { None }
                    })
            })
            .or_else(|| {
                // 尝试从 CARGO_MANIFEST_DIR 环境变量（编译时设置）
                std::env::var("CARGO_MANIFEST_DIR")
                    .ok()
                    .map(|d| PathBuf::from(d).join("scripts/convert_pytorch_to_safetensors.py"))
                    .filter(|p| p.exists())
            })
            .ok_or_else(|| anyhow::anyhow!(
                "Conversion script not found. Please ensure scripts/convert_pytorch_to_safetensors.py exists in the project directory."
            ))?;
        
        let output = std::process::Command::new("python3")
            .arg(&script_path)
            .arg(model_dir)
            .output()
            .context("Failed to execute conversion script")?;
        
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow::anyhow!(
                "Conversion script failed: {}",
                stderr
            ));
        }
        
        Ok(())
    }

    /// 从 Hugging Face 下载模型
    #[cfg(feature = "model-embedding")]
    fn download_huggingface_model(model_id: &str) -> Result<PathBuf> {
        // 确定缓存目录
        let cache_dir = if let Some(cache) = dirs::cache_dir() {
            cache.join("browser-mcp").join("models").join(model_id.replace('/', "_"))
        } else {
            PathBuf::from("~/.cache/browser-mcp/models").join(model_id.replace('/', "_"))
        };
        
        let cache_dir = shellexpand::full(cache_dir.to_string_lossy().as_ref())
            .map(|s| PathBuf::from(s.as_ref()))
            .unwrap_or_else(|_| PathBuf::from("./models").join(model_id.replace('/', "_")));
        
        // 创建缓存目录
        std::fs::create_dir_all(&cache_dir)
            .context("Failed to create model cache directory")?;
        
        // 检查模型是否已下载
        let config_path = cache_dir.join("config.json");
        let safetensors_path = cache_dir.join("model.safetensors");
        let pytorch_path = cache_dir.join("pytorch_model.bin");
        let tokenizer_path = cache_dir.join("tokenizer.json");
        let vocab_path = cache_dir.join("vocab.json");
        let merges_path = cache_dir.join("merges.txt");
        
        // 检查是否已有模型权重（safetensors 或 pytorch）
        let has_weights = safetensors_path.exists() || pytorch_path.exists();
        // 检查是否有 tokenizer（新格式或旧格式）
        let has_tokenizer = tokenizer_path.exists() || (vocab_path.exists() && merges_path.exists());
        if config_path.exists() && has_weights && has_tokenizer {
            // 如果有旧格式但没有新格式，尝试转换
            if !tokenizer_path.exists() && vocab_path.exists() && merges_path.exists() {
                tracing::info!("Found legacy tokenizer format, converting to tokenizer.json...");
                if let Err(e) = Self::convert_legacy_tokenizer(&cache_dir) {
                    tracing::warn!("Failed to convert legacy tokenizer: {}", e);
                }
            }
            tracing::info!("Model already cached at: {}", cache_dir.display());
            return Ok(cache_dir);
        }
        
        tracing::info!("Downloading model {} from Hugging Face...", model_id);
        
        // Hugging Face API 基础 URL
        let base_url = format!("https://huggingface.co/{}/resolve/main", model_id);
        
        // 使用 reqwest 下载文件
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(300))
            .build()
            .context("Failed to create HTTP client")?;
        
        // 下载 config.json
        if !config_path.exists() {
            let url = format!("{}/config.json", base_url);
            tracing::info!("Downloading config.json from {}...", url);
            let response = client.get(&url).send()?;
            if !response.status().is_success() {
                return Err(anyhow::anyhow!(
                    "Failed to download config.json: HTTP {}",
                    response.status()
                ));
            }
            let content = response.bytes()?;
            std::fs::write(&config_path, &content)?;
            tracing::info!("Downloaded config.json ({} bytes)", content.len());
        }
        
        // 尝试下载 tokenizer 文件
        // 优先尝试 tokenizer.json（新格式）
        if !tokenizer_path.exists() {
            let url = format!("{}/tokenizer.json", base_url);
            tracing::info!("Downloading tokenizer.json from {}...", url);
            let response = client.get(&url).send()?;
            if response.status().is_success() {
                let content = response.bytes()?;
                std::fs::write(&tokenizer_path, &content)?;
                tracing::info!("Downloaded tokenizer.json ({} bytes)", content.len());
            } else {
                // 如果 tokenizer.json 不存在，尝试下载旧格式（vocab.json + merges.txt）
                tracing::info!("tokenizer.json not found, trying legacy format (vocab.json + merges.txt)...");
                
                // 下载 vocab.json
                if !vocab_path.exists() {
                    let vocab_url = format!("{}/vocab.json", base_url);
                    let vocab_response = client.get(&vocab_url).send()?;
                    if vocab_response.status().is_success() {
                        let content = vocab_response.bytes()?;
                        std::fs::write(&vocab_path, &content)?;
                        tracing::info!("Downloaded vocab.json ({} bytes)", content.len());
                    } else {
                        return Err(anyhow::anyhow!(
                            "Failed to download vocab.json: HTTP {}",
                            vocab_response.status()
                        ));
                    }
                }
                
                // 下载 merges.txt
                if !merges_path.exists() {
                    let merges_url = format!("{}/merges.txt", base_url);
                    let merges_response = client.get(&merges_url).send()?;
                    if merges_response.status().is_success() {
                        let content = merges_response.bytes()?;
                        std::fs::write(&merges_path, &content)?;
                        tracing::info!("Downloaded merges.txt ({} bytes)", content.len());
                    } else {
                        return Err(anyhow::anyhow!(
                            "Failed to download merges.txt: HTTP {}",
                            merges_response.status()
                        ));
                    }
                }
                
                // 下载 tokenizer_config.json（可选）
                let tokenizer_config_path = cache_dir.join("tokenizer_config.json");
                if !tokenizer_config_path.exists() {
                    let config_url = format!("{}/tokenizer_config.json", base_url);
                    let config_response = client.get(&config_url).send()?;
                    if config_response.status().is_success() {
                        let content = config_response.bytes()?;
                        std::fs::write(&tokenizer_config_path, &content)?;
                        tracing::info!("Downloaded tokenizer_config.json ({} bytes)", content.len());
                    }
                }
                
                // 如果有旧格式文件，尝试转换为 tokenizer.json
                if vocab_path.exists() && merges_path.exists() {
                    tracing::info!("Converting legacy tokenizer format to tokenizer.json...");
                    if let Err(e) = Self::convert_legacy_tokenizer(&cache_dir) {
                        tracing::warn!("Failed to convert legacy tokenizer: {}. Will try to load directly.", e);
                    } else if tokenizer_path.exists() {
                        tracing::info!("Successfully converted tokenizer to tokenizer.json");
                    }
                }
            }
        }
        
        // 尝试下载模型权重：优先 safetensors，如果不存在则使用 pytorch_model.bin
        let weights_downloaded = if !safetensors_path.exists() {
            let url = format!("{}/model.safetensors", base_url);
            tracing::info!("Downloading model.safetensors from {}...", url);
            
            let response = client.get(&url).send()?;
            if response.status().is_success() {
                let content = response.bytes()
                    .context("Failed to read model.safetensors response")?;
                std::fs::write(&safetensors_path, &content)
                    .context("Failed to save model.safetensors")?;
                tracing::info!("Downloaded model.safetensors ({} bytes)", content.len());
                true
            } else {
                // 尝试下载 pytorch_model.bin
                let pytorch_url = format!("{}/pytorch_model.bin", base_url);
                tracing::warn!("model.safetensors not available, trying pytorch_model.bin...");
                let pytorch_response = client.get(&pytorch_url).send()?;
                if pytorch_response.status().is_success() {
                    let content = pytorch_response.bytes()
                        .context("Failed to read pytorch_model.bin response")?;
                    std::fs::write(&pytorch_path, &content)
                        .context("Failed to save pytorch_model.bin")?;
                    tracing::info!("Downloaded pytorch_model.bin ({} bytes)", content.len());
                    
                    // 尝试自动转换为 safetensors
                    tracing::info!("Attempting to convert pytorch_model.bin to safetensors format...");
                    if let Err(e) = Self::convert_pytorch_to_safetensors(&cache_dir) {
                        tracing::warn!("Failed to auto-convert model: {}. You may need to manually convert it using the provided script.", e);
                    } else {
                        tracing::info!("Successfully converted model to safetensors format");
                    }
                    true
                } else {
                    false
                }
            }
        } else {
            true
        };
        
        if !weights_downloaded {
            return Err(anyhow::anyhow!(
                "Failed to download model weights. Neither model.safetensors nor pytorch_model.bin is available for model {}.",
                model_id
            ));
        }
        
        tracing::info!("Model downloaded successfully to: {}", cache_dir.display());
        Ok(cache_dir)
    }

    /// 加载 BERT 模型和分词器
    #[cfg(feature = "model-embedding")]
    fn load_model(
        model_path: &str,
        tokenizer_path: &Option<String>,
        device: &Device,
    ) -> Result<(BertModel, Tokenizer)> {
        // 检查是否是 Hugging Face 模型 ID（格式：username/model-name）
        let actual_model_path = if model_path.contains('/') && !Path::new(model_path).exists() && !model_path.starts_with('.') && !model_path.starts_with('~') && !model_path.starts_with('/') {
            // 可能是 Hugging Face 模型 ID，尝试下载
            tracing::info!("Detected Hugging Face model ID: {}, attempting to download...", model_path);
            Self::download_huggingface_model(model_path)?
        } else {
            // 本地路径
            let expanded = shellexpand::full(model_path)
                .map(|s| s.to_string())
                .unwrap_or_else(|_| model_path.to_string());
            PathBuf::from(&expanded)
        };
        
        let model_path_buf = actual_model_path;
        
        // 加载分词器
        // 检查是否有旧格式的 tokenizer（vocab.json + merges.txt）
        let vocab_path = model_path_buf.join("vocab.json");
        let merges_path = model_path_buf.join("merges.txt");
        let tokenizer_json_path = model_path_buf.join("tokenizer.json");
        
        // 优先使用 vocab.json + merges.txt（更可靠，兼容性更好）
        // 如果两者都存在，优先使用旧格式，避免 tokenizer.json 的兼容性问题
        let tokenizer = if vocab_path.exists() && merges_path.exists() {
            // 直接使用旧格式构建，更可靠
            tracing::info!("Building tokenizer from vocab.json + merges.txt...");
            Self::build_tokenizer_from_legacy(&vocab_path, &merges_path)?
        } else if tokenizer_json_path.exists() {
            // 如果没有旧格式，尝试从 tokenizer.json 加载
            match Tokenizer::from_file(&tokenizer_json_path) {
                Ok(t) => {
                    tracing::info!("Successfully loaded tokenizer from {}", tokenizer_json_path.display());
                    t
                }
                Err(e) => {
                    tracing::warn!(
                        "Failed to load tokenizer from {}: {}. Please ensure vocab.json and merges.txt are available.",
                        tokenizer_json_path.display(),
                        e
                    );
                    return Err(anyhow::anyhow!(
                        "Failed to load tokenizer from {}: {}. Please ensure vocab.json and merges.txt are available.",
                        tokenizer_json_path.display(),
                        e
                    ));
                }
            }
        } else if let Some(ref tp) = tokenizer_path {
            // 如果指定了自定义 tokenizer 路径
            let custom_path = PathBuf::from(shellexpand::full(tp)?.as_ref());
            if custom_path.exists() {
                Tokenizer::from_file(&custom_path)
                    .map_err(|e| anyhow::anyhow!("Failed to load tokenizer from {}: {}", custom_path.display(), e))?
            } else {
                return Err(anyhow::anyhow!(
                    "Tokenizer file not found: {}. Please ensure the tokenizer file exists.",
                    custom_path.display()
                ));
            }
        } else {
            // 如果本地没有，尝试从 Hugging Face 加载
            // 对于 CodeBERT，使用 microsoft/codebert-base
            tracing::info!("Tokenizer not found locally, attempting to load from Hugging Face");
            return Err(anyhow::anyhow!(
                "Tokenizer files not found. Please ensure vocab.json and merges.txt (or tokenizer.json) exist in the model directory."
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
        
        // 解析 config.json
        // 某些模型（如 Jina）可能有额外字段，serde 默认会忽略未知字段
        // 但如果解析失败，可能是字段类型不匹配或必需字段缺失
        let bert_config: BertConfig = serde_json::from_str(&config_str)
            .with_context(|| {
                // 提供更详细的错误信息
                let preview = config_str.lines().take(10).collect::<Vec<_>>().join("\n");
                format!(
                    "Failed to parse config.json as BertConfig. \
                    The model may not be fully compatible with candle's BertConfig structure. \
                    Config preview:\n{}",
                    preview
                )
            })?;

        // 加载模型权重
        // 优先尝试 safetensors
        let weights_path = if model_path_buf.join("model.safetensors").exists() {
            model_path_buf.join("model.safetensors")
        } else if model_path_buf.join("pytorch_model.bin").exists() {
            // 如果只有 pytorch_model.bin，尝试自动转换
            tracing::info!("Only pytorch_model.bin found, attempting to convert to safetensors...");
            if let Err(e) = Self::convert_pytorch_to_safetensors(&model_path_buf) {
                return Err(anyhow::anyhow!(
                    "Only pytorch_model.bin found, but automatic conversion failed: {}. \
                    Please manually convert using: python3 scripts/convert_pytorch_to_safetensors.py {} \
                    Or install safetensors: pip install safetensors",
                    e,
                    model_path_buf.display()
                ));
            }
            // 转换后应该存在 safetensors 文件
            if model_path_buf.join("model.safetensors").exists() {
                model_path_buf.join("model.safetensors")
            } else {
                return Err(anyhow::anyhow!(
                    "Conversion completed but model.safetensors not found in: {}",
                    model_path_buf.display()
                ));
            }
        } else {
            return Err(anyhow::anyhow!(
                "Model weights not found. Please ensure model.safetensors or pytorch_model.bin exists in: {}",
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

    /// 检查是否使用模型进行向量化（而不是 TF-IDF）
    #[allow(dead_code)] // 在索引统计中使用
    pub fn is_using_model(&self) -> bool {
        #[cfg(feature = "model-embedding")]
        {
            self.model.is_some() && self.tokenizer.is_some()
        }
        #[cfg(not(feature = "model-embedding"))]
        {
            false
        }
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
