use anyhow::{Context, Result};
use serde::{Deserialize, Deserializer, Serialize};
use std::path::{Path, PathBuf};

/// 自定义反序列化函数，支持字符串和数组两种格式
fn deserialize_path_array<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
where
    D: Deserializer<'de>,
{
    use serde::de::Visitor;
    
    struct PathArrayVisitor;
    
    impl<'de> Visitor<'de> for PathArrayVisitor {
        type Value = Vec<String>;
        
        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("a string or an array of strings")
        }
        
        fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            Ok(vec![value.to_string()])
        }
        
        fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            Ok(vec![value])
        }
        
        fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
        where
            A: serde::de::SeqAccess<'de>,
        {
            let mut vec = Vec::new();
            while let Some(elem) = seq.next_element::<String>()? {
                vec.push(elem);
            }
            Ok(vec)
        }
    }
    
    deserializer.deserialize_any(PathArrayVisitor)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub codebase: CodebaseConfig,
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub embedding: crate::embedding::EmbeddingConfig,
    #[serde(default)]
    pub qdrant: QdrantSettings,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QdrantSettings {
    /// Qdrant 服务器 URL
    #[serde(default = "default_qdrant_url")]
    pub url: String,
    /// 集合名称
    #[serde(default = "default_collection_name")]
    pub collection_name: String,
    /// Qdrant二进制路径(可选)
    pub bin_path: Option<String>,
}

fn default_qdrant_url() -> String {
    "http://localhost:6334".to_string()
}

fn default_collection_name() -> String {
    "browser-mcp".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodebaseConfig {
    /// 代码库路径（支持数组格式，代码会自动选择第一个存在的路径）
    /// 支持两种格式：
    /// 1. 数组格式（推荐）：path = ['/path1', '/path2', '/path3']
    /// 2. 字符串格式（向后兼容）：path = '/path/to/codebase'
    #[serde(default, deserialize_with = "deserialize_path_array")]
    pub path: Vec<String>,
    /// 搜索时忽略的文件扩展名
    #[serde(default = "default_ignored_extensions")]
    pub ignored_extensions: Vec<String>,
    /// 搜索时忽略大于此大小的文件（字节）
    #[serde(default = "default_max_file_size")]
    pub max_file_size: u64,
    /// 代码切片最大大小（字节），默认 2000
    #[serde(default = "default_chunk_size")]
    pub chunk_size: usize,
    /// 是否启用代码切片（使用 tree-sitter）
    #[serde(default = "default_enable_chunking")]
    pub enable_chunking: bool,
    /// 是否启用向量化
    #[serde(default = "default_enable_embedding")]
    pub enable_embedding: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// 日志级别
    #[serde(default = "default_log_level")]
    pub log_level: String,
    /// 搜索默认最大结果数
    #[serde(default = "default_max_results")]
    pub max_results: usize,
    /// HTTPS 证书文件路径（可选，如果提供则启用 HTTPS）
    pub tls_cert: Option<String>,
    /// HTTPS 私钥文件路径（可选，如果提供则启用 HTTPS）
    pub tls_key: Option<String>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            log_level: default_log_level(),
            max_results: default_max_results(),
            tls_cert: None,
            tls_key: None,
        }
    }
}

fn default_ignored_extensions() -> Vec<String> {
    vec![
        "png".to_string(),
        "jpg".to_string(),
        "jpeg".to_string(),
        "gif".to_string(),
        "ico".to_string(),
        "svg".to_string(),
        "woff".to_string(),
        "woff2".to_string(),
        "ttf".to_string(),
        "otf".to_string(),
        "eot".to_string(),
        "pdf".to_string(),
        "zip".to_string(),
        "tar".to_string(),
        "gz".to_string(),
        "jar".to_string(),
        "class".to_string(),
        "so".to_string(),
        "dylib".to_string(),
        "dll".to_string(),
        "exe".to_string(),
    ]
}

fn default_max_file_size() -> u64 {
    1_000_000 // 1MB
}

impl Default for CodebaseConfig {
    fn default() -> Self {
        Self {
            path: vec!["./brave-browser".to_string()],
            ignored_extensions: default_ignored_extensions(),
            max_file_size: default_max_file_size(),
            chunk_size: default_chunk_size(),
            enable_chunking: default_enable_chunking(),
            enable_embedding: default_enable_embedding(),
        }
    }
}


fn default_log_level() -> String {
    "info".to_string()
}

fn default_max_results() -> usize {
    50
}

fn default_chunk_size() -> usize {
    2000 // 2KB per chunk
}

fn default_enable_chunking() -> bool {
    true
}

fn default_enable_embedding() -> bool {
    true
}

impl Config {
    /// 从文件加载配置
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path.as_ref())
            .with_context(|| format!("Failed to read config file: {:?}", path.as_ref()))?;
        
        let config: Config = toml::from_str(&content)
            .with_context(|| format!("Failed to parse config file: {:?}", path.as_ref()))?;
        
        Ok(config)
    }

    /// 从默认路径加载配置
    /// 按以下顺序查找配置文件：
    /// 1. 环境变量 `BROWSER_MCP_CONFIG` 指定的路径（最高优先级）
    /// 2. 当前目录的 `browser-mcp.toml`
    /// 3. 默认配置（如果都没找到）
    pub fn load() -> Result<Self> {
        // 首先检查环境变量
        if let Ok(config_path) = std::env::var("BROWSER_MCP_CONFIG") {
            return Self::from_file(config_path);
        }

        // 检查当前目录的 browser-mcp.toml
        let current_dir_config = PathBuf::from("browser-mcp.toml");
        if current_dir_config.exists() {
            return Self::from_file(current_dir_config);
        }

        // 如果都没找到，使用默认配置
        Ok(Self::default())
    }

    /// 获取所有代码库路径
    #[allow(dead_code)]
    pub fn codebase_paths(&self) -> Vec<PathBuf> {
        self.codebase.path.iter()
            .map(|p| PathBuf::from(p))
            .collect()
    }
}

impl Default for QdrantSettings {
    fn default() -> Self {
        Self {
            url: default_qdrant_url(),
            collection_name: default_collection_name(),
            bin_path: None,
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            codebase: CodebaseConfig::default(),
            server: ServerConfig::default(),
            embedding: crate::embedding::EmbeddingConfig::default(),
            qdrant: QdrantSettings::default(),
        }
    }
}

