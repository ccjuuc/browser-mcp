use anyhow::{Context, Result};

mod codebase;
mod app_config;
mod embedding;
mod indexer_state;
mod mcp;
mod parser;
mod storage;
mod http_server;

use codebase::CodebaseIndexer;
use app_config::Config;
use mcp::MCPServer;
use http_server::HttpServer;
use std::sync::Arc;
use storage::qdrant::QdrantStorage;

#[tokio::main]
async fn main() -> Result<()> {
    // 检查命令行参数
    let args: Vec<String> = std::env::args().collect();
    let http_mode = args.iter().any(|arg| arg == "--http");
    let port: u16 = args
        .iter()
        .position(|arg| arg == "--port")
        .and_then(|i| args.get(i + 1))
        .and_then(|p| p.parse().ok())
        .unwrap_or(3000);
    
    // 加载配置
    let config = Config::load()?;
    
    // 初始化日志
    let log_level = std::env::var("RUST_LOG")
        .unwrap_or_else(|_| config.server.log_level.clone());
    
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_new(&log_level)
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::from_default_env())
        )
        .init();

    let codebase_path = config.codebase_path();
    
    if !codebase_path.exists() {
        tracing::warn!(
            "Codebase path does not exist: {:?}. Please check your configuration.",
            codebase_path
        );
    }
    
    if http_mode {
        tracing::info!("🌐 Initializing HTTP MCP server on port {}", port);
    } else {
        tracing::info!("📟 Initializing stdio MCP server");
    }
    tracing::info!("Codebase: {:?}", codebase_path);
    tracing::info!("Config: max_results={}, max_file_size={} bytes", 
                   config.server.max_results, 
                   config.codebase.max_file_size);
    
    // 初始化 Qdrant 存储
    let qdrant_storage = initialize_qdrant(&config).await;

    let indexer = Arc::new(CodebaseIndexer::with_embedding_config(
        codebase_path.clone(),
        config.codebase.clone(),
        config.embedding.clone(),
        qdrant_storage,
    ));
    
    // 检查索引状态
    let state_path = std::env::current_dir()
        .unwrap_or_else(|_| std::path::PathBuf::from("."))
        .join("browser-mcp-index.json");
    
    let has_index_state = state_path.exists();
    if has_index_state {
        // 检查已索引的文件数量
        if let Ok(content) = std::fs::read_to_string(&state_path) {
            if let Ok(state) = serde_json::from_str::<serde_json::Value>(&content) {
                if let Some(file_states) = state.get("file_states").and_then(|v| v.as_object()) {
                    let indexed_count = file_states.len();
                    tracing::info!("Index state found with {} files indexed. Continuing indexing for new/modified files...", indexed_count);
                } else {
                    tracing::info!("Index state found but empty. Starting fresh indexing...");
                }
            } else {
                tracing::info!("Index state file corrupted. Starting fresh indexing...");
            }
        } else {
            tracing::info!("Index state found but unreadable. Starting fresh indexing...");
        }
    } else {
        tracing::info!("No index state found, starting initial indexing...");
    }
    
    // 如果启用了 Qdrant，始终启动索引任务（会跳过已索引且未修改的文件）
    if indexer.is_qdrant_enabled() {
        let indexer_clone = indexer.clone();
        tokio::spawn(async move {
            if let Err(e) = indexer_clone.index_codebase().await {
                tracing::error!("Background indexing failed: {}", e);
            }
        });
    }

    // 根据模式启动对应的服务器
    if http_mode {
        tracing::info!("🚀 Starting HTTP MCP Server on http://127.0.0.1:{}", port);
        let http_server = HttpServer::new(indexer, port);
        http_server.run().await?;
    } else {
        tracing::info!("🚀 Starting stdio MCP Server");
        let server = MCPServer::new(indexer);
        server.run().await?;
    }
    
    Ok(())
}

/// 下载 Qdrant 可执行文件
async fn download_qdrant_binary(target_path: &std::path::Path) -> Result<()> {
    use std::io::Write;
    
    // 检测平台
    let (os, arch, ext) = if cfg!(target_os = "linux") {
        if cfg!(target_arch = "x86_64") {
            ("linux", "amd64", "tar.gz")
        } else if cfg!(target_arch = "aarch64") {
            ("linux", "arm64", "tar.gz")
        } else {
            return Err(anyhow::anyhow!("Unsupported Linux architecture"));
        }
    } else if cfg!(target_os = "macos") {
        if cfg!(target_arch = "x86_64") {
            ("macos", "amd64", "tar.gz")
        } else if cfg!(target_arch = "aarch64") {
            ("macos", "arm64", "tar.gz")
        } else {
            return Err(anyhow::anyhow!("Unsupported macOS architecture"));
        }
    } else if cfg!(target_os = "windows") {
        if cfg!(target_arch = "x86_64") {
            ("windows", "amd64", "zip")
        } else {
            return Err(anyhow::anyhow!("Unsupported Windows architecture"));
        }
    } else {
        return Err(anyhow::anyhow!("Unsupported operating system"));
    };

    // 获取最新版本（使用固定版本以确保稳定性）
    let version = "v1.11.2"; // 可以后续改为从 API 获取最新版本
    let binary_name = if cfg!(target_os = "windows") {
        "qdrant.exe"
    } else {
        "qdrant"
    };
    
    let archive_name = format!("qdrant-{}-{}-{}.{}", version, os, arch, ext);
    let download_url = format!(
        "https://github.com/qdrant/qdrant/releases/download/{}/{}",
        version, archive_name
    );

    tracing::info!("Downloading Qdrant from: {}", download_url);
    
    // 创建目标目录
    if let Some(parent) = target_path.parent() {
        std::fs::create_dir_all(parent)
            .context("Failed to create Qdrant directory")?;
    }

    // 下载文件
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .build()
        .context("Failed to create HTTP client")?;

    let response = client
        .get(&download_url)
        .send()
        .await
        .context("Failed to download Qdrant")?;

    if !response.status().is_success() {
        return Err(anyhow::anyhow!(
            "Failed to download Qdrant: HTTP {}",
            response.status()
        ));
    }

    let archive_data = response
        .bytes()
        .await
        .context("Failed to read download response")?;

    // 解压并提取可执行文件
    let temp_dir = std::env::temp_dir().join(format!("qdrant-download-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&temp_dir)
        .context("Failed to create temp directory")?;

    let archive_path = temp_dir.join(&archive_name);
    std::fs::write(&archive_path, &archive_data)
        .context("Failed to write archive to temp file")?;

    // 解压
    if ext == "tar.gz" {
        let tar_gz = std::fs::File::open(&archive_path)
            .context("Failed to open archive")?;
        let tar = flate2::read::GzDecoder::new(tar_gz);
        let mut archive = tar::Archive::new(tar);
        archive.unpack(&temp_dir)
            .context("Failed to extract archive")?;
    } else if ext == "zip" {
        let mut zip = zip::ZipArchive::new(std::fs::File::open(&archive_path)
            .context("Failed to open archive")?)
            .context("Failed to read zip archive")?;
        zip.extract(&temp_dir)
            .context("Failed to extract zip archive")?;
    }

    // 查找可执行文件
    let extracted_binary = temp_dir.join(&archive_name.replace(&format!(".{}", ext), ""))
        .join(binary_name);
    
    if !extracted_binary.exists() {
        // 尝试在解压目录中查找
        let mut found = false;
        for entry in walkdir::WalkDir::new(&temp_dir).into_iter() {
            if let Ok(entry) = entry {
                if entry.file_name() == binary_name {
                    std::fs::copy(entry.path(), target_path)
                        .context("Failed to copy binary")?;
                    found = true;
                    break;
                }
            }
        }
        if !found {
            return Err(anyhow::anyhow!("Binary not found in archive"));
        }
    } else {
        std::fs::copy(&extracted_binary, target_path)
            .context("Failed to copy binary")?;
    }

    // 设置可执行权限（Unix 系统）
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(target_path)
            .context("Failed to get file metadata")?
            .permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(target_path, perms)
            .context("Failed to set executable permissions")?;
    }

    // 清理临时文件
    let _ = std::fs::remove_dir_all(&temp_dir);

    tracing::info!("Successfully downloaded Qdrant to: {}", target_path.display());
    Ok(())
}

async fn initialize_qdrant(config: &Config) -> Option<Arc<QdrantStorage>> {
    if config.qdrant.url.is_empty() {
        return None;
    }

    // 检查 Qdrant 是否已经在运行
    let qdrant_running = tokio::net::TcpStream::connect("127.0.0.1:6334")
        .await
        .is_ok();
    
    if !qdrant_running {
        // 确定 Qdrant 二进制路径
        let bin_path = if let Some(ref configured_path) = config.qdrant.bin_path {
            std::path::PathBuf::from(configured_path)
        } else {
            // 如果没有配置，使用默认路径
            let default_dir = std::env::current_dir()
                .unwrap_or_else(|_| std::path::PathBuf::from("."))
                .join("qdrant");
            let binary_name = if cfg!(target_os = "windows") {
                "qdrant.exe"
            } else {
                "qdrant"
            };
            default_dir.join(binary_name)
        };

        // 如果二进制文件不存在，尝试下载
        if !bin_path.exists() {
            tracing::info!("Qdrant binary not found at: {}, attempting to download...", bin_path.display());
            if let Err(e) = download_qdrant_binary(&bin_path).await {
                tracing::error!("Failed to download Qdrant: {}", e);
                return None;
            }
        }

        // 启动 Qdrant
        let bin_path_str = bin_path.to_string_lossy().to_string();
        tracing::info!("Qdrant not running, starting from: {}", bin_path_str);
        
        let qdrant_dir = bin_path.parent()
            .unwrap_or_else(|| std::path::Path::new("."));
        let storage_path = qdrant_dir.join("storage");
        
        if let Err(e) = std::fs::create_dir_all(&storage_path) {
            tracing::warn!("Failed to create storage directory: {}", e);
        }
        
        let config_path = qdrant_dir.join("config.yaml");
        let config_content = format!(
            r#"storage:
  storage_path: {}
  optimizers:
    deleted_threshold: 0.9
    vacuum_min_vector_number: 100000
    default_segment_number: 2
    flush_interval_sec: 30
    max_optimization_threads: 1

service:
  host: 0.0.0.0
  http_port: 6333
  grpc_port: 6334
  max_request_size_mb: 64
"#,
            storage_path.to_string_lossy().replace('\\', "/")
        );
        
        if let Err(e) = std::fs::write(&config_path, config_content) {
            tracing::warn!("Failed to create Qdrant config: {}", e);
        }
        
        let child = std::process::Command::new(&bin_path_str)
            .arg("--config-path")
            .arg(&config_path)
            .current_dir(qdrant_dir)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn();
        
        match child {
            Ok(mut child_handle) => {
                tracing::info!("Qdrant process started, waiting for it to be ready...");
                
                // 等待 Qdrant 启动并检查是否成功
                let mut attempts = 0;
                let max_attempts = 30; // 最多等待 30 秒
                let mut qdrant_ready = false;
                
                while attempts < max_attempts {
                    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
                    
                    // 检查进程是否还在运行
                    if let Ok(Some(_)) = child_handle.try_wait() {
                        tracing::error!("Qdrant process exited unexpectedly");
                        break;
                    }
                    
                    // 检查端口是否可访问
                    if tokio::net::TcpStream::connect("127.0.0.1:6334").await.is_ok() {
                        qdrant_ready = true;
                        tracing::info!("Qdrant is ready on port 6334");
                        break;
                    }
                    
                    attempts += 1;
                }
                
                if !qdrant_ready {
                    tracing::warn!("Qdrant may not be ready yet, but continuing...");
                }
            }
            Err(e) => {
                tracing::error!("Failed to start Qdrant: {}", e);
            }
        }
    } else {
        tracing::info!("Qdrant already running on port 6334");
    }

    // 再次检查 Qdrant 是否可用（给启动一些额外时间）
    let mut retries = 0;
    let max_retries = 5;
    let mut qdrant_available = false;
    
    while retries < max_retries {
        if tokio::net::TcpStream::connect("127.0.0.1:6334").await.is_ok() {
            qdrant_available = true;
            break;
        }
        retries += 1;
        if retries < max_retries {
            tracing::debug!("Waiting for Qdrant to be ready (attempt {}/{})...", retries, max_retries);
            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        }
    }
    
    if !qdrant_available {
        tracing::warn!("Qdrant is not available on port 6334. Vector search will be disabled.");
        return None;
    }

    match QdrantStorage::new(&config.qdrant.url, config.qdrant.collection_name.clone(), config.embedding.dimension as u64) {
        Ok(storage) => {
            if let Err(e) = storage.init().await {
                tracing::error!("Failed to initialize Qdrant: {}", e);
                tracing::warn!("Vector search will be disabled. You can still use text search.");
                None
            } else {
                tracing::info!("Qdrant initialized successfully");
                Some(Arc::new(storage))
            }
        }
        Err(e) => {
            tracing::error!("Failed to create Qdrant client: {}", e);
            tracing::warn!("Vector search will be disabled. You can still use text search.");
            None
        }
    }
}
