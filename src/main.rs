use anyhow::Result;

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

async fn initialize_qdrant(config: &Config) -> Option<Arc<QdrantStorage>> {
    if config.qdrant.url.is_empty() {
        return None;
    }

    // 检查 Qdrant 是否已经在运行
    let qdrant_running = tokio::net::TcpStream::connect("127.0.0.1:6334")
        .await
        .is_ok();
    
    if !qdrant_running {
        // 如果配置了 Qdrant 二进制路径，尝试启动 Qdrant
        if let Some(ref bin_path) = config.qdrant.bin_path {
            tracing::info!("Qdrant not running, starting from: {}", bin_path);
            
            let qdrant_dir = std::path::Path::new(bin_path)
                .parent()
                .unwrap_or(std::path::Path::new("."));
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
            
            let child = std::process::Command::new(bin_path)
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
