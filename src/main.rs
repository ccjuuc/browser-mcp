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
    
    // 检查是否需要索引
    let state_path = std::env::current_dir()
        .unwrap_or_else(|_| std::path::PathBuf::from("."))
        .join("browser-mcp-index.json");
    
    let needs_indexing = !state_path.exists();
    if needs_indexing {
        tracing::info!("No index state found, initial indexing required");
    } else {
        tracing::info!("Index state found, skipping initial indexing");
    }
    
    // 如果启用了 Qdrant 且需要索引，启动后台索引任务
    if indexer.is_qdrant_enabled() && needs_indexing {
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
                .spawn();
            
            match child {
                Ok(_) => {
                    tracing::info!("Qdrant started at {:?}", storage_path);
                    tokio::time::sleep(tokio::time::Duration::from_secs(3)).await;
                }
                Err(e) => {
                    tracing::error!("Failed to start Qdrant: {}", e);
                }
            }
        }
    } else {
        tracing::info!("Qdrant already running on port 6334");
    }

    match QdrantStorage::new(&config.qdrant.url, config.qdrant.collection_name.clone(), config.embedding.dimension as u64) {
        Ok(storage) => {
            if let Err(e) = storage.init().await {
                tracing::error!("Failed to initialize Qdrant: {}", e);
                None
            } else {
                Some(Arc::new(storage))
            }
        }
        Err(e) => {
            tracing::error!("Failed to create Qdrant client: {}", e);
            None
        }
    }
}
