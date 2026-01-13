use crate::codebase::CodebaseIndexer;
use crate::mcp::MCPServer;
use anyhow::Result;
use axum::{
    extract::{Json, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Router,
};
use serde_json::Value;
use std::sync::Arc;
use tower_http::cors::{Any, CorsLayer};

pub struct HttpServer {
    indexer: Arc<CodebaseIndexer>,
    port: u16,
}

impl HttpServer {
    pub fn new(indexer: Arc<CodebaseIndexer>, port: u16) -> Self {
        Self { indexer, port }
    }

    pub async fn run(self) -> Result<()> {
        let app = Router::new()
            .route("/", post(handle_mcp_request))
            .route("/health", get(health_check))
            .layer(
                CorsLayer::new()
                    .allow_origin(Any)
                    .allow_methods(Any)
                    .allow_headers(Any),
            )
            .with_state(self.indexer);

        let addr = std::net::SocketAddr::from(([0, 0, 0, 0], self.port));
        
        tracing::info!("🚀 Starting HTTP MCP Server...");
        tracing::info!("🌐 HTTP MCP Server listening on http://0.0.0.0:{}", self.port);
        tracing::info!("📡 MCP endpoint: POST http://localhost:{}/", self.port);
        tracing::info!("❤️  Health check: GET http://localhost:{}/health", self.port);
        tracing::info!("✅ Server ready to accept connections");

        let listener = tokio::net::TcpListener::bind(addr).await?;
        
        tracing::info!("📡 Server started successfully, waiting for requests...");
        axum::serve(listener, app).await?;

        Ok(())
    }
}

async fn health_check() -> impl IntoResponse {
    tracing::debug!("🏥 Health check requested");
    Json(serde_json::json!({
        "status": "ok",
        "service": "browser-mcp",
        "version": env!("CARGO_PKG_VERSION"),
        "mode": "http"
    }))
}

async fn handle_mcp_request(
    State(indexer): State<Arc<CodebaseIndexer>>,
    Json(request): Json<Value>,
) -> impl IntoResponse {
    // 提取请求信息用于日志
    let method = request
        .get("method")
        .and_then(|m| m.as_str())
        .unwrap_or("unknown")
        .to_string();
    let request_id = request.get("id").cloned();
    
    tracing::info!(
        "📥 Received MCP request: method={}, id={:?}",
        method,
        request_id
    );
    
    // 创建一个 stdio 风格的处理器
    let mcp_server = MCPServer::new(indexer);
    
    // 调用处理逻辑
    let start_time = std::time::Instant::now();
    match mcp_server.handle_json_request(request).await {
        Ok(response) => {
            let duration = start_time.elapsed();
            let response_size = serde_json::to_string(&response)
                .map(|s| s.len())
                .unwrap_or(0);
            
            // 检查响应中是否有错误
            let has_error = response
                .get("error")
                .is_some();
            
            if has_error {
                let error_code = response
                    .get("error")
                    .and_then(|e| e.get("code"))
                    .and_then(|c| c.as_i64())
                    .unwrap_or(-1);
                let error_msg = response
                    .get("error")
                    .and_then(|e| e.get("message"))
                    .and_then(|m| m.as_str())
                    .unwrap_or("unknown error");
                
                tracing::warn!(
                    "⚠️  MCP request completed with error: method={}, id={:?}, code={}, message={}, duration={:?}ms",
                    method,
                    request_id,
                    error_code,
                    error_msg,
                    duration.as_millis()
                );
            } else {
                tracing::info!(
                    "✅ MCP request completed successfully: method={}, id={:?}, duration={:?}ms, response_size={} bytes",
                    method,
                    request_id,
                    duration.as_millis(),
                    response_size
                );
            }
            
            (StatusCode::OK, Json(response)).into_response()
        }
        Err(e) => {
            let duration = start_time.elapsed();
            let error_msg = e.to_string();
            
            tracing::error!(
                "❌ MCP request failed: method={}, id={:?}, error={}, duration={:?}ms",
                method,
                request_id,
                error_msg,
                duration.as_millis()
            );
            
            let error_response = serde_json::json!({
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32603,
                    "message": error_msg
                }
            });
            
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(error_response),
            )
                .into_response()
        }
    }
}
