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
        tracing::info!("🌐 HTTP MCP Server listening on http://0.0.0.0:{}", self.port);
        tracing::info!("📡 MCP endpoint: POST http://localhost:{}/", self.port);
        tracing::info!("❤️  Health check: GET http://localhost:{}/health", self.port);

        let listener = tokio::net::TcpListener::bind(addr).await?;
        axum::serve(listener, app).await?;

        Ok(())
    }
}

async fn health_check() -> impl IntoResponse {
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
    // 创建一个 stdio 风格的处理器
    let mcp_server = MCPServer::new(indexer);
    
    // 调用处理逻辑（我们需要导出一个公共方法）
    match mcp_server.handle_json_request(request). await {
        Ok(response) => (StatusCode::OK, Json(response)).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "jsonrpc": "2.0",
                "error": {
                    "code": -32603,
                    "message": e.to_string()
                }
            })),
        )
            .into_response(),
    }
}
