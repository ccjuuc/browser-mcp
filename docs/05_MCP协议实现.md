# MCP 协议实现

本文档详细介绍 Model Context Protocol (MCP) 的实现原理和在项目中的应用。

## 1. MCP 协议概述

### 1.1 什么是 MCP？

Model Context Protocol (MCP) 是 Anthropic 开发的一个开放协议，用于让 AI 助手（如 Claude）与外部工具和数据源交互。

### 1.2 核心概念

- **服务器（Server）**：提供工具和资源的服务（如本项目）
- **客户端（Client）**：使用服务器的 AI 助手（如 Claude Desktop、Cursor）
- **工具（Tools）**：可执行的操作（如搜索代码、读取文件）
- **资源（Resources）**：可访问的数据（如代码搜索、文件列表）

### 1.3 通信方式

MCP 使用 JSON-RPC 2.0 协议进行通信：

- **stdio 模式**：通过标准输入输出（stdin/stdout）通信
- **HTTP 模式**：通过 HTTP POST 请求通信

## 2. JSON-RPC 2.0 基础

### 2.1 请求格式

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "search_code",
    "arguments": {
      "query": "function add"
    }
  }
}
```

**字段说明**：
- `jsonrpc`：协议版本，固定为 "2.0"
- `id`：请求 ID，用于匹配请求和响应
- `method`：要调用的方法名
- `params`：方法参数

### 2.2 响应格式

**成功响应**：
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "content": [{
      "type": "text",
      "text": "[{\"file_path\": \"src/main.rs\", ...}]"
    }]
  }
}
```

**错误响应**：
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "error": {
    "code": -32603,
    "message": "Internal error"
  }
}
```

### 2.3 错误码

- `-32700`：Parse error（解析错误）
- `-32600`：Invalid Request（无效请求）
- `-32601`：Method not found（方法不存在）
- `-32602`：Invalid params（参数无效）
- `-32603`：Internal error（内部错误）
- `-32000`：Server error（服务器错误，自定义）

## 3. MCP 核心方法

### 3.1 initialize（初始化）

客户端连接后首先调用，进行握手和协商能力。

**请求**：
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "initialize",
  "params": {
    "protocolVersion": "2024-11-05",
    "capabilities": {},
    "clientInfo": {
      "name": "claude-desktop",
      "version": "1.0.0"
    }
  }
}
```

**响应**：
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "protocolVersion": "2024-11-05",
    "capabilities": {
      "resources": {},
      "tools": {}
    },
    "serverInfo": {
      "name": "browser-mcp",
      "version": "0.1.0"
    }
  }
}
```

**实现代码**：
```rust
fn initialize_response(&self, id: Option<Value>) -> MCPResponse {
    MCPResponse {
        jsonrpc: "2.0".to_string(),
        id,
        result: Some(serde_json::json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "resources": {},
                "tools": {}
            },
            "serverInfo": {
                "name": "browser-mcp",
                "version": "0.1.0"
            }
        })),
        error: None,
    }
}
```

### 3.2 tools/list（列出工具）

返回服务器提供的所有工具。

**请求**：
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/list"
}
```

**响应**：
```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "result": {
    "tools": [
      {
        "name": "search_code",
        "description": "Search for code using text matching",
        "inputSchema": {
          "type": "object",
          "properties": {
            "query": {
              "type": "string",
              "description": "The text or pattern to search for"
            },
            "max_results": {
              "type": "number",
              "description": "Maximum number of results"
            }
          },
          "required": ["query"]
        }
      },
      // ... 更多工具
    ]
  }
}
```

**实现代码**：
```rust
async fn list_tools(&self, id: Option<Value>) -> Result<MCPResponse> {
    let tools = vec![
        serde_json::json!({
            "name": "search_code",
            "description": "Search for code using text matching",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The text or pattern to search for"
                    },
                    "max_results": {
                        "type": "number",
                        "description": "Maximum number of results (default: 50)"
                    }
                },
                "required": ["query"]
            }
        }),
        // ... 其他工具定义
    ];

    Ok(MCPResponse {
        jsonrpc: "2.0".to_string(),
        id,
        result: Some(serde_json::json!({ "tools": tools })),
        error: None,
    })
}
```

### 3.3 tools/call（调用工具）

执行工具并返回结果。

**请求**：
```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "method": "tools/call",
  "params": {
    "name": "search_code",
    "arguments": {
      "query": "function add",
      "max_results": 10
    }
  }
}
```

**响应**：
```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "result": {
    "content": [{
      "type": "text",
      "text": "[{\"file_path\": \"src/main.rs\", \"line_number\": 10, ...}]"
    }]
  }
}
```

**实现代码**：
```rust
async fn handle_tool_call(
    &self,
    id: Option<Value>,
    params: Option<Value>,
) -> Result<MCPResponse> {
    let params_value = params.context("Missing params")?;
    let tool_name = params_value
        .get("name")
        .and_then(|v| v.as_str())
        .context("Missing tool name")?;
    
    let arguments = params_value.get("arguments");

    match tool_name {
        "search_code" => {
            let query = arguments
                .and_then(|a| a.get("query"))
                .and_then(|q| q.as_str())
                .unwrap_or("")
                .to_string();
            
            let max_results = arguments
                .and_then(|a| a.get("max_results"))
                .and_then(|m| m.as_u64())
                .map(|m| m as usize)
                .unwrap_or(50);
            
            match self.indexer.search_code(&query, max_results).await {
                Ok(results) => Ok(MCPResponse {
                    jsonrpc: "2.0".to_string(),
                    id,
                    result: Some(serde_json::json!({
                        "content": [{
                            "type": "text",
                            "text": serde_json::to_string(&results)?
                        }]
                    })),
                    error: None,
                }),
                Err(e) => Ok(self.error_response(
                    id,
                    -32603,
                    format!("Search failed: {}", e),
                )),
            }
        }
        // ... 其他工具处理
    }
}
```

### 3.4 resources/list（列出资源）

返回可用的资源列表。

**请求**：
```json
{
  "jsonrpc": "2.0",
  "id": 4,
  "method": "resources/list"
}
```

**响应**：
```json
{
  "jsonrpc": "2.0",
  "id": 4,
  "result": {
    "resources": [
      {
        "uri": "codebase://search",
        "name": "Code Search",
        "description": "Search the codebase",
        "mimeType": "application/json"
      },
      {
        "uri": "codebase://files",
        "name": "List Files",
        "description": "List files in the codebase",
        "mimeType": "application/json"
      }
    ]
  }
}
```

### 3.5 resources/read（读取资源）

读取资源内容。

**请求**：
```json
{
  "jsonrpc": "2.0",
  "id": 5,
  "method": "resources/read",
  "params": {
    "uri": "codebase://search"
  }
}
```

**响应**：
```json
{
  "jsonrpc": "2.0",
  "id": 5,
  "result": {
    "contents": [{
      "uri": "codebase://search",
      "mimeType": "application/json",
      "text": "{\"description\": \"Use the search_code tool to search the codebase\"}"
    }]
  }
}
```

## 4. 项目实现的工具

### 4.1 search_code（文本搜索）

**功能**：在代码库中搜索文本和模式。

**参数**：
- `query` (string, 必需)：搜索查询字符串
- `max_results` (number, 可选)：最大结果数（默认 50）

**返回**：搜索结果列表，包含文件路径、行号、行内容等。

### 4.2 search_by_embedding（语义搜索）

**功能**：基于向量相似度进行语义搜索。

**参数**：
- `query` (string, 必需)：自然语言查询
- `max_results` (number, 可选)：最大结果数（默认 10）

**返回**：代码块和相似度分数。

### 4.3 list_directory（列出目录）

**功能**：列出指定目录中的文件和子目录。

**参数**：
- `path` (string, 必需)：目录路径（相对于代码库根目录）

**返回**：文件和目录列表。

### 4.4 read_file（读取文件）

**功能**：读取文件内容。

**参数**：
- `file_path` (string, 必需)：文件路径（相对于代码库根目录）

**返回**：文件内容。

### 4.5 chunk_file（代码切片）

**功能**：对文件进行代码切片。

**参数**：
- `file_path` (string, 必需)：文件路径

**返回**：代码块列表。

### 4.6 chunk_files（批量代码切片）

**功能**：批量对多个文件进行代码切片。

**参数**：
- `file_paths` (array, 必需)：文件路径数组

**返回**：所有文件的代码块列表。

## 5. stdio 模式实现

### 5.1 通信方式

stdio 模式通过标准输入输出进行通信：

- **输入**：从 `stdin` 读取 JSON-RPC 请求（每行一个 JSON）
- **输出**：向 `stdout` 写入 JSON-RPC 响应

### 5.2 实现代码

```rust
pub async fn run(&self) -> Result<()> {
    let stdin = tokio::io::stdin();
    let stdout = tokio::io::stdout();
    
    let mut stdin_reader = BufReader::new(stdin);
    let mut stdout_writer = stdout;
    
    let mut initialized = false;
    let mut buffer = String::new();
    
    loop {
        buffer.clear();
        
        // 读取一行（一个 JSON-RPC 请求）
        let bytes_read = stdin_reader.read_line(&mut buffer).await?;
        
        if bytes_read == 0 {
            break;  // EOF
        }
        
        let line = buffer.trim();
        if line.is_empty() {
            continue;
        }
        
        // 解析 JSON-RPC 请求
        let request: MCPRequest = match serde_json::from_str(line) {
            Ok(req) => req,
            Err(e) => {
                tracing::warn!("Failed to parse request: {} - {}", e, line);
                continue;
            }
        };
        
        // 处理请求
        let response = self.handle_request(&request, &mut initialized).await?;
        
        // 发送响应（只有有 ID 的请求才需要响应）
        if request.id.is_some() {
            let response_json = serde_json::to_string(&response)?;
            stdout_writer.write_all(response_json.as_bytes()).await?;
            stdout_writer.write_all(b"\n").await?;
            stdout_writer.flush().await?;
        }
    }
    
    Ok(())
}
```

### 5.3 初始化状态管理

```rust
async fn handle_request(
    &self,
    request: &MCPRequest,
    initialized: &mut bool,
) -> Result<MCPResponse> {
    match request.method.as_str() {
        "initialize" => {
            if *initialized {
                return Ok(self.error_response(
                    request.id.clone(),
                    -32000,
                    "Already initialized".to_string(),
                ));
            }
            *initialized = true;
            Ok(self.initialize_response(request.id.clone()))
        }
        _ => {
            if !*initialized {
                return Ok(self.error_response(
                    request.id.clone(),
                    -32000,
                    "Not initialized".to_string(),
                ));
            }
            // 处理其他请求
        }
    }
}
```

## 6. HTTP 模式实现

### 6.1 通信方式

HTTP 模式通过 HTTP POST 请求进行通信：

- **端点**：`POST http://localhost:3000/`
- **Content-Type**：`application/json`
- **请求体**：JSON-RPC 请求
- **响应**：JSON-RPC 响应

### 6.2 实现代码

```rust
pub async fn run(self) -> Result<()> {
    let app = Router::new()
        .route("/", post(handle_mcp_request))      // MCP 端点
        .route("/health", get(health_check))       // 健康检查
        .layer(CorsLayer::new()
            .allow_origin(Any)
            .allow_methods(Any)
            .allow_headers(Any))
        .with_state(self.indexer);
    
    let addr = std::net::SocketAddr::from(([0, 0, 0, 0], self.port));
    tracing::info!("HTTP MCP Server listening on http://0.0.0.0:{}", self.port);
    
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    
    Ok(())
}

async fn handle_mcp_request(
    State(indexer): State<Arc<CodebaseIndexer>>,
    Json(request): Json<Value>,
) -> impl IntoResponse {
    let mcp_server = MCPServer::new(indexer);
    
    match mcp_server.handle_json_request(request).await {
        Ok(response) => (StatusCode::OK, Json(response)).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "jsonrpc": "2.0",
                "error": {
                    "code": -32603,
                    "message": e.to_string()
                }
            }))
        ).into_response(),
    }
}
```

### 6.3 健康检查端点

```rust
async fn health_check() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "ok",
        "service": "browser-mcp",
        "version": env!("CARGO_PKG_VERSION"),
        "mode": "http"
    }))
}
```

## 7. 错误处理

### 7.1 错误响应格式

```rust
fn error_response(&self, id: Option<Value>, code: i32, message: String) -> MCPResponse {
    MCPResponse {
        jsonrpc: "2.0".to_string(),
        id,
        result: None,
        error: Some(MCPError {
            code,
            message,
            data: None,
        }),
    }
}
```

### 7.2 常见错误场景

1. **未初始化**：调用除 `initialize` 之外的方法前必须初始化
2. **方法不存在**：返回 `-32601`
3. **参数无效**：返回 `-32602`
4. **内部错误**：工具执行失败时返回 `-32603`

## 8. 数据序列化

### 8.1 请求结构

```rust
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct MCPRequest {
    jsonrpc: String,
    id: Option<Value>,
    method: String,
    params: Option<Value>,
}
```

### 8.2 响应结构

```rust
#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct MCPResponse {
    jsonrpc: String,
    id: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<MCPError>,
}
```

### 8.3 命名约定

使用 `rename_all = "camelCase"` 确保 JSON 字段使用驼峰命名：

```rust
// Rust 字段：file_path
// JSON 字段：filePath
```

## 9. 与客户端集成

### 9.1 Claude Desktop 配置

```json
{
  "mcpServers": {
    "browser-mcp": {
      "command": "/path/to/browser-mcp/target/release/browser-mcp",
      "args": []
    }
  }
}
```

### 9.2 Cursor 配置

类似地，在 Cursor 的配置文件中添加服务器配置。

### 9.3 HTTP 模式使用

对于 HTTP 模式，客户端可以通过 HTTP 请求调用：

```bash
curl -X POST http://localhost:3000/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/list"
  }'
```

## 10. 协议扩展

### 10.1 添加新工具

要添加新工具，需要：

1. 在 `list_tools` 中添加工具定义
2. 在 `handle_tool_call` 中添加处理逻辑
3. 实现相应的业务逻辑

### 10.2 添加新资源

类似地，可以添加新的资源类型。

## 11. 总结

MCP 协议提供了标准化的方式让 AI 助手与外部工具交互：

1. **JSON-RPC 2.0**：标准的 RPC 协议
2. **工具和资源**：两种交互方式
3. **stdio/HTTP**：两种传输方式
4. **类型安全**：使用 Rust 的类型系统保证正确性

通过实现 MCP 协议，Browser MCP Server 可以与各种 AI 助手无缝集成，提供强大的代码库访问能力。

