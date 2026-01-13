use crate::codebase::CodebaseIndexer;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct MCPRequest {
    jsonrpc: String,
    id: Option<Value>,
    method: String,
    params: Option<Value>,
}

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

#[derive(Debug, Serialize, Deserialize)]
struct MCPError {
    code: i32,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    data: Option<Value>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ReadResourceParams {
    uri: String,
}

use std::sync::Arc;

pub struct MCPServer {
    indexer: Arc<CodebaseIndexer>,
}

impl MCPServer {
    pub fn new(indexer: Arc<CodebaseIndexer>) -> Self {
        Self { indexer }
    }

    pub async fn run(&self) -> Result<()> {
        let stdin = tokio::io::stdin();
        let stdout = tokio::io::stdout();
        
        let mut stdin_reader = BufReader::new(stdin);
        let mut stdout_writer = stdout;
        
        let mut initialized = false;
        let mut buffer = String::new();
        
        loop {
            buffer.clear();
            let bytes_read = stdin_reader.read_line(&mut buffer).await?;
            
            if bytes_read == 0 {
                break;
            }
            
            let line = buffer.trim();
            if line.is_empty() {
                continue;
            }
            
            let request: MCPRequest = match serde_json::from_str(line) {
                Ok(req) => req,
                Err(e) => {
                    tracing::warn!("Failed to parse request: {} - {}", e, line);
                    continue;
                }
            };
            
            let response = self.handle_request(&request, &mut initialized).await?;
            
            if request.id.is_some() {
                let response_json = serde_json::to_string(&response)?;
                stdout_writer.write_all(response_json.as_bytes()).await?;
                stdout_writer.write_all(b"\n").await?;
                stdout_writer.flush().await?;
            }
        }
        
        Ok(())
    }

    /// Handle a JSON-RPC request (for HTTP mode)
    /// HTTP mode is stateless, so we don't track initialization state
    pub async fn handle_json_request(&self, request: Value) -> Result<Value> {
        let req: MCPRequest = serde_json::from_value(request)?;
        
        // In HTTP mode, we handle requests directly without initialization checks
        // Each HTTP request is independent
        let response = match req.method.as_str() {
            "initialize" => {
                // Always allow initialize in HTTP mode (stateless)
                self.initialize_response(req.id.clone())
            }
            "resources/list" => {
                // No initialization check needed in HTTP mode
                self.list_resources(req.id.clone()).await?
            }
            "resources/read" => {
                // No initialization check needed in HTTP mode
                self.read_resource(req.id.clone(), req.params.clone()).await?
            }
            "tools/list" => {
                // No initialization check needed in HTTP mode
                self.list_tools(req.id.clone()).await?
            }
            "tools/call" => {
                // No initialization check needed in HTTP mode
                self.handle_tool_call(req.id.clone(), req.params.clone()).await?
            }
            _ => {
                return Ok(serde_json::to_value(self.error_response(
                    req.id.clone(),
                    -32601,
                    format!("Method not found: {}", req.method),
                ))?);
            }
        };
        
        Ok(serde_json::to_value(response)?)
    }

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
            "resources/list" => {
                if !*initialized {
                    return Ok(self.error_response(
                        request.id.clone(),
                        -32000,
                        "Not initialized".to_string(),
                    ));
                }
                self.list_resources(request.id.clone()).await
            }
            "resources/read" => {
                if !*initialized {
                    return Ok(self.error_response(
                        request.id.clone(),
                        -32000,
                        "Not initialized".to_string(),
                    ));
                }
                self.read_resource(request.id.clone(), request.params.clone()).await
            }
            "tools/list" => {
                if !*initialized {
                    return Ok(self.error_response(
                        request.id.clone(),
                        -32000,
                        "Not initialized".to_string(),
                    ));
                }
                self.list_tools(request.id.clone()).await
            }
            "tools/call" => {
                if !*initialized {
                    return Ok(self.error_response(
                        request.id.clone(),
                        -32000,
                        "Not initialized".to_string(),
                    ));
                }
                self.handle_tool_call(request.id.clone(), request.params.clone()).await
            }
            _ => Ok(self.error_response(
                request.id.clone(),
                -32601,
                format!("Method not found: {}", request.method),
            )),
        }
    }

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

    async fn list_resources(&self, id: Option<Value>) -> Result<MCPResponse> {
        let resources = vec![
            serde_json::json!({
                "uri": "codebase://search",
                "name": "Code Search",
                "description": "Search the brave-browser codebase",
                "mimeType": "application/json"
            }),
            serde_json::json!({
                "uri": "codebase://files",
                "name": "List Files",
                "description": "List files in the codebase",
                "mimeType": "application/json"
            }),
        ];

        Ok(MCPResponse {
            jsonrpc: "2.0".to_string(),
            id,
            result: Some(serde_json::json!({
                "resources": resources
            })),
            error: None,
        })
    }

    async fn read_resource(
        &self,
        id: Option<Value>,
        params: Option<Value>,
    ) -> Result<MCPResponse> {
        let params: ReadResourceParams = match params {
            Some(p) => serde_json::from_value(p)
                .context("Failed to parse read resource params")?,
            None => {
                return Ok(self.error_response(
                    id,
                    -32602,
                    "Missing params".to_string(),
                ));
            }
        };

        match params.uri.as_str() {
            "codebase://search" => {
                Ok(MCPResponse {
                    jsonrpc: "2.0".to_string(),
                    id,
                    result: Some(serde_json::json!({
                        "contents": [{
                            "uri": params.uri,
                            "mimeType": "application/json",
                            "text": "{\"description\": \"Use the search_code tool to search the codebase\"}"
                        }]
                    })),
                    error: None,
                })
            }
            "codebase://files" => {
                Ok(MCPResponse {
                    jsonrpc: "2.0".to_string(),
                    id,
                    result: Some(serde_json::json!({
                        "contents": [{
                            "uri": params.uri,
                            "mimeType": "application/json",
                            "text": "{\"description\": \"Use the list_directory tool to list files\"}"
                        }]
                    })),
                    error: None,
                })
            }
            _ => {
                // Try to read as file path
                if params.uri.starts_with("file://") {
                    let file_path = params.uri.strip_prefix("file://").unwrap();
                    match self.indexer.read_file(file_path).await {
                        Ok(content) => Ok(MCPResponse {
                            jsonrpc: "2.0".to_string(),
                            id,
                            result: Some(serde_json::json!({
                                "contents": [{
                                    "uri": params.uri,
                                    "mimeType": "text/plain",
                                    "text": content
                                }]
                            })),
                            error: None,
                        }),
                        Err(e) => Ok(self.error_response(
                            id,
                            -32603,
                            format!("Failed to read file: {}", e),
                        )),
                    }
                } else {
                    Ok(self.error_response(
                        id,
                        -32602,
                        format!("Unknown resource URI: {}", params.uri),
                    ))
                }
            }
        }
    }

    async fn list_tools(&self, id: Option<Value>) -> Result<MCPResponse> {
        let tools = vec![
            serde_json::json!({
                "name": "search_code",
                "description": "Search for code using text matching. This is useful for finding specific strings, variable names, or simple patterns.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The text or pattern to search for"
                        },
                        "max_results": {
                            "type": "number",
                            "description": "Maximum number of results to return (default: 50)"
                        }
                    },
                    "required": ["query"]
                }
            }),
            serde_json::json!({
                "name": "search_by_embedding",
                "description": "Search for semantically similar code using vector embeddings. This is best for finding code based on functionality or concept (e.g., 'how is video decoding implemented').",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The natural language query describing what you are looking for"
                        },
                        "max_results": {
                            "type": "number",
                            "description": "Maximum number of results to return (default: 10)"
                        }
                    },
                    "required": ["query"]
                }
            }),
            serde_json::json!({
                "name": "list_directory",
                "description": "List all files and directories in a given path",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "The relative path to list (e.g., 'src/components')"
                        }
                    },
                    "required": ["path"]
                }
            }),
            serde_json::json!({
                "name": "read_file",
                "description": "Read the content of a file",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "The relative path of the file to read"
                        }
                    },
                    "required": ["file_path"]
                }
            }),
        ];

        Ok(MCPResponse {
            jsonrpc: "2.0".to_string(),
            id,
            result: Some(serde_json::json!({
                "tools": tools
            })),
            error: None,
        })
    }

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
            "list_directory" => {
                let path = arguments
                    .and_then(|a| a.get("path"))
                    .and_then(|p| p.as_str())
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| "".to_string());
                
                match self.indexer.list_files(&path).await {
                    Ok(files) => Ok(MCPResponse {
                        jsonrpc: "2.0".to_string(),
                        id,
                        result: Some(serde_json::json!({
                            "content": [{
                                "type": "text",
                                "text": serde_json::to_string(&files)?
                            }]
                        })),
                        error: None,
                    }),
                    Err(e) => Ok(self.error_response(
                        id,
                        -32603,
                        format!("List directory failed: {}", e),
                    )),
                }
            }
            "read_file" => {
                let file_path = arguments
                    .and_then(|a| a.get("file_path"))
                    .and_then(|p| p.as_str())
                    .context("Missing file_path argument")?;
                
                match self.indexer.read_file(file_path).await {
                    Ok(content) => Ok(MCPResponse {
                        jsonrpc: "2.0".to_string(),
                        id,
                        result: Some(serde_json::json!({
                            "content": [{
                                "type": "text",
                                "text": content
                            }]
                        })),
                        error: None,
                    }),
                    Err(e) => Ok(self.error_response(
                        id,
                        -32603,
                        format!("Read file failed: {}", e),
                    )),
                }
            }
            "chunk_file" => {
                let file_path = arguments
                    .and_then(|a| a.get("file_path"))
                    .and_then(|p| p.as_str())
                    .context("Missing file_path argument")?;
                
                match self.indexer.chunk_file(file_path).await {
                    Ok(chunks) => Ok(MCPResponse {
                        jsonrpc: "2.0".to_string(),
                        id,
                        result: Some(serde_json::json!({
                            "content": [{
                                "type": "text",
                                "text": serde_json::to_string(&chunks)?
                            }]
                        })),
                        error: None,
                    }),
                    Err(e) => Ok(self.error_response(
                        id,
                        -32603,
                        format!("Chunk file failed: {}", e),
                    )),
                }
            }
            "chunk_files" => {
                let file_paths = arguments
                    .and_then(|a| a.get("file_paths"))
                    .and_then(|p| p.as_array())
                    .and_then(|arr| {
                        arr.iter()
                            .map(|v| v.as_str().map(|s| s.to_string()))
                            .collect::<Option<Vec<String>>>()
                    })
                    .context("Missing or invalid file_paths argument")?;
                
                match self.indexer.chunk_files(&file_paths).await {
                    Ok(chunks) => Ok(MCPResponse {
                        jsonrpc: "2.0".to_string(),
                        id,
                        result: Some(serde_json::json!({
                            "content": [{
                                "type": "text",
                                "text": serde_json::to_string(&chunks)?
                            }]
                        })),
                        error: None,
                    }),
                    Err(e) => Ok(self.error_response(
                        id,
                        -32603,
                        format!("Chunk files failed: {}", e),
                    )),
                }
            }
            "search_by_embedding" => {
                let query = arguments
                    .and_then(|a| a.get("query"))
                    .and_then(|q| q.as_str())
                    .context("Missing query argument")?;
                
                let max_results = arguments
                    .and_then(|a| a.get("max_results"))
                    .and_then(|m| m.as_u64())
                    .map(|m| m as usize)
                    .unwrap_or(10);
                
                match self.indexer.search_by_embedding(query, max_results).await {
                    Ok(results) => {
                        // 将结果转换为 JSON 格式，包含相似度分数
                        // 同时将绝对路径转换为相对路径，方便 Cursor 使用 read_file 工具
                        // 如果代码片段太短，自动读取文件获取更多上下文
                        let mut results_json = Vec::new();
                        
                        for (chunk, similarity) in results {
                            // 将绝对路径转换为相对路径（相对于代码库根目录）
                            let relative_path = if std::path::Path::new(&chunk.file_path).is_absolute() {
                                // 尝试从常见路径模式中提取相对路径
                                if let Some(idx) = chunk.file_path.find("/src/") {
                                    chunk.file_path[idx + 5..].to_string()
                                } else if let Some(idx) = chunk.file_path.find("/brave_browser/") {
                                    chunk.file_path[idx + 15..].to_string()
                                } else {
                                    chunk.file_path.clone()
                                }
                            } else {
                                chunk.file_path.clone()
                            };
                            
                            // content 已经存储在 Qdrant 中，直接使用，无需读取文件
                            // 如果用户需要更多上下文，可以使用 file_path 调用 read_file 工具
                            results_json.push(serde_json::json!({
                                "chunk": {
                                    "file_path": relative_path,  // 相对路径，需要更多上下文时可调用 read_file 工具
                                    "content": chunk.content,  // 从 Qdrant 中获取的代码片段（已存储）
                                    "language": chunk.language,
                                    "start_line": chunk.start_line,
                                    "end_line": chunk.end_line,
                                    "node_type": chunk.node_type,
                                    "node_name": chunk.node_name,
                                },
                                "similarity": similarity
                            }));
                        }
                        
                        Ok(MCPResponse {
                            jsonrpc: "2.0".to_string(),
                            id,
                            result: Some(serde_json::json!({
                                "content": [{
                                    "type": "text",
                                    "text": serde_json::to_string(&results_json)?
                                }]
                            })),
                            error: None,
                        })
                    }
                    Err(e) => Ok(self.error_response(
                        id,
                        -32603,
                        format!("Vector search failed: {}", e),
                    )),
                }
            }
            _ => Ok(self.error_response(
                id,
                -32601,
                format!("Unknown tool: {}", tool_name),
            )),
        }
    }

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
}

