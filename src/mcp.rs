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
    pub async fn handle_json_request(&self, request: Value) -> Result<Value> {
        let req: MCPRequest = serde_json::from_value(request)?;
        let mut initialized = true; // HTTP mode doesn't track initialization state
        let response = self.handle_request(&req, &mut initialized).await?;
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
                            "text": "{\"description\": \"Use the list_files tool to list files\"}"
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
            "list_files" => {
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
                        format!("List files failed: {}", e),
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
                        let results_json: Vec<serde_json::Value> = results
                            .into_iter()
                            .map(|(chunk, similarity)| {
                                serde_json::json!({
                                    "chunk": chunk,
                                    "similarity": similarity
                                })
                            })
                            .collect();
                        
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

