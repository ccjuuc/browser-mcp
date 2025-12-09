use crate::app_config::CodebaseConfig;
use crate::embedding::Embedder;
use crate::indexer_state::IndexerState;
use crate::parser::{CodeChunk, CodeParser};
use crate::storage::qdrant::QdrantStorage;
use anyhow::{Context, Result};
use ignore::WalkBuilder;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;
use tokio::fs;
use tokio::sync::RwLock;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub file_path: String,
    pub line_number: usize,
    pub line_content: String,
    pub context: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileInfo {
    pub path: String,
    pub size: u64,
    pub is_directory: bool,
}

pub struct CodebaseIndexer {
    codebase_path: PathBuf,
    config: CodebaseConfig,
    embedder: Option<Embedder>,
    qdrant: Option<Arc<QdrantStorage>>,
    #[allow(dead_code)]
    index: Arc<RwLock<Option<Vec<String>>>>, // For future indexing
}

impl CodebaseIndexer {
    #[allow(dead_code)]
    pub fn new(codebase_path: PathBuf) -> Self {
        Self::with_config(codebase_path, CodebaseConfig::default())
    }

    #[allow(dead_code)]
    pub fn with_config(codebase_path: PathBuf, config: CodebaseConfig) -> Self {
        Self::with_embedding_config(codebase_path, config, crate::embedding::EmbeddingConfig::default(), None)
    }

    pub fn with_embedding_config(
        codebase_path: PathBuf,
        config: CodebaseConfig,
        embedding_config: crate::embedding::EmbeddingConfig,
        qdrant: Option<Arc<QdrantStorage>>,
    ) -> Self {
        let embedder = if config.enable_embedding {
            Embedder::new(embedding_config).ok()
        } else {
            None
        };
        
        Self {
            codebase_path,
            config,
            embedder,
            qdrant,
            index: Arc::new(RwLock::new(None)),
        }
    }

    pub fn is_qdrant_enabled(&self) -> bool {
        self.qdrant.is_some()
    }

    pub async fn search_code(&self, query: &str, max_results: usize) -> Result<Vec<SearchResult>> {
        let query_lower = query.to_lowercase();
        let regex = Regex::new(&regex::escape(&query_lower))?;
        
        let mut results = Vec::new();
        let codebase_path = self.codebase_path.clone();

        // Use ignore crate to walk the directory, respecting .gitignore
        let walker = WalkBuilder::new(&codebase_path)
            .hidden(false)
            .git_ignore(true)
            .git_exclude(true)
            .build();

        for entry in walker {
            let entry = entry?;
            let path = entry.path();
            
            if !path.is_file() {
                continue;
            }

            // Skip binary files based on config
            if let Some(ext) = path.extension() {
                let ext_str = ext.to_string_lossy().to_lowercase();
                if self.config.ignored_extensions.iter().any(|ignored| ignored == &ext_str) {
                    continue;
                }
            }

            // Check file size based on config
            if let Ok(metadata) = path.metadata() {
                if metadata.len() > self.config.max_file_size {
                    continue;
                }
            }

            if results.len() >= max_results {
                break;
            }

            match self.search_in_file(path, &regex, &query_lower).await {
                Ok(mut file_results) => {
                    results.append(&mut file_results);
                }
                Err(e) => {
                    tracing::debug!("Error searching in file {:?}: {}", path, e);
                }
            }
        }

        results.truncate(max_results);
        Ok(results)
    }

    async fn search_in_file(
        &self,
        path: &Path,
        regex: &Regex,
        query: &str,
    ) -> Result<Vec<SearchResult>> {
        let content = fs::read_to_string(path).await?;
        let lines: Vec<&str> = content.lines().collect();
        let mut results = Vec::new();

        for (line_num, line) in lines.iter().enumerate() {
            let line_lower = line.to_lowercase();
            if regex.is_match(&line_lower) || line_lower.contains(query) {
                let relative_path = path
                    .strip_prefix(&self.codebase_path)
                    .unwrap_or(path)
                    .to_string_lossy()
                    .to_string();

                let context = if line_num > 0 && line_num < lines.len() - 1 {
                    Some(format!(
                        "{}: {}\n{}: {}\n{}: {}",
                        line_num,
                        lines[line_num.saturating_sub(1)],
                        line_num + 1,
                        line,
                        line_num + 2,
                        lines.get(line_num + 1).unwrap_or(&"")
                    ))
                } else {
                    None
                };

                results.push(SearchResult {
                    file_path: relative_path,
                    line_number: line_num + 1,
                    line_content: line.to_string(),
                    context,
                });
            }
        }

        Ok(results)
    }

    pub async fn list_files(&self, path: &str) -> Result<Vec<FileInfo>> {
        let target_path = if path.is_empty() {
            self.codebase_path.clone()
        } else {
            self.codebase_path.join(path)
        };

        if !target_path.exists() {
            return Err(anyhow::anyhow!("Path does not exist: {}", path));
        }

        let mut files = Vec::new();
        let mut entries = fs::read_dir(&target_path).await?;

        while let Some(entry) = entries.next_entry().await? {
            let entry_path = entry.path();
            let metadata = entry.metadata().await?;

            let relative_path = entry_path
                .strip_prefix(&self.codebase_path)
                .unwrap_or(&entry_path)
                .to_string_lossy()
                .to_string();

            files.push(FileInfo {
                path: relative_path,
                size: metadata.len(),
                is_directory: metadata.is_dir(),
            });
        }

        files.sort_by(|a, b| {
            a.is_directory
                .cmp(&b.is_directory)
                .reverse()
                .then_with(|| a.path.cmp(&b.path))
        });

        Ok(files)
    }

    pub async fn read_file(&self, file_path: &str) -> Result<String> {
        let full_path = self.codebase_path.join(file_path);
        
        // Security check: ensure the path is within codebase
        if !full_path.starts_with(&self.codebase_path) {
            return Err(anyhow::anyhow!("Path outside codebase: {}", file_path));
        }

        fs::read_to_string(&full_path)
            .await
            .with_context(|| format!("Failed to read file: {}", file_path))
    }

    /// 对文件进行代码切片
    pub async fn chunk_file(&self, file_path: &str) -> Result<Vec<CodeChunk>> {
        let full_path = self.codebase_path.join(file_path);
        
        // Security check
        if !full_path.starts_with(&self.codebase_path) {
            return Err(anyhow::anyhow!("Path outside codebase: {}", file_path));
        }

        if !self.config.enable_chunking {
            // 如果未启用切片，返回空结果
            return Ok(Vec::new());
        }

        let content = fs::read_to_string(&full_path)
            .await
            .with_context(|| format!("Failed to read file: {}", file_path))?;

        // 在 tokio 运行时中，我们需要在 blocking 线程中运行 tree-sitter
        // 因为 tree-sitter 是同步的
        let path_clone = full_path.clone();
        let content_clone = content.clone();
        let chunk_size = self.config.chunk_size;

        let chunks = tokio::task::spawn_blocking(move || {
            let mut parser = CodeParser::new();
            parser.parse_file(&path_clone, &content_clone, chunk_size)
        })
        .await
        .context("Failed to join parsing task")?
        .context("Failed to parse file")?;

        // 如果启用了向量化，为每个切片生成向量
        let mut chunks_with_embeddings = chunks;
        if self.config.enable_embedding {
            if let Some(ref embedder) = self.embedder {
                // 批量生成向量以提高效率
                let texts: Vec<String> = chunks_with_embeddings
                    .iter()
                    .map(|chunk| {
                        // 构建用于向量化的文本（包含文件路径、节点类型、节点名称和内容）
                        let mut text = String::new();
                        if let Some(ref name) = chunk.node_name {
                            text.push_str(&format!("{} ", name));
                        }
                        text.push_str(&chunk.node_type);
                        text.push_str(" ");
                        text.push_str(&chunk.content);
                        text
                    })
                    .collect();

                match embedder.embed_batch(&texts).await {
                    Ok(embeddings) => {
                        for (chunk, embedding) in chunks_with_embeddings.iter_mut().zip(embeddings.iter()) {
                            chunk.embedding = Some(embedding.clone());
                        }
                    }
                    Err(e) => {
                        tracing::warn!("Failed to generate embeddings: {}", e);
                    }
                }
            }
        }

        Ok(chunks_with_embeddings)
    }

    /// 批量对多个文件进行代码切片
    pub async fn chunk_files(&self, file_paths: &[String]) -> Result<Vec<CodeChunk>> {
        let mut all_chunks = Vec::new();

        for file_path in file_paths {
            match self.chunk_file(file_path).await {
                Ok(mut chunks) => {
                    all_chunks.append(&mut chunks);
                }
                Err(e) => {
                    tracing::warn!("Failed to chunk file {}: {}", file_path, e);
                }
            }
        }

        Ok(all_chunks)
    }

    /// 索引整个代码库并存储到 Qdrant
    pub async fn index_codebase(&self) -> Result<()> {
        if let Some(ref qdrant) = self.qdrant {
            tracing::info!("Starting codebase indexing to Qdrant...");
            
            // Load index state - 放在当前工作目录（项目根目录）
            let state_path = std::env::current_dir()
                .unwrap_or_else(|_| std::path::PathBuf::from("."))
                .join("browser-mcp-index.json");
            
            tracing::info!("Index state file: {:?}", state_path);
            
            let mut state = IndexerState::load(&state_path).unwrap_or_else(|e| {
                tracing::warn!("Failed to load index state, starting fresh: {}", e);
                IndexerState::new()
            });

            let codebase_path = self.codebase_path.clone();
            tracing::info!("Scanning codebase at: {:?}", codebase_path);
            
            // 先收集所有需要处理的文件路径
            tracing::info!("Collecting files to index...");
            let mut files_to_process: Vec<(PathBuf, u64, u64)> = Vec::new(); // (path, modified_time, file_size)
            
            let walker = WalkBuilder::new(&codebase_path)
                .hidden(false)
                .git_ignore(true)
                .git_exclude(true)
                .build();
            
            for entry in walker {
                let entry = entry?;
                let path = entry.path();
                
                if !path.is_file() {
                    continue;
                }
                
                // Filtering
                if let Some(ext) = path.extension() {
                    let ext_str = ext.to_string_lossy().to_lowercase();
                    if self.config.ignored_extensions.iter().any(|ignored| ignored == &ext_str) {
                        continue;
                    }
                }
                
                let metadata = match path.metadata() {
                    Ok(m) => m,
                    Err(_) => continue,
                };

                if metadata.len() > self.config.max_file_size {
                    continue;
                }

                let modified_time = metadata
                    .modified()
                    .unwrap_or(SystemTime::UNIX_EPOCH)
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                
                let file_size = metadata.len();
                
                files_to_process.push((path.to_path_buf(), modified_time, file_size));
            }
            
            let total_files = files_to_process.len();
            tracing::info!("Found {} files to process", total_files);
            tracing::info!("Starting indexing (progress will be updated for each file)...");

            let mut indexed_count = 0;
            let mut skipped_count = 0;
            let mut model_embedded_count = 0; // 使用模型向量化的文件数量
            let mut processed_count = 0; // 已处理的文件数（包括跳过的）
            
            // 处理每个文件
            for (path, modified_time, file_size) in files_to_process {
                let relative_path = path
                    .strip_prefix(&self.codebase_path)
                    .unwrap_or(&path)
                    .to_string_lossy()
                    .to_string();

                processed_count += 1;
                
                if !state.should_index(&relative_path, modified_time, file_size) {
                    skipped_count += 1;
                    // 每处理一个文件就更新一次进度
                    eprint!("\r[Indexing] Processed: {}/{} | Indexed: {} | Skipped: {} | Model-embedded: {}... ", 
                           processed_count, total_files, indexed_count, skipped_count, model_embedded_count);
                    use std::io::Write;
                    let _ = std::io::stderr().flush();
                    continue;
                }

                match self.chunk_file(&relative_path).await {
                    Ok(chunks) => {
                        if !chunks.is_empty() {
                            // 检查是否使用了模型向量化（而不是 TF-IDF）
                            // 只有当 chunks 有 embedding 且 embedder 使用模型时才计数
                            let has_embeddings = chunks.iter().any(|chunk| chunk.embedding.is_some());
                            if has_embeddings {
                                if let Some(ref embedder) = self.embedder {
                                    if embedder.is_using_model() {
                                        model_embedded_count += 1;
                                    }
                                }
                            }
                            
                            qdrant.upsert_chunks(chunks).await?;
                            indexed_count += 1;
                            
                            // Update state in memory
                            state.update(relative_path.clone(), modified_time, file_size);
                            
                            // Save state periodically (every 50 files) to avoid too frequent IO
                            if indexed_count % 50 == 0 {
                                if let Err(e) = state.save(&state_path) {
                                    tracing::warn!("Failed to save index state: {}", e);
                                }
                            }
                        }
                        
                        // 每处理一个文件就更新一次进度
                        eprint!("\r[Indexing] Processed: {}/{} | Indexed: {} | Skipped: {} | Model-embedded: {}... ", 
                               processed_count, total_files, indexed_count, skipped_count, model_embedded_count);
                        use std::io::Write;
                        let _ = std::io::stderr().flush();
                    }
                    Err(e) => {
                        // 降低日志级别，避免刷屏
                        tracing::debug!("Failed to index file {}: {}", relative_path, e);
                        // 即使失败也要更新进度
                        eprint!("\r[Indexing] Processed: {}/{} | Indexed: {} | Skipped: {} | Model-embedded: {}... ", 
                               processed_count, total_files, indexed_count, skipped_count, model_embedded_count);
                        use std::io::Write;
                        let _ = std::io::stderr().flush();
                    }
                }
            }
            
            // 清除进度行，打印最终结果
            eprint!("\r");
            
            // Final save
            if let Err(e) = state.save(&state_path) {
                tracing::warn!("Failed to save final index state: {}", e);
            }

            tracing::info!(
                "Indexing completed. Indexed: {}, Skipped: {}, Model-embedded: {}",
                indexed_count, skipped_count, model_embedded_count
            );
        } else {
            tracing::warn!("Qdrant storage not configured, skipping indexing.");
        }
        Ok(())
    }

    /// 基于向量相似度搜索代码切片
    pub async fn search_by_embedding(
        &self,
        query: &str,
        max_results: usize,
    ) -> Result<Vec<(CodeChunk, f32)>> {
        if !self.config.enable_embedding {
            return Err(anyhow::anyhow!("Embedding is not enabled"));
        }

        let embedder = self
            .embedder
            .as_ref()
            .context("Embedder not initialized")?;

        // 生成查询向量
        let query_embedding = embedder.embed(query).await?;
        
        // 1. 如果配置了 Qdrant，使用 Qdrant 搜索
        if let Some(ref qdrant) = self.qdrant {
             return qdrant.search(query_embedding, max_results as u64).await;
        }

        // 2. 否则使用本地线性扫描（慢）
        tracing::warn!("Qdrant not configured, falling back to slow linear scan");

        // 遍历所有文件，收集带向量的切片
        let mut candidates: Vec<(CodeChunk, f32)> = Vec::new();
        let codebase_path = self.codebase_path.clone();

        let walker = WalkBuilder::new(&codebase_path)
            .hidden(false)
            .git_ignore(true)
            .git_exclude(true)
            .build();

        for entry in walker {
            let entry = entry?;
            let path = entry.path();

            if !path.is_file() {
                continue;
            }

            // 跳过二进制文件
            if let Some(ext) = path.extension() {
                let ext_str = ext.to_string_lossy().to_lowercase();
                if self.config.ignored_extensions.iter().any(|ignored| ignored == &ext_str) {
                    continue;
                }
            }

            // 检查文件大小
            if let Ok(metadata) = path.metadata() {
                if metadata.len() > self.config.max_file_size {
                    continue;
                }
            }

            // 对文件进行切片
            let relative_path = path
                .strip_prefix(&self.codebase_path)
                .unwrap_or(path)
                .to_string_lossy()
                .to_string();

            if let Ok(mut chunks) = self.chunk_file(&relative_path).await {
                for chunk in chunks.drain(..) {
                    if let Some(ref embedding) = chunk.embedding {
                        let similarity = Embedder::cosine_similarity(&query_embedding, embedding);
                        candidates.push((chunk, similarity));
                    }
                }
            }
        }

        // 按相似度排序并返回前 N 个结果
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(max_results);

        Ok(candidates)
    }
}

