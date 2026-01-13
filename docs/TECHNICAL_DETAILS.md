# Browser MCP 技术实现详解

本文档详细介绍了 Browser MCP 服务器的技术实现细节，包括技术选型、架构设计、依赖库说明等，旨在帮助开发小白深入理解整个项目的技术方案。

## 目录

1. [项目概述](#项目概述)
2. [技术架构](#技术架构)
3. [核心模块详解](#核心模块详解)
4. [依赖库详解](#依赖库详解)
5. [技术选型原因](#技术选型原因)
6. [数据流](#数据流)
7. [性能优化](#性能优化)

---

## 项目概述

Browser MCP 是一个基于 Rust 实现的 Model Context Protocol (MCP) 服务器，用于对大型代码库进行智能索引和语义搜索。它支持多种编程语言，使用 Tree-sitter 进行代码解析，使用向量嵌入进行语义搜索，使用 Qdrant 进行向量存储。

### 核心功能

- **代码索引**: 使用 Tree-sitter 解析代码，提取语义单元
- **向量化**: 使用 CodeBERT 模型或 TF-IDF 生成代码向量
- **向量搜索**: 使用 Qdrant 进行高效的相似度搜索
- **文本搜索**: 使用正则表达式进行精确匹配搜索
- **MCP 协议**: 实现标准 MCP 协议，支持 stdio 和 HTTP 两种模式

---

## 技术架构

### 整体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    MCP Client (Cursor/Claude)                │
└───────────────────────┬───────────────────────────────────────┘
                        │ JSON-RPC 2.0
                        │
        ┌───────────────▼───────────────┐
        │    MCP Server (mcp.rs)        │
        │  - 协议处理                   │
        │  - 请求路由                   │
        │  - 响应格式化                 │
        └───────────────┬───────────────┘
                        │
        ┌───────────────▼───────────────┐
        │  CodebaseIndexer (codebase.rs)│
        │  - 文件搜索                   │
        │  - 文件读取                   │
        │  - 目录列表                   │
        └───────────────┬───────────────┘
                        │
        ┌───────────────▼───────────────┐
        │   CodeParser (parser.rs)      │
        │  - Tree-sitter 解析           │
        │  - AST 遍历                   │
        │  - 代码切片                   │
        └───────────────┬───────────────┘
                        │
        ┌───────────────▼───────────────┐
        │   Embedder (embedding.rs)     │
        │  - CodeBERT 模型加载           │
        │  - 向量生成                   │
        │  - TF-IDF 降级                │
        └───────────────┬───────────────┘
                        │
        ┌───────────────▼───────────────┐
        │  QdrantStorage (storage/)     │
        │  - 向量存储                   │
        │  - 相似度搜索                 │
        └───────────────────────────────┘
```

### 模块关系

```
main.rs (入口)
├── app_config.rs (配置管理)
├── mcp.rs (MCP 协议实现)
├── http_server.rs (HTTP 服务器)
├── codebase.rs (代码库操作)
│   ├── parser.rs (代码解析)
│   ├── embedding.rs (向量化)
│   └── storage/qdrant.rs (向量存储)
└── indexer_state.rs (索引状态管理)
```

---

## 核心模块详解

### 1. main.rs - 程序入口

**职责**: 
- 解析命令行参数
- 加载配置文件
- 初始化日志系统
- 启动 MCP 服务器（stdio 或 HTTP 模式）

**关键代码**:
```rust
#[tokio::main]  // 使用 tokio 异步运行时
async fn main() -> Result<()> {
    let config = Config::load()?;  // 加载配置
    // 初始化日志
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_new(&log_level)
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::from_default_env())
        )
        .init();
    // 创建索引器
    let indexer = Arc::new(CodebaseIndexer::with_embedding_config(...));
    // 启动服务器
    if http_mode {
        HttpServer::new(indexer, port).run().await?;
    } else {
        MCPServer::new(indexer).run().await?;
    }
}
```

**为什么使用 `#[tokio::main]`?**
- Rust 的异步编程需要运行时支持
- Tokio 是最成熟的 Rust 异步运行时
- 允许使用 `async/await` 语法处理 I/O 操作

**日志输出位置**:
- **默认行为**: 使用 `tracing_subscriber::fmt()` 将日志输出到**标准错误流 (stderr)**
- **stdio 模式**: 日志输出到控制台（终端）
- **HTTP 模式**: 日志也输出到控制台（终端）
- **当前实现**: **不写入日志文件**，所有日志都输出到终端

**如何将日志写入文件**:
如果需要将日志写入文件，可以修改日志初始化代码：

```rust
use tracing_subscriber::fmt::writer::BoxMakeWriter;
use std::fs::OpenOptions;

// 创建日志文件（追加模式）
let log_file = OpenOptions::new()
    .create(true)
    .append(true)
    .open("browser-mcp.log")?;

tracing_subscriber::fmt()
    .with_writer(BoxMakeWriter::new(log_file))  // 写入文件
    .with_env_filter(...)
    .init();
```

**日志文件位置**（如果配置了文件输出）:
- 默认会在**当前工作目录**创建日志文件
- 可以通过绝对路径指定日志文件位置

---

### 2. app_config.rs - 配置管理

**职责**:
- 从 TOML 文件加载配置
- 提供默认配置
- 支持多代码库路径配置

**关键数据结构**:
```rust
pub struct Config {
    pub codebase: CodebaseConfig,      // 代码库配置
    pub server: ServerConfig,          // 服务器配置
    pub embedding: EmbeddingConfig,    // 向量化配置
    pub qdrant: QdrantSettings,       // Qdrant 配置
}

pub struct CodebaseConfig {
    pub path: Vec<String>,            // 支持多个代码库路径
    pub ignored_extensions: Vec<String>,  // 忽略的文件扩展名
    pub max_file_size: u64,           // 最大文件大小
    pub chunk_size: usize,            // 代码切片大小
    pub enable_chunking: bool,        // 是否启用代码切片
    pub enable_embedding: bool,       // 是否启用向量化
}
```

**配置查找顺序**（从高到低）:
1. **环境变量 `BROWSER_MCP_CONFIG`** 指定的路径（最高优先级）
2. **当前目录的 `browser-mcp.toml`**
3. **默认配置**（如果以上都没找到）

**配置文件位置**:
- 推荐：在当前工作目录创建 `browser-mcp.toml`
- 环境变量：通过 `BROWSER_MCP_CONFIG` 指定自定义路径
- 如果都没找到，使用内置默认配置

**配置优先级说明**：
- 环境变量优先级最高，可以覆盖所有其他配置
- 当前目录的 `browser-mcp.toml` 作为项目配置
- 一旦找到配置文件就使用，不再查找后续位置

**为什么使用 TOML?**
- 人类可读，易于编辑
- 支持嵌套结构
- Rust 有成熟的 TOML 解析库 (`toml`)

---

### 3. mcp.rs - MCP 协议实现

**职责**:
- 实现 JSON-RPC 2.0 协议
- 处理 MCP 标准方法（initialize, tools/list, tools/call 等）
- 支持 stdio 和 HTTP 两种传输模式

**关键方法**:
```rust
// 处理 JSON-RPC 请求
async fn handle_request(&self, request: &MCPRequest) -> Result<MCPResponse>

// 列出可用工具
async fn list_tools(&self) -> Result<MCPResponse>

// 处理工具调用
async fn handle_tool_call(&self, tool_name: &str, arguments: Value) -> Result<MCPResponse>
```

**支持的 MCP 方法**:
- `initialize`: 初始化连接
- `tools/list`: 列出可用工具
- `tools/call`: 调用工具（search_code, search_by_embedding, read_file, list_directory）
- `resources/list`: 列出资源
- `resources/read`: 读取资源

**为什么手动实现 MCP?**
- MCP 协议相对简单（JSON-RPC 2.0）
- 没有成熟的 Rust MCP SDK
- 手动实现可以更好地控制行为
- 减少依赖，提高性能

---

### 4. http_server.rs - HTTP 服务器

**职责**:
- 提供 HTTP 接口（作为 stdio 的替代）
- 处理 CORS
- 健康检查端点

**技术栈**:
- **Axum**: 现代、高性能的 Rust Web 框架
- **Tower**: 中间件生态系统
- **Tower-HTTP**: HTTP 特定中间件（CORS）

**关键代码**:
```rust
let app = Router::new()
    .route("/", post(handle_mcp_request))      // MCP 端点
    .route("/health", get(health_check))       // 健康检查
    .layer(CorsLayer::new()                    // CORS 支持
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any))
    .with_state(indexer);                      // 共享状态
```

**为什么选择 Axum?**
- 基于 Tokio，性能优秀
- 类型安全的路由系统
- 良好的错误处理
- 活跃的社区

---

### 5. codebase.rs - 代码库操作核心

**职责**:
- 文件搜索（文本匹配）
- 文件读取
- 目录列表
- 代码库索引
- 向量搜索协调

**关键方法**:
```rust
// 文本搜索
pub async fn search_code(&self, query: &str, max_results: usize) -> Result<Vec<SearchResult>>

// 向量搜索
pub async fn search_by_embedding(&self, query: &str, max_results: usize) -> Result<Vec<(CodeChunk, f32)>>

// 索引代码库
pub async fn index_codebase(&self) -> Result<()>
```

**多路径支持**:
- 支持配置多个代码库路径
- 搜索时遍历所有路径
- 索引时处理所有路径

**为什么使用 `ignore` crate?**
- 自动处理 `.gitignore` 规则
- 高效的文件遍历
- 支持隐藏文件过滤

---

### 6. parser.rs - 代码解析

**职责**:
- 使用 Tree-sitter 解析代码
- 提取 AST 节点（函数、类、结构体等）
- 生成代码切片

**支持的编程语言**:
- Rust, JavaScript, TypeScript, Python
- C/C++/Objective-C++, Java, Go
- JSON, YAML, HTML, CSS

**关键数据结构**:
```rust
pub struct CodeChunk {
    pub file_path: String,      // 文件路径
    pub language: String,       // 语言类型
    pub content: String,        // 代码内容
    pub start_line: usize,     // 起始行号
    pub end_line: usize,        // 结束行号
    pub node_type: String,      // AST 节点类型
    pub node_name: Option<String>, // 节点名称（函数名、类名等）
    pub embedding: Option<Vec<f32>>, // 向量嵌入（可选）
}
```

**解析流程**:
1. 根据文件扩展名确定语言
2. 使用对应的 Tree-sitter 语言解析器
3. 遍历 AST，提取感兴趣的节点
4. 生成代码切片

**为什么使用 Tree-sitter?**
- 增量解析（只解析变更部分）
- 支持多种语言
- 提供准确的 AST
- 性能优秀

**Tree-sitter 工作原理**:
```
源代码 → Tree-sitter Parser → AST (抽象语法树)
                                    ↓
                            遍历 AST 节点
                                    ↓
                            提取代码切片
```

---

### 7. embedding.rs - 向量化

**职责**:
- 加载 CodeBERT 模型
- 生成代码向量嵌入
- TF-IDF 降级方案

**技术方案**:
1. **优先方案**: CodeBERT + Candle
   - 使用 Hugging Face 的 CodeBERT 模型
   - 通过 Candle 框架加载和推理
   - 支持 CPU、CUDA、Metal 后端

2. **降级方案**: TF-IDF
   - 如果模型加载失败，自动降级
   - 基于词频的简单向量化
   - 无需额外依赖

**关键代码**:
```rust
pub struct Embedder {
    config: EmbeddingConfig,
    model: Option<Arc<BertModel>>,    // CodeBERT 模型
    tokenizer: Option<Arc<Tokenizer>>, // 分词器
    device: Device,                    // 计算设备（CPU/GPU）
}

// 生成向量
pub async fn embed(&self, text: &str) -> Result<Vec<f32>>
pub async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>
```

**为什么选择 CodeBERT?**
- 专门为代码设计的 BERT 模型
- 在代码理解任务上表现优秀
- 支持多种编程语言
- 模型大小适中（~500MB）

**为什么使用 Candle?**
- 纯 Rust 实现，无 Python 依赖
- 支持多种后端（CPU、CUDA、Metal）
- 性能优秀
- 内存安全

**向量生成流程**:
```
代码文本 → Tokenizer (分词) → Token IDs
                                    ↓
                            CodeBERT 模型
                                    ↓
                            向量嵌入 (768维)
```

---

### 8. storage/qdrant.rs - 向量存储

**职责**:
- 连接 Qdrant 服务器
- 创建集合（Collection）
- 存储代码切片和向量
- 执行向量相似度搜索

**关键方法**:
```rust
// 初始化集合
pub async fn init(&self) -> Result<()>

// 插入代码切片
pub async fn upsert_chunks(&self, chunks: Vec<CodeChunk>) -> Result<()>

// 向量搜索
pub async fn search(&self, query_vector: &[f32], limit: u64) -> Result<Vec<(CodeChunk, f32)>>
```

**数据结构**:
```rust
pub struct QdrantStorage {
    client: Qdrant,              // Qdrant 客户端
    collection_name: String,     // 集合名称
    vector_size: u64,            // 向量维度（768 for CodeBERT）
}
```

**为什么选择 Qdrant?**
- 专为向量搜索设计
- 高性能（支持百万级向量）
- 支持多种距离度量（余弦、欧氏等）
- Rust 原生客户端
- 可以嵌入运行（无需单独服务器）

**向量搜索原理**:
```
查询向量 → Qdrant → 计算余弦相似度 → 返回 Top-K 结果
```

**存储结构**:
- **Point ID**: UUID (基于文件路径和位置生成)
- **Vector**: 768 维浮点向量
- **Payload**: JSON 数据（文件路径、代码内容、行号等）

---

### 9. indexer_state.rs - 索引状态管理

**职责**:
- 跟踪已索引的文件
- 记录文件修改时间
- 支持增量索引（只索引变更文件）

**数据结构**:
```rust
pub struct IndexerState {
    file_states: HashMap<String, FileState>,  // 文件状态映射
}

pub struct FileState {
    pub modified_time: u64,     // 修改时间戳
    pub file_size: u64,         // 文件大小
    pub indexed: bool,          // 是否已索引
}
```

**为什么需要索引状态?**
- 避免重复索引未变更的文件
- 支持增量更新
- 提高索引效率

---

## 依赖库详解

### 异步运行时

#### tokio (1.35)
**作用**: Rust 异步运行时
**为什么需要**: 
- Rust 的异步编程需要运行时支持
- 提供异步 I/O、定时器、任务调度等功能
**使用场景**:
- 文件 I/O 操作
- 网络请求
- 异步任务执行

**关键特性**:
```rust
#[tokio::main]  // 标记异步主函数
async fn main() {
    let content = fs::read_to_string("file.txt").await?;  // 异步读取文件
}
```

---

### 序列化/反序列化

#### serde (1.0) + serde_json (1.0)
**作用**: 序列化框架
**为什么需要**:
- JSON-RPC 协议需要 JSON 序列化
- 配置文件需要序列化
- 数据存储需要序列化

**使用场景**:
```rust
#[derive(Serialize, Deserialize)]  // 自动生成序列化代码
pub struct Config {
    pub path: String,
}
```

#### toml (0.8)
**作用**: TOML 配置文件解析
**为什么需要**:
- 配置文件使用 TOML 格式
- 人类可读，易于编辑

---

### 错误处理

#### anyhow (1.0)
**作用**: 错误处理库
**为什么需要**:
- 提供统一的错误类型
- 支持错误链（error chain）
- 简化错误处理代码

**使用示例**:
```rust
use anyhow::{Result, Context};

fn read_config() -> Result<Config> {
    let content = std::fs::read_to_string("config.toml")
        .context("Failed to read config file")?;  // 添加上下文
    toml::from_str(&content)
        .context("Failed to parse config")?
}
```

---

### 日志

#### tracing (0.1) + tracing-subscriber (0.3)
**作用**: 结构化日志框架
**为什么需要**:
- 提供结构化日志（带字段）
- 支持日志级别过滤
- 性能优秀（零成本抽象）

**使用示例**:
```rust
tracing::info!("Indexing file: {}", file_path);
tracing::debug!("Vector similarity: {}", similarity);
tracing::warn!("Model not found, using TF-IDF");
```

**为什么选择 tracing 而不是 log?**
- 支持异步上下文
- 更好的性能
- 结构化日志
- 与 Tokio 集成良好

**日志输出位置说明**:
- **默认行为**: `tracing_subscriber::fmt()` 默认将日志输出到**标准错误流 (stderr)**，即终端/控制台
- **当前实现**: **不写入日志文件**，所有日志都实时输出到运行该程序的终端
- **日志位置**:
  - **stdio 模式**: 日志显示在启动该进程的终端窗口
  - **HTTP 模式**: 日志显示在启动 HTTP 服务器的终端窗口
  - **文件路径**: 当前代码中**没有配置日志文件**，日志不会写入任何文件

**如何查看日志**:
1. **直接在终端运行**: 
   ```bash
   ./target/release/browser-mcp
   # 日志会直接显示在终端
   ```

2. **重定向到文件**:
   ```bash
   # 将标准输出和错误都重定向到文件
   ./target/release/browser-mcp > browser-mcp.log 2>&1
   
   # 只重定向错误日志（stderr）
   ./target/release/browser-mcp 2> browser-mcp-error.log
   ```

3. **同时显示在终端和文件** (使用 `tee`):
   ```bash
   ./target/release/browser-mcp 2>&1 | tee browser-mcp.log
   ```

**如何配置日志文件输出** (需要修改代码):
如果要让日志直接写入文件，可以修改 `src/main.rs` 中的日志初始化代码：

```rust
use tracing_subscriber::fmt::writer::BoxMakeWriter;
use std::fs::OpenOptions;

// 创建日志文件（追加模式）
let log_file = OpenOptions::new()
    .create(true)
    .append(true)
    .open("browser-mcp.log")?;

tracing_subscriber::fmt()
    .with_writer(BoxMakeWriter::new(log_file))  // 写入文件
    .with_env_filter(
        tracing_subscriber::EnvFilter::try_new(&log_level)
            .unwrap_or_else(|_| tracing_subscriber::EnvFilter::from_default_env())
    )
    .init();
```

这样配置后，日志文件会在**当前工作目录**创建，文件名为 `browser-mcp.log`。

---

### 文件操作

#### ignore (0.4)
**作用**: 文件遍历，支持 .gitignore
**为什么需要**:
- 自动处理 .gitignore 规则
- 高效遍历大型代码库
- 支持隐藏文件过滤

**使用示例**:
```rust
let walker = WalkBuilder::new(&path)
    .git_ignore(true)      // 启用 .gitignore
    .hidden(false)         // 不遍历隐藏文件
    .build();
```

#### walkdir (2.4)
**作用**: 目录遍历（备用）
**为什么需要**:
- 简单的目录遍历场景
- 作为 ignore 的补充

---

### 文本处理

#### regex (1.10)
**作用**: 正则表达式
**为什么需要**:
- 文本搜索需要模式匹配
- 代码搜索需要正则支持

**使用示例**:
```rust
let regex = Regex::new(&query)?;
if regex.is_match(&line) {
    // 匹配成功
}
```

---

### 代码解析

#### tree-sitter (0.24) + tree-sitter-* (各种语言)
**作用**: 代码解析器
**为什么需要**:
- 提取代码的语义结构
- 支持多种编程语言
- 提供准确的 AST

**支持的语言包**:
- `tree-sitter-rust`: Rust
- `tree-sitter-javascript`: JavaScript
- `tree-sitter-typescript`: TypeScript
- `tree-sitter-python`: Python
- `tree-sitter-cpp`: C++
- `tree-sitter-c`: C
- `tree-sitter-java`: Java
- `tree-sitter-go`: Go
- `tree-sitter-json`: JSON
- `tree-sitter-yaml`: YAML
- `tree-sitter-html`: HTML
- `tree-sitter-css`: CSS

---

### 机器学习

#### candle-core (0.8.2, optional)
**作用**: 张量计算核心
**为什么需要**:
- 模型推理需要张量操作
- 支持多种后端（CPU、CUDA、Metal）

#### candle-nn (0.8.2, optional)
**作用**: 神经网络构建块
**为什么需要**:
- 定义和加载神经网络模型
- 提供层、优化器等组件

#### candle-transformers (0.8.2, optional)
**作用**: Transformer 模型实现
**为什么需要**:
- 提供预训练的 BERT 模型实现
- 简化模型加载和使用

#### tokenizers (0.19)
**作用**: 文本分词器
**为什么需要**:
- CodeBERT 需要分词
- 将文本转换为 token IDs

**为什么选择 Candle 而不是 PyTorch?**
- 纯 Rust，无 Python 依赖
- 更好的性能
- 内存安全
- 支持多种后端

---

### 向量数据库

#### qdrant-client (1.16.0)
**作用**: Qdrant 向量数据库客户端
**为什么需要**:
- 存储代码向量
- 执行相似度搜索

**为什么选择 Qdrant?**
- 专为向量搜索设计
- 高性能
- Rust 原生客户端
- 可以嵌入运行

---

### Web 框架

#### axum (0.7)
**作用**: Web 框架
**为什么需要**:
- 提供 HTTP 服务器
- 处理 HTTP 请求和响应

#### tower (0.4) + tower-http (0.5)
**作用**: 中间件框架
**为什么需要**:
- 提供 CORS 支持
- 请求/响应处理中间件

**为什么选择 Axum?**
- 基于 Tokio，性能优秀
- 类型安全
- 活跃的社区

---

### 工具库

#### dirs (5.0)
**作用**: 获取系统目录路径
**为什么需要**:
- 查找配置文件位置
- 跨平台目录路径

#### shellexpand (3.1)
**作用**: Shell 路径展开
**为什么需要**:
- 支持 `~` 展开为用户目录
- 支持环境变量展开

#### uuid (1.19.0)
**作用**: UUID 生成
**为什么需要**:
- 为 Qdrant 中的点生成唯一 ID

#### reqwest (0.12)
**作用**: HTTP 客户端
**为什么需要**:
- 从 Hugging Face 下载模型
- 下载 Qdrant 二进制文件

#### flate2 (1.0) + tar (0.4) + zip (0.6)
**作用**: 压缩文件处理
**为什么需要**:
- 解压下载的模型文件
- 解压 Qdrant 二进制文件

---

## 技术选型原因

### 为什么选择 Rust?

1. **性能**: 接近 C++ 的性能，但更安全
2. **内存安全**: 编译时保证内存安全，避免常见错误
3. **并发**: 优秀的异步编程支持（Tokio）
4. **生态系统**: 丰富的 crate 生态系统
5. **跨平台**: 一次编写，多平台运行

### 为什么选择 Tree-sitter?

1. **增量解析**: 只解析变更部分，性能优秀
2. **多语言支持**: 支持 40+ 种编程语言
3. **准确性**: 提供准确的 AST
4. **Rust 绑定**: 有成熟的 Rust 绑定

### 为什么选择 CodeBERT?

1. **专门设计**: 为代码理解任务设计
2. **性能**: 在代码搜索任务上表现优秀
3. **多语言**: 支持多种编程语言
4. **模型大小**: 相对较小（~500MB）

### 为什么选择 Candle?

1. **纯 Rust**: 无 Python 依赖
2. **性能**: 优秀的性能
3. **多后端**: 支持 CPU、CUDA、Metal
4. **内存安全**: Rust 的内存安全保证

### 为什么选择 Qdrant?

1. **专为向量设计**: 专为向量搜索优化
2. **高性能**: 支持百万级向量
3. **Rust 客户端**: 原生 Rust 客户端
4. **可嵌入**: 可以嵌入运行

### 为什么选择 Axum?

1. **性能**: 基于 Tokio，性能优秀
2. **类型安全**: 编译时类型检查
3. **易用性**: API 设计优雅
4. **社区**: 活跃的社区支持

---

## 数据流

### 索引流程

```
1. 读取配置文件
   ↓
2. 遍历代码库文件（使用 ignore）
   ↓
3. 使用 Tree-sitter 解析代码
   ↓
4. 提取代码切片（函数、类等）
   ↓
5. 使用 CodeBERT 生成向量
   ↓
6. 存储到 Qdrant
   ↓
7. 更新索引状态
```

### 搜索流程

#### 文本搜索
```
1. 接收搜索查询
   ↓
2. 遍历代码库文件
   ↓
3. 使用正则表达式匹配
   ↓
4. 返回匹配结果
```

#### 向量搜索
```
1. 接收自然语言查询
   ↓
2. 使用 CodeBERT 生成查询向量
   ↓
3. 在 Qdrant 中搜索相似向量
   ↓
4. 按相似度排序
   ↓
5. 返回 Top-K 结果
```

---

## 性能优化

### 1. 异步 I/O
- 使用 Tokio 异步运行时
- 非阻塞文件 I/O
- 并发处理多个请求

### 2. 增量索引
- 只索引变更的文件
- 使用索引状态跟踪
- 避免重复处理

### 3. 批量处理
- 批量生成向量
- 批量插入 Qdrant
- 减少网络开销

### 4. 智能过滤
- 忽略二进制文件
- 忽略大文件
- 使用 .gitignore

### 5. 向量搜索优化
- 使用 Qdrant 的 HNSW 索引
- 余弦相似度计算优化
- 限制返回结果数量

---

## 总结

Browser MCP 是一个技术栈现代化的代码搜索工具，主要特点：

1. **高性能**: 使用 Rust + Tokio 异步运行时
2. **智能搜索**: Tree-sitter + CodeBERT + Qdrant
3. **易扩展**: 模块化设计，易于添加新功能
4. **跨平台**: 支持 macOS、Linux、Windows
5. **标准协议**: 实现 MCP 标准，易于集成

通过本文档，你应该能够：
- 理解项目的整体架构
- 了解各个模块的作用
- 理解技术选型的原因
- 掌握关键依赖库的使用

如有疑问，请参考源代码或提交 Issue。
