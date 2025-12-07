# Browser MCP Server

一个用于访问和搜索 Brave Browser 代码库的 MCP (Model Context Protocol) 服务器。

## 功能特性

- 🔍 **代码搜索**: 在代码库中搜索文本和模式
- 📁 **文件浏览**: 列出和读取代码库中的文件
- 🌳 **代码切片**: 使用 tree-sitter 进行语义感知的代码切片
- 🔢 **向量化**: 自动为代码切片生成向量嵌入，支持语义搜索
  - 默认尝试使用 CodeBERT + ONNX Runtime（如果模型可用）
  - 自动降级到 TF-IDF 向量化（快速、无需依赖、资源消耗低）
  - 模型加载框架已就绪，推理部分需要架构优化
  - 详细对比见 [EMBEDDING_COMPARISON.md](EMBEDDING_COMPARISON.md)
  - CodeBERT vs ONNX Runtime 区别见 [ORT_VS_CODEBERT.md](ORT_VS_CODEBERT.md)
- 🌐 **多语言支持**: 支持 Rust, JavaScript, TypeScript, Python, C/C++/Objective-C++, Java, Go, JSON, YAML, HTML, CSS 等
- 🚀 **高性能**: 使用异步 I/O 和智能文件过滤
- 🛡️ **安全**: 自动忽略二进制文件和大文件

## 安装

```bash
# 克隆或下载代码
cd browser-mcp

# 构建项目
cargo build --release
```

## 配置

服务器使用 TOML 配置文件进行配置。配置文件按以下优先级查找：

1. 环境变量 `BROWSER_MCP_CONFIG` 指定的路径
2. 当前目录的 `browser-mcp.toml`
3. 用户配置目录的 `browser-mcp/config.toml` (macOS: `~/Library/Application Support/browser-mcp/config.toml`)
4. 如果都没有找到，使用默认配置

### 创建配置文件

复制示例配置文件并修改：

```bash
cp browser-mcp.toml.example browser-mcp.toml
```

然后编辑 `browser-mcp.toml`，设置你的代码库路径：

```toml
[codebase]
# Brave Browser 代码库的路径（绝对路径或相对路径）
path = "/path/to/brave-browser"

# 搜索时忽略的文件扩展名
ignored_extensions = [
    "png", "jpg", "jpeg", "gif", "ico", "svg",
    "woff", "woff2", "ttf", "otf", "eot",
    "pdf", "zip", "tar", "gz", "jar",
    "class", "so", "dylib", "dll", "exe"
]

# 搜索时忽略大于此大小的文件（字节），默认 1MB
max_file_size = 1_000_000

# 代码切片最大大小（字节），默认 2000
chunk_size = 2000

# 是否启用代码切片（使用 tree-sitter），默认 true
enable_chunking = true

# 是否启用向量化，默认 true
# 使用本地 TF-IDF 向量化方法
enable_embedding = true

[server]
# 日志级别: trace, debug, info, warn, error
log_level = "info"

# 搜索默认最大结果数
max_results = 50
```

## 使用方法

### 作为 MCP 服务器运行

```bash
# 开发模式
cargo run

# 发布模式
cargo build --release
./target/release/browser-mcp
```

服务器通过 stdin/stdout 与 MCP 客户端通信。

### 与 Claude Desktop 集成

在 Claude Desktop 的配置文件中添加服务器配置（通常位于 `~/Library/Application Support/Claude/claude_desktop_config.json` 或 Windows 的 `%APPDATA%\Claude\claude_desktop_config.json`）：

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

**注意**: 确保在运行 `browser-mcp` 的目录中有 `browser-mcp.toml` 配置文件，或者将配置文件放在用户配置目录中。

重启 Claude Desktop 后，你就可以在对话中使用代码库搜索功能了。

### 工具

服务器提供以下工具：

1. **search_code**: 在代码库中搜索文本
   - `query`: 搜索查询字符串
   - `max_results`: 最大结果数量（默认 50）

2. **list_files**: 列出指定目录中的文件
   - `path`: 目录路径（相对于代码库根目录）

3. **read_file**: 读取文件内容
   - `file_path`: 文件路径（相对于代码库根目录）

4. **chunk_file**: 对文件进行代码切片（使用 tree-sitter）
   - `file_path`: 文件路径（相对于代码库根目录）
   - 返回代码切片列表，每个切片包含函数、类等语义单元

5. **chunk_files**: 批量对多个文件进行代码切片
   - `file_paths`: 文件路径数组

6. **search_by_embedding**: 基于向量相似度的语义搜索
   - `query`: 搜索查询文本
   - `max_results`: 最大结果数量（默认 10）
   - 返回代码切片及其相似度分数

## 资源

- `codebase://search`: 代码搜索资源
- `codebase://files`: 文件列表资源

## 开发

```bash
# 运行测试
cargo test

# 运行并查看日志（可以通过环境变量或配置文件设置）
RUST_LOG=debug cargo run

# 或者修改配置文件中的 log_level 字段
```

## 代码向量化模型

项目支持集成专用的代码向量化模型（如 CodeBERT）。详见 [MODEL_INTEGRATION.md](MODEL_INTEGRATION.md)。

### 快速开始

1. **转换模型**（需要 Python 环境）：
   ```bash
   python scripts/convert_codebert_to_onnx.py \
       --model microsoft/codebert-base \
       --output ./model
   ```

2. **配置模型路径**：
   ```toml
   [embedding]
   model_path = "./model/model.onnx"
   tokenizer_path = "./model/tokenizer.json"
   dimension = 768  # CodeBERT 输出维度
   ```

3. **构建并运行**：
   ```bash
   cargo build --release --features model-embedding
   ./target/release/browser-mcp
   ```

## 许可证

MIT

