use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;
use tree_sitter::{Language, Parser, Tree};

/// 支持的编程语言
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LanguageType {
    Rust,
    JavaScript,
    TypeScript,
    Python,
    Cpp,
    C,
    Java,
    Go,
    Json,
    Yaml,
    #[allow(dead_code)]
    Markdown,
    Html,
    Css,
    Unknown,
}

impl LanguageType {
    /// 根据文件扩展名检测语言类型
    pub fn from_extension(ext: &str) -> Self {
        match ext.to_lowercase().as_str() {
            "rs" => Self::Rust,
            "js" | "jsx" | "mjs" | "cjs" => Self::JavaScript,
            "ts" | "tsx" => Self::TypeScript,
            "py" | "pyw" | "pyi" => Self::Python,
            "cpp" | "cc" | "cxx" | "c++" | "hpp" | "hh" | "hxx" | "h++" | "mm" => Self::Cpp,  // .mm 是 Objective-C++
            "c" | "h" | "m" => Self::C,  // .m 是 Objective-C
            "java" => Self::Java,
            "go" => Self::Go,
            "json" => Self::Json,
            "yaml" | "yml" => Self::Yaml,
            // "md" | "markdown" => Self::Markdown,  // 暂时移除
            "html" | "htm" => Self::Html,
            "css" => Self::Css,
            _ => Self::Unknown,
        }
    }

    /// 根据文件路径检测语言类型
    pub fn from_path(path: &Path) -> Self {
        if let Some(ext) = path.extension() {
            Self::from_extension(ext.to_string_lossy().as_ref())
        } else {
            // 尝试从文件名检测
            if let Some(file_name) = path.file_name() {
                let name = file_name.to_string_lossy().to_lowercase();
                if name == "makefile" || name.starts_with("makefile.") {
                    return Self::Unknown; // Makefile 暂不支持
                }
            }
            Self::Unknown
        }
    }

    /// 获取 tree-sitter Language
    pub fn get_language(&self) -> Option<Language> {
        match self {
            // 使用常量的语言
            Self::Rust => Some(tree_sitter_rust::LANGUAGE.into()),
            Self::JavaScript => Some(tree_sitter_javascript::LANGUAGE.into()),
            Self::TypeScript => Some(tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into()),
            Self::Python => Some(tree_sitter_python::LANGUAGE.into()),
            Self::Yaml => Some(tree_sitter_yaml::LANGUAGE.into()),
            // Self::Markdown => Some(tree_sitter_markdown::language()),  // 暂时移除
            Self::Html => Some(tree_sitter_html::LANGUAGE.into()),
            // 使用函数的语言
            Self::Cpp => Some(tree_sitter_cpp::language()),
            Self::C => Some(tree_sitter_c::language()),
            Self::Java => Some(tree_sitter_java::language()),
            Self::Go => Some(tree_sitter_go::language()),
            Self::Json => Some(tree_sitter_json::language()),
            Self::Css => Some(tree_sitter_css::LANGUAGE.into()),
            Self::Markdown => None,  // 暂时不支持（版本冲突）
            Self::Unknown => None,
        }
    }

    /// 检查是否支持该语言
    pub fn is_supported(&self) -> bool {
        self.get_language().is_some()
    }
}

/// 代码切片结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeChunk {
    /// 文件路径
    pub file_path: String,
    /// 语言类型
    pub language: String,
    /// 切片内容
    pub content: String,
    /// 起始行号
    pub start_line: usize,
    /// 结束行号
    pub end_line: usize,
    /// 起始字节位置
    pub start_byte: usize,
    /// 结束字节位置
    pub end_byte: usize,
    /// 节点类型（如 function, class, struct 等）
    pub node_type: String,
    /// 节点名称（如果有）
    pub node_name: Option<String>,
    /// 向量嵌入（可选）
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,
}

/// 代码解析器
pub struct CodeParser {
    parser: Parser,
}

impl CodeParser {
    pub fn new() -> Self {
        let parser = Parser::new();
        Self { parser }
    }

    /// 解析代码文件并生成切片
    pub fn parse_file(
        &mut self,
        file_path: &Path,
        content: &str,
        max_chunk_size: usize,
    ) -> Result<Vec<CodeChunk>> {
        let lang_type = LanguageType::from_path(file_path);
        
        if !lang_type.is_supported() {
            // 对于不支持的语言，使用简单的行切片
            return Ok(Self::chunk_by_lines(file_path, content, max_chunk_size));
        }

        let language = match lang_type.get_language() {
            Some(lang) => lang,
            None => return Ok(Self::chunk_by_lines(file_path, content, max_chunk_size)),
        };
        
        if let Err(_) = self.parser.set_language(&language) {
            // 设置语言失败，回退到行切片
            return Ok(Self::chunk_by_lines(file_path, content, max_chunk_size));
        }

        // 尝试解析，如果失败则回退到行切片
        let tree = match self.parser.parse(content, None) {
            Some(tree) => tree,
            None => {
                // 解析失败（可能是语法错误或文件太大），使用行切片
                return Ok(Self::chunk_by_lines(file_path, content, max_chunk_size));
            }
        };

        Self::extract_chunks(file_path, content, &tree, lang_type, max_chunk_size)
    }

    /// 从语法树中提取代码切片
    fn extract_chunks(
        file_path: &Path,
        content: &str,
        tree: &Tree,
        lang_type: LanguageType,
        max_chunk_size: usize,
    ) -> Result<Vec<CodeChunk>> {
        let mut chunks = Vec::new();
        let root_node = tree.root_node();
        let lines: Vec<&str> = content.lines().collect();

        // 根据语言类型选择要提取的节点类型
        let target_node_types = Self::get_target_node_types(lang_type);

        // 遍历语法树，提取有意义的代码块
        let mut cursor = root_node.walk();
        Self::traverse_tree(
            &root_node,
            &mut cursor,
            content,
            &lines,
            &target_node_types,
            max_chunk_size,
            &mut chunks,
            file_path,
            lang_type,
        );

        Ok(chunks)
    }

    /// 获取目标节点类型列表
    fn get_target_node_types(lang_type: LanguageType) -> Vec<&'static str> {
        match lang_type {
            LanguageType::Rust => vec![
                "function_item", "struct_item", "enum_item", "trait_item",
                "impl_item", "mod_item", "const_item", "static_item",
            ],
            LanguageType::JavaScript | LanguageType::TypeScript => vec![
                "function_declaration", "arrow_function", "class_declaration",
                "method_definition", "interface_declaration", "type_alias_declaration",
            ],
            LanguageType::Python => vec![
                "function_definition", "class_definition", "decorated_definition",
            ],
            LanguageType::Cpp | LanguageType::C => vec![
                "function_definition", "class_specifier", "struct_specifier",
                "enum_specifier", "namespace_definition",
            ],
            LanguageType::Java => vec![
                "class_declaration", "method_declaration", "interface_declaration",
                "enum_declaration",
            ],
            LanguageType::Go => vec![
                "function_declaration", "type_declaration", "method_declaration",
            ],
            _ => vec!["function", "class", "struct"], // 通用节点类型
        }
    }

    /// 遍历语法树并提取代码块（迭代版本，避免栈溢出）
    fn traverse_tree(
        node: &tree_sitter::Node,
        _cursor: &mut tree_sitter::TreeCursor,
        content: &str,
        _lines: &[&str],
        target_types: &[&str],
        max_chunk_size: usize,
        chunks: &mut Vec<CodeChunk>,
        file_path: &Path,
        lang_type: LanguageType,
    ) {
        // 使用迭代而不是递归，避免栈溢出
        let mut stack = vec![*node];
        
        while let Some(current_node) = stack.pop() {
            let node_type = current_node.kind();
            
            if target_types.contains(&node_type) {
                let start_byte = current_node.start_byte();
                let mut end_byte = current_node.end_byte();
                let mut chunk_content = &content[start_byte..end_byte];

                // 对于结构体/类定义，如果内容太短（可能只包含声明），尝试扩展范围
                // 检查是否包含完整定义（有大括号），如果没有，尝试包含后续内容
                if (node_type == "struct_specifier" || node_type == "class_specifier" || 
                    node_type == "enum_specifier" || node_type == "union_specifier") &&
                   chunk_content.len() < 100 && 
                   !chunk_content.contains('{') {
                    // 尝试查找匹配的大括号，扩展内容范围
                    let mut brace_count = 0;
                    let mut found_start = false;
                    let mut extended_end = end_byte;
                    
                    for (i, ch) in content[end_byte..].char_indices() {
                        if ch == '{' {
                            brace_count += 1;
                            found_start = true;
                        } else if ch == '}' {
                            if found_start {
                                brace_count -= 1;
                                if brace_count == 0 {
                                    extended_end = end_byte + i + 1;
                                    break;
                                }
                            }
                        }
                    }
                    
                    // 如果找到了匹配的大括号，使用扩展后的内容
                    if extended_end > end_byte {
                        end_byte = extended_end;
                        chunk_content = &content[start_byte..end_byte];
                    }
                }

                // 如果代码块太大，尝试进一步分割
                if chunk_content.len() > max_chunk_size {
                    // 对于大块，将子节点推入栈中继续处理
                    let mut child_cursor = current_node.walk();
                    for child in current_node.children(&mut child_cursor) {
                        stack.push(child);
                    }
                } else {
                    // 提取节点名称（如果有）
                    let node_name = Self::extract_node_name(&current_node, content);

                    let start_point = current_node.start_position();
                    // 重新计算结束位置（可能已扩展）
                    let end_point = if end_byte > current_node.end_byte() {
                        // 需要重新计算行号
                        let extended_content = &content[..end_byte];
                        let line_count = extended_content.lines().count();
                        tree_sitter::Point { row: line_count.saturating_sub(1), column: 0 }
                    } else {
                        current_node.end_position()
                    };
                    let start_line = start_point.row + 1; // 转换为 1-based
                    let end_line = end_point.row + 1;

                    let relative_path = file_path
                        .to_string_lossy()
                        .to_string();

                    chunks.push(CodeChunk {
                        file_path: relative_path,
                        language: format!("{:?}", lang_type),
                        content: chunk_content.to_string(),
                        start_line,
                        end_line,
                        start_byte,
                        end_byte,
                        node_type: node_type.to_string(),
                        node_name,
                        embedding: None, // 将在后续步骤中添加
                    });
                }
            } else {
                // 将子节点推入栈中继续遍历
                let mut child_cursor = current_node.walk();
                for child in current_node.children(&mut child_cursor) {
                    stack.push(child);
                }
            }
        }
    }

    /// 提取节点名称
    fn extract_node_name(node: &tree_sitter::Node, content: &str) -> Option<String> {
        // 查找 identifier 子节点
        let mut cursor = node.walk();
        for child in node.children(&mut cursor) {
            if child.kind() == "identifier" || child.kind() == "type_identifier" {
                let name = &content[child.start_byte()..child.end_byte()];
                return Some(name.to_string());
            }
        }
        None
    }

    /// 对于不支持的语言，使用简单的行切片
    fn chunk_by_lines(file_path: &Path, content: &str, max_chunk_size: usize) -> Vec<CodeChunk> {
        let lines: Vec<&str> = content.lines().collect();
        let mut chunks = Vec::new();
        let mut current_chunk = String::new();
        let mut start_line = 1;
        let mut start_byte = 0;
        let mut current_byte = 0;

        for (line_num, line) in lines.iter().enumerate() {
            let line_with_newline = format!("{}\n", line);
            let line_bytes = line_with_newline.as_bytes().len();

            if current_chunk.len() + line_bytes > max_chunk_size && !current_chunk.is_empty() {
                // 保存当前块
                let end_line = line_num;
                let end_byte = current_byte;
                
                chunks.push(CodeChunk {
                    file_path: file_path.to_string_lossy().to_string(),
                    language: "Unknown".to_string(),
                    content: current_chunk.clone(),
                    start_line,
                    end_line,
                    start_byte,
                    end_byte,
                    node_type: "line_chunk".to_string(),
                    node_name: None,
                    embedding: None,
                });

                // 开始新块
                current_chunk = line_with_newline;
                start_line = line_num + 1;
                start_byte = current_byte;
            } else {
                current_chunk.push_str(&line_with_newline);
            }

            current_byte += line_bytes;
        }

        // 添加最后一个块
        if !current_chunk.is_empty() {
            chunks.push(CodeChunk {
                file_path: file_path.to_string_lossy().to_string(),
                language: "Unknown".to_string(),
                content: current_chunk,
                start_line,
                end_line: lines.len(),
                start_byte,
                end_byte: current_byte,
                node_type: "line_chunk".to_string(),
                node_name: None,
                embedding: None,
            });
        }

        chunks
    }
}

impl Default for CodeParser {
    fn default() -> Self {
        Self::new()
    }
}

