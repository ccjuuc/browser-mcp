# Browser MCP Server - 快速参考

## 🎯 启动命令

### **本地使用（Cursor）**
```powershell
.\target\release\browser-mcp.exe
```

### **HTTP 远程访问**
```powershell
# 默认端口 3000
.\target\release\browser-mcp.exe --http

# 自定义端口
.\target\release\browser-mcp.exe --http --port 8080
```

## 📋 参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--http` | 启用 HTTP 模式 | stdio 模式 |
| `--port <端口>` | HTTP 端口号 | 3000 |

## 🔧 环境变量

```powershell
$env:RUST_LOG='info'   # debug/info/warn/error
```

## 📡 HTTP API

```
GET  http://localhost:3000/health  # 健康检查
POST http://localhost:3000/        # MCP JSON-RPC
```

## 🎉 完整示例

```powershell
cd H:\browser-mcp
$env:RUST_LOG='info'
.\target\release\browser-mcp.exe --http --port 3000
```

详细文档：
- [命令行参考](COMMAND_LINE.md)
- [HTTP 模式](HTTP_MODE.md)
- [Cursor 配置](CURSOR_SETUP.md)
