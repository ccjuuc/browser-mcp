#!/bin/bash
# 生成自签名证书用于测试 HTTPS
# 使用兼容 rustls 的证书格式

# 生成私钥
openssl genrsa -out key.pem 2048

# 直接生成自签名证书（使用 x509 命令，兼容性更好）
openssl req -x509 -new -nodes -key key.pem -sha256 -days 365 \
  -out cert.pem \
  -subj "/CN=localhost/O=Browser MCP/C=US" \
  -addext "subjectAltName=DNS:localhost,DNS:*.localhost,IP:127.0.0.1,IP:::1"

echo "✅ 证书生成完成！"
echo "📄 cert.pem - 证书文件"
echo "🔑 key.pem - 私钥文件"
echo ""
echo "现在可以在 browser-mcp.toml 中配置："
echo "[server]"
echo "tls_cert = \"./cert.pem\""
echo "tls_key = \"./key.pem\""
