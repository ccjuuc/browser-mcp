use crate::parser::CodeChunk;
use anyhow::{Context, Result};
use qdrant_client::Qdrant;
use qdrant_client::qdrant::{CreateCollection, VectorsConfig, VectorParams, Distance, PointStruct, SearchPoints, UpsertPoints};
use qdrant_client::Payload;
use serde_json::json;
use uuid::Uuid;

pub struct QdrantStorage {
    client: Qdrant,
    collection_name: String,
    vector_size: u64,
}

impl QdrantStorage {
    pub fn new(url: &str, collection_name: String, vector_size: u64) -> Result<Self> {
        let client = Qdrant::from_url(url)
            .skip_compatibility_check()
            .build()?;
        
        Ok(Self {
            client,
            collection_name,
            vector_size,
        })
    }

    pub async fn init(&self) -> Result<()> {
        if !self.client.collection_exists(&self.collection_name).await? {
            tracing::info!("Creating Qdrant collection: {}", self.collection_name);
            self.client
                .create_collection(CreateCollection {
                    collection_name: self.collection_name.clone(),
                    vectors_config: Some(VectorsConfig {
                        config: Some(qdrant_client::qdrant::vectors_config::Config::Params(
                            VectorParams {
                                size: self.vector_size,
                                distance: Distance::Cosine.into(),
                                ..Default::default()
                            },
                        )),
                    }),
                    ..Default::default()
                })
                .await
                .context("Failed to create collection")?;
        }
        Ok(())
    }

    pub async fn upsert_chunks(&self, chunks: Vec<CodeChunk>) -> Result<()> {
        if chunks.is_empty() {
            return Ok(());
        }

        let points: Vec<PointStruct> = chunks
            .into_iter()
            .filter_map(|chunk| {
                if let Some(embedding) = chunk.embedding {
                    let payload = json!({
                        "file_path": chunk.file_path,
                        "language": chunk.language,
                        "content": chunk.content,
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                        "start_byte": chunk.start_byte,
                        "end_byte": chunk.end_byte,
                        "node_type": chunk.node_type,
                        "node_name": chunk.node_name,
                    });

                    let unique_key = format!("{}:{}", chunk.file_path, chunk.start_byte);
                    let id = Uuid::new_v5(&Uuid::NAMESPACE_URL, unique_key.as_bytes());

                    let payload_conv = Payload::try_from(payload).ok()?;

                    Some(PointStruct::new(
                        id.to_string(),
                        embedding,
                        payload_conv,
                    ))
                } else {
                    None
                }
            })
            .collect();

        if points.is_empty() {
            return Ok(());
        }

        self.client
            .upsert_points(UpsertPoints {
                collection_name: self.collection_name.clone(),
                points,
                ..Default::default()
            })
            .await
            .context("Failed to upsert points")?;

        Ok(())
    }

    pub async fn search(
        &self,
        vector: Vec<f32>,
        limit: u64,
    ) -> Result<Vec<(CodeChunk, f32)>> {
        let search_result = self
            .client
            .search_points(SearchPoints {
                collection_name: self.collection_name.clone(),
                vector,
                limit,
                with_payload: Some(true.into()),
                ..Default::default()
            })
            .await
            .context("Failed to search points")?;

        let mut results = Vec::new();
        for point in search_result.result {
            let payload = point.payload;
            let score = point.score;

            let get_str = |k: &str| -> String {
                payload.get(k)
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .unwrap_or_default()
            };
            
            let get_u64 = |k: &str| -> usize {
                payload.get(k)
                    .and_then(|v| v.as_integer())
                    .unwrap_or(0) as usize
            };

            let chunk = CodeChunk {
                file_path: get_str("file_path"),
                language: get_str("language"),
                content: get_str("content"),
                start_line: get_u64("start_line"),
                end_line: get_u64("end_line"),
                start_byte: get_u64("start_byte"),
                end_byte: get_u64("end_byte"),
                node_type: get_str("node_type"),
                node_name: payload.get("node_name").and_then(|v| v.as_str()).map(|s| s.to_string()),
                embedding: None,
            };

            results.push((chunk, score));
        }

        Ok(results)
    }
}
