use super::Client;
use crate::embeddings::EmbeddingError;
use crate::http_client::{self, HttpClientExt};
use crate::wasm_compat::WasmCompatSend;
use crate::{embeddings, json_utils};
use serde::{Deserialize, Serialize};
use serde_json::json;

/// Venice embedding model exposed in the quickstart docs.
pub const TEXT_EMBEDDING_BGE_M3: &str = "text-embedding-bge-m3";

#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    pub data: Vec<EmbeddingData>,
    pub usage: Option<Usage>,
}

#[derive(Debug, Deserialize, Clone, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Deserialize)]
struct EmbeddingData {
    pub embedding: Vec<serde_json::Number>,
}

#[derive(Clone)]
pub struct EmbeddingModel<T = reqwest::Client> {
    client: Client<T>,
    pub model: String,
    ndims: usize,
}

fn model_dimensions_from_identifier(identifier: &str) -> Option<usize> {
    match identifier {
        TEXT_EMBEDDING_BGE_M3 => Some(1024),
        _ => None,
    }
}

impl<T> embeddings::EmbeddingModel for EmbeddingModel<T>
where
    T: HttpClientExt + Clone + std::fmt::Debug + Default + WasmCompatSend + 'static,
{
    const MAX_DOCUMENTS: usize = 1024;

    type Client = Client<T>;

    fn make(client: &Self::Client, model: impl Into<String>, ndims: Option<usize>) -> Self {
        let model = model.into();
        let ndims = ndims
            .or(model_dimensions_from_identifier(&model))
            .unwrap_or_default();

        Self::new(client.clone(), model, ndims)
    }

    fn ndims(&self) -> usize {
        self.ndims
    }

    async fn embed_texts(
        &self,
        documents: impl IntoIterator<Item = String>,
    ) -> Result<Vec<embeddings::Embedding>, EmbeddingError> {
        let documents = documents.into_iter().collect::<Vec<_>>();

        let mut body = json!({
            "model": self.model,
            "input": documents,
            "encoding_format": "float",
        });

        if self.ndims > 0 {
            body = json_utils::merge(body, json!({ "dimensions": self.ndims }));
        }

        let body = serde_json::to_vec(&body)?;

        let req = self
            .client
            .post("/embeddings")?
            .body(body)
            .map_err(|err| EmbeddingError::HttpError(err.into()))?;

        let response = self.client.send(req).await?;

        if response.status().is_success() {
            let body: Vec<u8> = response.into_body().await?;
            let response: EmbeddingResponse = serde_json::from_slice(&body)?;

            tracing::info!(
                target: "rig",
                "Venice embedding token usage: {:?}",
                response.usage
            );

            if response.data.len() != documents.len() {
                return Err(EmbeddingError::ResponseError(
                    "Response data length does not match input length".into(),
                ));
            }

            Ok(response
                .data
                .into_iter()
                .zip(documents.into_iter())
                .map(|(embedding, document)| embeddings::Embedding {
                    document,
                    vec: embedding
                        .embedding
                        .into_iter()
                        .filter_map(|n| n.as_f64())
                        .collect(),
                })
                .collect())
        } else {
            let text = http_client::text(response).await?;
            Err(EmbeddingError::ProviderError(text))
        }
    }
}

impl<T> EmbeddingModel<T> {
    pub fn new(client: Client<T>, model: impl Into<String>, ndims: usize) -> Self {
        Self {
            client,
            model: model.into(),
            ndims,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{TEXT_EMBEDDING_BGE_M3, model_dimensions_from_identifier};

    #[test]
    fn known_embedding_model_dimensions_are_reported() {
        assert_eq!(
            model_dimensions_from_identifier(TEXT_EMBEDDING_BGE_M3),
            Some(1024)
        );
    }
}
