use super::Client;
use crate::completion::{self, CompletionError, CompletionRequest};
use crate::http_client::{self, HttpClientExt};
use crate::providers::openai;
use crate::telemetry::SpanCombinator;
use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tracing::{Instrument, Level, enabled, info_span};

/// The `venice-uncensored` model. Venice's default text model.
pub const VENICE_UNCENSORED: &str = "venice-uncensored";
/// The `llama-3.3-70b` model. Balanced general-purpose text model.
pub const LLAMA_3_3_70B: &str = "llama-3.3-70b";
/// The `zai-org-glm-4.7` model. Venice flagship model for complex tasks.
pub const ZAI_ORG_GLM_4_7: &str = "zai-org-glm-4.7";
/// The `qwen3-vl-235b-a22b` model. Vision-capable multimodal model.
pub const QWEN3_VL_235B_A22B: &str = "qwen3-vl-235b-a22b";

/// Controls Venice web search behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WebSearchMode {
    /// Let Venice decide whether to use web search.
    Auto,
    /// Disable web search.
    Off,
    /// Force web search.
    On,
}

/// Typed helper for Venice-specific request parameters.
///
/// These values are serialized under the `venice_parameters` request field and can be
/// passed to `.additional_params(...)` on an agent or request builder.
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
pub struct VeniceParameters {
    /// Whether Venice should use web search for the request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_web_search: Option<WebSearchMode>,
    /// Whether Venice should include search-result citations in the response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_web_citations: Option<bool>,
    /// Whether Venice should scrape URLs returned from search.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enable_web_scraping: Option<bool>,
    /// Whether to include Venice's system prompt in the request.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_venice_system_prompt: Option<bool>,
    /// Whether to stream search results inline with response chunks.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub include_search_results_in_stream: Option<bool>,
    /// Whether to also return search results as document-style tool output.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub return_search_results_as_documents: Option<bool>,
    /// Public Venice character slug to use for the response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub character_slug: Option<String>,
    /// Whether to strip model thinking content from the final response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strip_thinking_response: Option<bool>,
    /// Whether to disable Venice thinking output entirely.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub disable_thinking: Option<bool>,
}

impl VeniceParameters {
    /// Set Venice web search mode.
    pub fn enable_web_search(mut self, mode: WebSearchMode) -> Self {
        self.enable_web_search = Some(mode);
        self
    }

    /// Toggle Venice web citations.
    pub fn enable_web_citations(mut self, enabled: bool) -> Self {
        self.enable_web_citations = Some(enabled);
        self
    }

    /// Toggle Venice web scraping.
    pub fn enable_web_scraping(mut self, enabled: bool) -> Self {
        self.enable_web_scraping = Some(enabled);
        self
    }

    /// Toggle inclusion of Venice's own system prompt.
    pub fn include_venice_system_prompt(mut self, enabled: bool) -> Self {
        self.include_venice_system_prompt = Some(enabled);
        self
    }

    /// Toggle streaming of search results.
    pub fn include_search_results_in_stream(mut self, enabled: bool) -> Self {
        self.include_search_results_in_stream = Some(enabled);
        self
    }

    /// Toggle returning search results as documents.
    pub fn return_search_results_as_documents(mut self, enabled: bool) -> Self {
        self.return_search_results_as_documents = Some(enabled);
        self
    }

    /// Set the Venice character slug.
    pub fn character_slug(mut self, slug: impl Into<String>) -> Self {
        self.character_slug = Some(slug.into());
        self
    }

    /// Toggle stripping thinking output from the response.
    pub fn strip_thinking_response(mut self, enabled: bool) -> Self {
        self.strip_thinking_response = Some(enabled);
        self
    }

    /// Toggle Venice thinking mode.
    pub fn disable_thinking(mut self, enabled: bool) -> Self {
        self.disable_thinking = Some(enabled);
        self
    }

    /// Convert these provider-specific parameters into Rig `additional_params`.
    pub fn into_additional_params(self) -> Value {
        serde_json::json!({ "venice_parameters": self })
    }

    /// Merge these provider-specific parameters into existing Rig `additional_params`.
    pub fn merge_into_additional_params(self, additional_params: Option<Value>) -> Option<Value> {
        let venice_parameters = self.into_additional_params();
        Some(match additional_params {
            Some(existing) => crate::json_utils::merge(existing, venice_parameters),
            None => venice_parameters,
        })
    }
}

#[derive(Clone)]
pub struct CompletionModel<T = reqwest::Client> {
    pub(crate) client: Client<T>,
    pub model: String,
    pub strict_tools: bool,
    pub tool_result_array_content: bool,
}

impl<T> CompletionModel<T>
where
    T: Default + std::fmt::Debug + Clone + 'static,
{
    pub fn new(client: Client<T>, model: impl Into<String>) -> Self {
        Self {
            client,
            model: model.into(),
            strict_tools: false,
            tool_result_array_content: false,
        }
    }

    /// Enable strict OpenAI-compatible tool schemas for Venice tool calls.
    pub fn with_strict_tools(mut self) -> Self {
        self.strict_tools = true;
        self
    }

    /// Serialize tool results using OpenAI's array content format.
    pub fn with_tool_result_array_content(mut self) -> Self {
        self.tool_result_array_content = true;
        self
    }
}

impl<T> completion::CompletionModel for CompletionModel<T>
where
    T: HttpClientExt
        + Default
        + std::fmt::Debug
        + Clone
        + WasmCompatSend
        + WasmCompatSync
        + 'static,
{
    type Response = openai::CompletionResponse;
    type StreamingResponse = openai::StreamingCompletionResponse;

    type Client = Client<T>;

    fn make(client: &Self::Client, model: impl Into<String>) -> Self {
        Self::new(client.clone(), model)
    }

    async fn completion(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<completion::CompletionResponse<Self::Response>, CompletionError> {
        let request = openai::CompletionRequest::try_from(openai::OpenAIRequestParams {
            model: self.model.to_owned(),
            request: completion_request,
            strict_tools: self.strict_tools,
            tool_result_array_content: self.tool_result_array_content,
        })?;

        if enabled!(Level::TRACE) {
            tracing::trace!(
                target: "rig::completions",
                "Venice Chat Completions request: {}",
                serde_json::to_string_pretty(&request)?
            );
        }

        let input_messages = request.get_input_messages();
        let body = serde_json::to_vec(&request)?;
        let req = self
            .client
            .post("/chat/completions")?
            .body(body)
            .map_err(|err| CompletionError::HttpError(err.into()))?;

        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::completions",
                "chat",
                gen_ai.operation.name = "chat",
                gen_ai.provider.name = "venice",
                gen_ai.request.model = self.model,
                gen_ai.system_instructions = request.get_system_prompt(),
                gen_ai.response.id = tracing::field::Empty,
                gen_ai.response.model = tracing::field::Empty,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
                gen_ai.usage.cached_tokens = tracing::field::Empty,
                gen_ai.input.messages = serde_json::to_string(&input_messages).ok(),
                gen_ai.output.messages = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };

        async move {
            let response = self.client.send(req).await?;

            if response.status().is_success() {
                let text = http_client::text(response).await?;
                let venice_response: VeniceResponse = serde_json::from_str(&text)?;

                let current_span = tracing::Span::current();
                current_span.record_response_metadata(&venice_response.raw_response);
                current_span.record_token_usage(&venice_response.raw_response.usage);

                if enabled!(Level::TRACE) {
                    tracing::trace!(
                        target: "rig::completions",
                        "Venice Chat Completions response: {}",
                        serde_json::to_string_pretty(&venice_response)?
                    );
                }

                venice_response.raw_response.try_into()
            } else {
                let text = http_client::text(response).await?;
                Err(CompletionError::ProviderError(text))
            }
        }
        .instrument(span)
        .await
    }

    async fn stream(
        &self,
        completion_request: CompletionRequest,
    ) -> Result<
        crate::streaming::StreamingCompletionResponse<Self::StreamingResponse>,
        CompletionError,
    > {
        let request = openai::CompletionRequest::try_from(openai::OpenAIRequestParams {
            model: self.model.to_owned(),
            request: completion_request,
            strict_tools: self.strict_tools,
            tool_result_array_content: self.tool_result_array_content,
        })?;

        if enabled!(Level::TRACE) {
            tracing::trace!(
                target: "rig::completions",
                "Venice Chat Completions streaming request: {}",
                serde_json::to_string_pretty(&request)?
            );
        }

        let input_messages = request.get_input_messages();
        let mut request_as_json = serde_json::to_value(&request)?;
        request_as_json = crate::json_utils::merge(
            request_as_json,
            serde_json::json!({"stream": true, "stream_options": {"include_usage": true}}),
        );

        let req_body = serde_json::to_vec(&request_as_json)?;
        let req = self
            .client
            .post("/chat/completions")?
            .body(req_body)
            .map_err(|err| CompletionError::HttpError(err.into()))?;

        let span = if tracing::Span::current().is_disabled() {
            info_span!(
                target: "rig::completions",
                "chat_streaming",
                gen_ai.operation.name = "chat_streaming",
                gen_ai.provider.name = "venice",
                gen_ai.request.model = self.model,
                gen_ai.response.id = tracing::field::Empty,
                gen_ai.response.model = self.model,
                gen_ai.usage.output_tokens = tracing::field::Empty,
                gen_ai.usage.input_tokens = tracing::field::Empty,
                gen_ai.usage.cached_tokens = tracing::field::Empty,
                gen_ai.input.messages = serde_json::to_string(&input_messages).ok(),
                gen_ai.output.messages = tracing::field::Empty,
            )
        } else {
            tracing::Span::current()
        };

        openai::send_compatible_streaming_request(self.client.clone(), req)
            .instrument(span)
            .await
    }
}

#[derive(Debug, Deserialize)]
struct VeniceResponse {
    #[serde(flatten)]
    raw_response: openai::CompletionResponse,
    #[serde(default)]
    venice_parameters: Option<Value>,
}

impl Serialize for VeniceResponse {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut object =
            match serde_json::to_value(&self.raw_response).map_err(serde::ser::Error::custom)? {
                Value::Object(map) => map,
                _ => Map::new(),
            };

        if let Some(venice_parameters) = &self.venice_parameters {
            object.insert("venice_parameters".to_string(), venice_parameters.clone());
        }

        Value::Object(object).serialize(serializer)
    }
}

trait VeniceRequestExt {
    fn get_input_messages(&self) -> Vec<openai::Message>;
    fn get_system_prompt(&self) -> Option<String>;
}

impl VeniceRequestExt for openai::CompletionRequest {
    fn get_input_messages(&self) -> Vec<openai::Message> {
        <openai::CompletionRequest as crate::telemetry::ProviderRequestExt>::get_input_messages(
            self,
        )
    }

    fn get_system_prompt(&self) -> Option<String> {
        <openai::CompletionRequest as crate::telemetry::ProviderRequestExt>::get_system_prompt(self)
    }
}

#[cfg(test)]
mod tests {
    use super::{VeniceParameters, WebSearchMode};

    #[test]
    fn venice_parameters_wrap_into_additional_params() {
        let params = VeniceParameters::default()
            .enable_web_search(WebSearchMode::Auto)
            .include_venice_system_prompt(true)
            .enable_web_citations(true)
            .into_additional_params();

        assert_eq!(
            params,
            serde_json::json!({
                "venice_parameters": {
                    "enable_web_search": "auto",
                    "include_venice_system_prompt": true,
                    "enable_web_citations": true
                }
            })
        );
    }

    #[test]
    fn venice_parameters_merge_existing_additional_params() {
        let merged = VeniceParameters::default()
            .enable_web_search(WebSearchMode::On)
            .merge_into_additional_params(Some(serde_json::json!({
                "temperature_override": 0.2
            })))
            .expect("merged params should exist");

        assert_eq!(
            merged,
            serde_json::json!({
                "temperature_override": 0.2,
                "venice_parameters": {
                    "enable_web_search": "on"
                }
            })
        );
    }
}
