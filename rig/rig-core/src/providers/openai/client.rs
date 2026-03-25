use crate::{
    client::{
        self, BearerAuth, Capabilities, Capable, DebugExt, Nothing, Provider, ProviderBuilder,
        ProviderClient,
    },
    extractor::ExtractorBuilder,
    http_client::{self, HttpClientExt},
    prelude::CompletionClient,
    wasm_compat::{WasmCompatSend, WasmCompatSync},
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

// ================================================================
// Main OpenAI Client
// ================================================================
const OPENAI_API_BASE_URL: &str = "https://api.openai.com/v1";

// ================================================================
// OpenAI Responses API Extension
// ================================================================
#[derive(Debug, Default, Clone, Copy)]
pub struct OpenAIResponsesExt;

#[derive(Debug, Default, Clone, Copy)]
pub struct OpenAIResponsesExtBuilder;

// ================================================================
// OpenAI Completions API Extension
// ================================================================
#[derive(Debug, Default, Clone, Copy)]
pub struct OpenAICompletionsExt;

#[derive(Debug, Default, Clone, Copy)]
pub struct OpenAICompletionsExtBuilder;

type OpenAIApiKey = BearerAuth;

// Responses API client (default)
pub type Client<H = reqwest::Client> = client::Client<OpenAIResponsesExt, H>;
pub type ClientBuilder<H = reqwest::Client> =
    client::ClientBuilder<OpenAIResponsesExtBuilder, OpenAIApiKey, H>;

// Completions API client
pub type CompletionsClient<H = reqwest::Client> = client::Client<OpenAICompletionsExt, H>;
pub type CompletionsClientBuilder<H = reqwest::Client> =
    client::ClientBuilder<OpenAICompletionsExtBuilder, OpenAIApiKey, H>;

impl Provider for OpenAIResponsesExt {
    type Builder = OpenAIResponsesExtBuilder;
    const VERIFY_PATH: &'static str = "/models";
}

impl Provider for OpenAICompletionsExt {
    type Builder = OpenAICompletionsExtBuilder;
    const VERIFY_PATH: &'static str = "/models";
}

impl<H> Capabilities<H> for OpenAIResponsesExt {
    type Completion = Capable<super::responses_api::ResponsesCompletionModel<H>>;
    type Embeddings = Capable<super::EmbeddingModel<H>>;
    type Transcription = Capable<super::TranscriptionModel<H>>;
    type ModelListing = Nothing;
    #[cfg(feature = "image")]
    type ImageGeneration = Capable<super::ImageGenerationModel<H>>;
    #[cfg(feature = "audio")]
    type AudioGeneration = Capable<super::audio_generation::AudioGenerationModel<H>>;
}

impl<H> Capabilities<H> for OpenAICompletionsExt {
    type Completion = Capable<super::completion::CompletionModel<H>>;
    type Embeddings = Capable<super::EmbeddingModel<H>>;
    type Transcription = Capable<super::TranscriptionModel<H>>;
    type ModelListing = Nothing;
    #[cfg(feature = "image")]
    type ImageGeneration = Capable<super::ImageGenerationModel<H>>;
    #[cfg(feature = "audio")]
    type AudioGeneration = Capable<super::audio_generation::AudioGenerationModel<H>>;
}

impl DebugExt for OpenAIResponsesExt {}

impl DebugExt for OpenAICompletionsExt {}

impl ProviderBuilder for OpenAIResponsesExtBuilder {
    type Extension<H>
        = OpenAIResponsesExt
    where
        H: HttpClientExt;
    type ApiKey = OpenAIApiKey;

    const BASE_URL: &'static str = OPENAI_API_BASE_URL;

    fn build<H>(
        _builder: &client::ClientBuilder<Self, Self::ApiKey, H>,
    ) -> http_client::Result<Self::Extension<H>>
    where
        H: HttpClientExt,
    {
        Ok(OpenAIResponsesExt)
    }
}

impl ProviderBuilder for OpenAICompletionsExtBuilder {
    type Extension<H>
        = OpenAICompletionsExt
    where
        H: HttpClientExt;
    type ApiKey = OpenAIApiKey;

    const BASE_URL: &'static str = OPENAI_API_BASE_URL;

    fn build<H>(
        _builder: &client::ClientBuilder<Self, Self::ApiKey, H>,
    ) -> http_client::Result<Self::Extension<H>>
    where
        H: HttpClientExt,
    {
        Ok(OpenAICompletionsExt)
    }
}

impl<H> Client<H>
where
    H: HttpClientExt
        + Clone
        + std::fmt::Debug
        + Default
        + WasmCompatSend
        + WasmCompatSync
        + 'static,
{
    /// Create an extractor builder with the given completion model.
    /// Uses the OpenAI Responses API (default behavior).
    pub fn extractor<U>(
        &self,
        model: impl Into<String>,
    ) -> ExtractorBuilder<super::responses_api::ResponsesCompletionModel<H>, U>
    where
        U: JsonSchema + for<'a> Deserialize<'a> + Serialize + WasmCompatSend + WasmCompatSync,
    {
        ExtractorBuilder::new(self.completion_model(model))
    }

    /// Create a Completions API client from this Responses API client.
    /// Useful for switching to the traditional Chat Completions API.
    pub fn completions_api(self) -> CompletionsClient<H> {
        self.with_ext(OpenAICompletionsExt)
    }
}

impl Client<reqwest::Client> {
    /// Create a new OpenAI Responses API client from custom environment variable names.
    ///
    /// This is useful for OpenAI-compatible providers that expose their API key and base URL
    /// under provider-specific environment variables.
    ///
    /// # Example
    /// ```no_run
    /// use rig::providers::openai;
    ///
    /// let client = openai::Client::from_env_vars("VENICE_API_KEY", Some("VENICE_API_BASE_URL"))
    ///     .completions_api();
    /// ```
    ///
    /// Pass `None` for `base_url_var` to keep the default OpenAI base URL.
    pub fn from_env_vars(api_key_var: &str, base_url_var: Option<&str>) -> Self {
        build_client_from_env_vars(Self::builder(), api_key_var, base_url_var)
    }
}

#[cfg(not(target_family = "wasm"))]
impl Client<reqwest::Client> {
    /// WebSocket mode currently uses a native `tokio-tungstenite` transport and does
    /// not reuse custom `HttpClientExt` backends, so this API is only exposed for the
    /// default `reqwest::Client` transport.
    pub fn responses_websocket_builder(
        &self,
        model: impl Into<String>,
    ) -> super::responses_api::websocket::ResponsesWebSocketSessionBuilder {
        super::responses_api::websocket::ResponsesWebSocketSessionBuilder::new(
            self.completion_model(model),
        )
    }

    /// This API is OpenAI-specific and only available on non-wasm targets in `rig-core`.
    pub async fn responses_websocket(
        &self,
        model: impl Into<String>,
    ) -> Result<
        super::responses_api::websocket::ResponsesWebSocketSession,
        crate::completion::CompletionError,
    > {
        self.responses_websocket_builder(model).connect().await
    }
}

impl<H> CompletionsClient<H>
where
    H: HttpClientExt
        + Clone
        + std::fmt::Debug
        + Default
        + WasmCompatSend
        + WasmCompatSync
        + 'static,
{
    /// Create an extractor builder with the given completion model.
    /// Uses the OpenAI Chat Completions API.
    pub fn extractor<U>(
        &self,
        model: impl Into<String>,
    ) -> ExtractorBuilder<super::completion::CompletionModel<H>, U>
    where
        U: JsonSchema + for<'a> Deserialize<'a> + Serialize + WasmCompatSend + WasmCompatSync,
    {
        ExtractorBuilder::new(self.completion_model(model))
    }

    /// Create a Responses API client from this Completions API client.
    /// Useful for switching to the newer Responses API.
    pub fn responses_api(self) -> Client<H> {
        self.with_ext(OpenAIResponsesExt)
    }
}

impl CompletionsClient<reqwest::Client> {
    /// Create a new OpenAI Completions API client from custom environment variable names.
    ///
    /// This is useful for OpenAI-compatible providers that expose their API key and base URL
    /// under provider-specific environment variables.
    ///
    /// # Example
    /// ```no_run
    /// use rig::providers::openai;
    ///
    /// let client =
    ///     openai::CompletionsClient::from_env_vars("VENICE_API_KEY", Some("VENICE_API_BASE_URL"));
    /// ```
    ///
    /// Pass `None` for `base_url_var` to keep the default OpenAI base URL.
    pub fn from_env_vars(api_key_var: &str, base_url_var: Option<&str>) -> Self {
        build_client_from_env_vars(Self::builder(), api_key_var, base_url_var)
    }
}

fn build_client_from_env_vars<Ext, B>(
    mut builder: client::ClientBuilder<B, client::NeedsApiKey, reqwest::Client>,
    api_key_var: &str,
    base_url_var: Option<&str>,
) -> client::Client<Ext, reqwest::Client>
where
    Ext: Provider,
    B: ProviderBuilder<ApiKey = OpenAIApiKey, Extension<reqwest::Client> = Ext>,
{
    let api_key = std::env::var(api_key_var).unwrap_or_else(|_| panic!("{api_key_var} not set"));

    if let Some(base_url_var) = base_url_var
        && let Ok(base_url) = std::env::var(base_url_var)
    {
        builder = builder.base_url(&base_url);
    }

    builder.api_key(&api_key).build().unwrap()
}

impl ProviderClient for Client {
    type Input = OpenAIApiKey;

    /// Create a new OpenAI Responses API client from the `OPENAI_API_KEY` environment variable.
    /// Panics if the environment variable is not set.
    fn from_env() -> Self {
        Self::from_env_vars("OPENAI_API_KEY", Some("OPENAI_BASE_URL"))
    }

    fn from_val(input: Self::Input) -> Self {
        Self::new(input).unwrap()
    }
}

impl ProviderClient for CompletionsClient {
    type Input = OpenAIApiKey;

    /// Create a new OpenAI Completions API client from the `OPENAI_API_KEY` environment variable.
    /// Panics if the environment variable is not set.
    fn from_env() -> Self {
        Self::from_env_vars("OPENAI_API_KEY", Some("OPENAI_BASE_URL"))
    }

    fn from_val(input: Self::Input) -> Self {
        Self::new(input).unwrap()
    }
}

#[derive(Debug, Deserialize)]
pub struct ApiErrorResponse {
    pub(crate) message: String,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub(crate) enum ApiResponse<T> {
    Ok(T),
    Err(ApiErrorResponse),
}

#[cfg(test)]
mod tests {
    use super::{Client, CompletionsClient, OPENAI_API_BASE_URL};
    use crate::message::ImageDetail;
    use crate::providers::openai::{
        AssistantContent, Function, ImageUrl, Message, ToolCall, ToolType, UserContent,
    };
    use crate::{OneOrMany, message};
    use serde_path_to_error::deserialize;
    use std::sync::{Mutex, OnceLock};

    fn env_lock() -> &'static Mutex<()> {
        static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        ENV_LOCK.get_or_init(|| Mutex::new(()))
    }

    struct EnvVarGuard {
        names: Vec<&'static str>,
    }

    impl EnvVarGuard {
        fn set(vars: &[(&'static str, &'static str)]) -> Self {
            for (name, value) in vars {
                // Tests mutate process-global environment state, so we serialize them with a mutex.
                unsafe {
                    std::env::set_var(name, value);
                }
            }

            Self {
                names: vars.iter().map(|(name, _)| *name).collect(),
            }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            for name in &self.names {
                unsafe {
                    std::env::remove_var(name);
                }
            }
        }
    }

    #[test]
    fn test_deserialize_message() {
        let assistant_message_json = r#"
        {
            "role": "assistant",
            "content": "\n\nHello there, how may I assist you today?"
        }
        "#;

        let assistant_message_json2 = r#"
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "\n\nHello there, how may I assist you today?"
                }
            ],
            "tool_calls": null
        }
        "#;

        let assistant_message_json3 = r#"
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_h89ipqYUjEpCPI6SxspMnoUU",
                    "type": "function",
                    "function": {
                        "name": "subtract",
                        "arguments": "{\"x\": 2, \"y\": 5}"
                    }
                }
            ],
            "content": null,
            "refusal": null
        }
        "#;

        let user_message_json = r#"
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What's in this image?"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                    }
                },
                {
                    "type": "audio",
                    "input_audio": {
                        "data": "...",
                        "format": "mp3"
                    }
                }
            ]
        }
        "#;

        let assistant_message: Message = {
            let jd = &mut serde_json::Deserializer::from_str(assistant_message_json);
            deserialize(jd).unwrap_or_else(|err| {
                panic!(
                    "Deserialization error at {} ({}:{}): {}",
                    err.path(),
                    err.inner().line(),
                    err.inner().column(),
                    err
                );
            })
        };

        let assistant_message2: Message = {
            let jd = &mut serde_json::Deserializer::from_str(assistant_message_json2);
            deserialize(jd).unwrap_or_else(|err| {
                panic!(
                    "Deserialization error at {} ({}:{}): {}",
                    err.path(),
                    err.inner().line(),
                    err.inner().column(),
                    err
                );
            })
        };

        let assistant_message3: Message = {
            let jd: &mut serde_json::Deserializer<serde_json::de::StrRead<'_>> =
                &mut serde_json::Deserializer::from_str(assistant_message_json3);
            deserialize(jd).unwrap_or_else(|err| {
                panic!(
                    "Deserialization error at {} ({}:{}): {}",
                    err.path(),
                    err.inner().line(),
                    err.inner().column(),
                    err
                );
            })
        };

        let user_message: Message = {
            let jd = &mut serde_json::Deserializer::from_str(user_message_json);
            deserialize(jd).unwrap_or_else(|err| {
                panic!(
                    "Deserialization error at {} ({}:{}): {}",
                    err.path(),
                    err.inner().line(),
                    err.inner().column(),
                    err
                );
            })
        };

        match assistant_message {
            Message::Assistant { content, .. } => {
                assert_eq!(
                    content[0],
                    AssistantContent::Text {
                        text: "\n\nHello there, how may I assist you today?".to_string()
                    }
                );
            }
            _ => panic!("Expected assistant message"),
        }

        match assistant_message2 {
            Message::Assistant {
                content,
                tool_calls,
                ..
            } => {
                assert_eq!(
                    content[0],
                    AssistantContent::Text {
                        text: "\n\nHello there, how may I assist you today?".to_string()
                    }
                );

                assert_eq!(tool_calls, vec![]);
            }
            _ => panic!("Expected assistant message"),
        }

        match assistant_message3 {
            Message::Assistant {
                content,
                tool_calls,
                refusal,
                ..
            } => {
                assert!(content.is_empty());
                assert!(refusal.is_none());
                assert_eq!(
                    tool_calls[0],
                    ToolCall {
                        id: "call_h89ipqYUjEpCPI6SxspMnoUU".to_string(),
                        r#type: ToolType::Function,
                        function: Function {
                            name: "subtract".to_string(),
                            arguments: serde_json::json!({"x": 2, "y": 5}),
                        },
                    }
                );
            }
            _ => panic!("Expected assistant message"),
        }

        match user_message {
            Message::User { content, .. } => {
                let (first, second) = {
                    let mut iter = content.into_iter();
                    (iter.next().unwrap(), iter.next().unwrap())
                };
                assert_eq!(
                    first,
                    UserContent::Text {
                        text: "What's in this image?".to_string()
                    }
                );
                assert_eq!(second, UserContent::Image { image_url: ImageUrl { url: "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg".to_string(), detail: ImageDetail::default() } });
            }
            _ => panic!("Expected user message"),
        }
    }

    #[test]
    fn test_message_to_message_conversion() {
        let user_message = message::Message::User {
            content: OneOrMany::one(message::UserContent::text("Hello")),
        };

        let assistant_message = message::Message::Assistant {
            id: None,
            content: OneOrMany::one(message::AssistantContent::text("Hi there!")),
        };

        let converted_user_message: Vec<Message> = user_message.clone().try_into().unwrap();
        let converted_assistant_message: Vec<Message> =
            assistant_message.clone().try_into().unwrap();

        match converted_user_message[0].clone() {
            Message::User { content, .. } => {
                assert_eq!(
                    content.first(),
                    UserContent::Text {
                        text: "Hello".to_string()
                    }
                );
            }
            _ => panic!("Expected user message"),
        }

        match converted_assistant_message[0].clone() {
            Message::Assistant { content, .. } => {
                assert_eq!(
                    content[0].clone(),
                    AssistantContent::Text {
                        text: "Hi there!".to_string()
                    }
                );
            }
            _ => panic!("Expected assistant message"),
        }

        let original_user_message: message::Message =
            converted_user_message[0].clone().try_into().unwrap();
        let original_assistant_message: message::Message =
            converted_assistant_message[0].clone().try_into().unwrap();

        assert_eq!(original_user_message, user_message);
        assert_eq!(original_assistant_message, assistant_message);
    }

    #[test]
    fn test_message_from_message_conversion() {
        let user_message = Message::User {
            content: OneOrMany::one(UserContent::Text {
                text: "Hello".to_string(),
            }),
            name: None,
        };

        let assistant_message = Message::Assistant {
            content: vec![AssistantContent::Text {
                text: "Hi there!".to_string(),
            }],
            refusal: None,
            audio: None,
            name: None,
            tool_calls: vec![],
        };

        let converted_user_message: message::Message = user_message.clone().try_into().unwrap();
        let converted_assistant_message: message::Message =
            assistant_message.clone().try_into().unwrap();

        match converted_user_message.clone() {
            message::Message::User { content } => {
                assert_eq!(content.first(), message::UserContent::text("Hello"));
            }
            _ => panic!("Expected user message"),
        }

        match converted_assistant_message.clone() {
            message::Message::Assistant { content, .. } => {
                assert_eq!(
                    content.first(),
                    message::AssistantContent::text("Hi there!")
                );
            }
            _ => panic!("Expected assistant message"),
        }

        let original_user_message: Vec<Message> = converted_user_message.try_into().unwrap();
        let original_assistant_message: Vec<Message> =
            converted_assistant_message.try_into().unwrap();

        assert_eq!(original_user_message[0], user_message);
        assert_eq!(original_assistant_message[0], assistant_message);
    }

    #[test]
    fn test_user_message_single_text_serializes_as_string() {
        let user_message = Message::User {
            content: OneOrMany::one(UserContent::Text {
                text: "Hello world".to_string(),
            }),
            name: None,
        };

        let serialized = serde_json::to_value(&user_message).unwrap();

        assert_eq!(serialized["role"], "user");
        assert_eq!(serialized["content"], "Hello world");
    }

    #[test]
    fn test_user_message_multiple_parts_serializes_as_array() {
        let user_message = Message::User {
            content: OneOrMany::many(vec![
                UserContent::Text {
                    text: "What's in this image?".to_string(),
                },
                UserContent::Image {
                    image_url: ImageUrl {
                        url: "https://example.com/image.jpg".to_string(),
                        detail: ImageDetail::default(),
                    },
                },
            ])
            .unwrap(),
            name: None,
        };

        let serialized = serde_json::to_value(&user_message).unwrap();

        assert_eq!(serialized["role"], "user");
        assert!(serialized["content"].is_array());
        assert_eq!(serialized["content"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn test_user_message_single_image_serializes_as_array() {
        let user_message = Message::User {
            content: OneOrMany::one(UserContent::Image {
                image_url: ImageUrl {
                    url: "https://example.com/image.jpg".to_string(),
                    detail: ImageDetail::default(),
                },
            }),
            name: None,
        };

        let serialized = serde_json::to_value(&user_message).unwrap();

        assert_eq!(serialized["role"], "user");
        // Single non-text content should still serialize as array
        assert!(serialized["content"].is_array());
    }
    #[test]
    fn test_client_initialization() {
        let _client =
            crate::providers::openai::Client::new("dummy-key").expect("Client::new() failed");
        let _client_from_builder = crate::providers::openai::Client::builder()
            .api_key("dummy-key")
            .build()
            .expect("Client::builder() failed");
    }

    #[test]
    fn test_from_env_vars_uses_custom_env_names() {
        let _lock = env_lock().lock().unwrap();
        let _guard = EnvVarGuard::set(&[
            ("RIG_TEST_VENICE_API_KEY", "venice-key"),
            ("RIG_TEST_VENICE_BASE_URL", "https://api.venice.ai/api/v1"),
        ]);

        let client =
            Client::from_env_vars("RIG_TEST_VENICE_API_KEY", Some("RIG_TEST_VENICE_BASE_URL"));

        assert_eq!(client.base_url(), "https://api.venice.ai/api/v1");
        assert_eq!(
            client.headers().get(http::header::AUTHORIZATION).unwrap(),
            "Bearer venice-key"
        );
    }

    #[test]
    fn test_from_env_vars_keeps_default_base_url_when_optional_var_is_missing() {
        let _lock = env_lock().lock().unwrap();
        let _guard = EnvVarGuard::set(&[("RIG_TEST_OPENAI_API_KEY_ONLY", "openai-key")]);

        let client = CompletionsClient::from_env_vars(
            "RIG_TEST_OPENAI_API_KEY_ONLY",
            Some("RIG_TEST_MISSING_OPENAI_BASE_URL"),
        );

        assert_eq!(client.base_url(), OPENAI_API_BASE_URL);
        assert_eq!(
            client.headers().get(http::header::AUTHORIZATION).unwrap(),
            "Bearer openai-key"
        );
    }
}
