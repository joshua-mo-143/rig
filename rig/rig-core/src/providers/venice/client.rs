use crate::client::{
    self, BearerAuth, Capabilities, Capable, DebugExt, Nothing, Provider, ProviderBuilder,
    ProviderClient,
};
use crate::http_client::{self, HttpClientExt};

/// Venice API base URL.
pub const VENICE_API_BASE_URL: &str = "https://api.venice.ai/api/v1";

#[derive(Debug, Default, Clone, Copy)]
pub struct VeniceExt;

#[derive(Debug, Default, Clone, Copy)]
pub struct VeniceBuilder;

type VeniceApiKey = BearerAuth;

pub type Client<H = reqwest::Client> = client::Client<VeniceExt, H>;
pub type ClientBuilder<H = reqwest::Client> = client::ClientBuilder<VeniceBuilder, VeniceApiKey, H>;

impl Provider for VeniceExt {
    type Builder = VeniceBuilder;

    const VERIFY_PATH: &'static str = "/models";
}

impl<H> Capabilities<H> for VeniceExt {
    type Completion = Capable<super::CompletionModel<H>>;
    type Embeddings = Capable<super::EmbeddingModel<H>>;
    type Transcription = Nothing;
    type ModelListing = Nothing;
    #[cfg(feature = "image")]
    type ImageGeneration = Nothing;
    #[cfg(feature = "audio")]
    type AudioGeneration = Nothing;
}

impl DebugExt for VeniceExt {}

impl ProviderBuilder for VeniceBuilder {
    type Extension<H>
        = VeniceExt
    where
        H: HttpClientExt;
    type ApiKey = VeniceApiKey;

    const BASE_URL: &'static str = VENICE_API_BASE_URL;

    fn build<H>(
        _builder: &client::ClientBuilder<Self, Self::ApiKey, H>,
    ) -> http_client::Result<Self::Extension<H>>
    where
        H: HttpClientExt,
    {
        Ok(VeniceExt)
    }
}

impl ProviderClient for Client {
    type Input = String;

    /// Create a new Venice client from the `VENICE_API_KEY` environment variable.
    /// Panics if the environment variable is not set.
    fn from_env() -> Self {
        let api_key = std::env::var("VENICE_API_KEY").expect("VENICE_API_KEY not set");
        Self::new(&api_key).unwrap()
    }

    fn from_val(input: Self::Input) -> Self {
        Self::new(&input).unwrap()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_client_initialization() {
        let _client =
            crate::providers::venice::Client::new("dummy-key").expect("Client::new() failed");
        let _client_from_builder = crate::providers::venice::Client::builder()
            .api_key("dummy-key")
            .build()
            .expect("Client::builder() failed");
    }
}
