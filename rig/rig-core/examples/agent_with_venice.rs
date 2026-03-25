use rig::prelude::*;
use rig::{completion::Prompt, providers::openai};
use serde_json::json;

const VENICE_API_BASE_URL: &str = "https://api.venice.ai/api/v1";
const VENICE_UNCENSORED: &str = "venice-uncensored";

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let api_key = std::env::var("VENICE_API_KEY")
        .or_else(|_| std::env::var("OPENAI_API_KEY"))
        .expect("Set VENICE_API_KEY (or OPENAI_API_KEY) before running this example");

    // Venice exposes an OpenAI-compatible Chat Completions API, so we point Rig's
    // OpenAI client at Venice's base URL and switch the model to Completions mode.
    let client = openai::Client::builder()
        .api_key(&api_key)
        .base_url(VENICE_API_BASE_URL)
        .build()?;

    let agent = client
        .completion_model(VENICE_UNCENSORED)
        .completions_api()
        .into_agent_builder()
        .preamble("You are a privacy-focused AI assistant.")
        .temperature(0.7)
        .additional_params(json!({
            "venice_parameters": {
                "enable_web_search": "auto",
                "include_venice_system_prompt": true
            }
        }))
        .build();

    let response = agent
        .prompt("Give me three concise ideas for a privacy-first AI product.")
        .await?;

    println!("{response}");

    Ok(())
}
