use rig::completion::Prompt;
use rig::prelude::*;
use rig::providers::openai;
use serde_json::json;
use std::env;

const VENICE_API_BASE_URL: &str = "https://api.venice.ai/api/v1";
const VENICE_UNCENSORED: &str = "venice-uncensored";

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let api_key = env::var("VENICE_API_KEY").expect("VENICE_API_KEY not set");

    // Venice exposes an OpenAI-compatible Chat Completions API, so Rig's OpenAI
    // client can target it by overriding the base URL.
    let agent = openai::Client::builder()
        .api_key(api_key)
        .base_url(VENICE_API_BASE_URL)
        .build()?
        .completions_api()
        .agent(VENICE_UNCENSORED)
        .preamble("You are a privacy-first research assistant.")
        .temperature(0.7)
        .additional_params(json!({
            "venice_parameters": {
                "enable_web_search": "auto",
                "include_venice_system_prompt": true
            }
        }))
        .build();

    let response = agent
        .prompt("Explain why teams might choose an OpenAI-compatible API in two sentences.")
        .await?;

    println!("{response}");

    Ok(())
}
