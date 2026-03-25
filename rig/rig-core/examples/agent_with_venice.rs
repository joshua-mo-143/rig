use rig::completion::Prompt;
use rig::prelude::*;
use rig::providers::venice;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let client = venice::Client::from_env();

    let agent = client
        .agent(venice::ZAI_ORG_GLM_4_7)
        .preamble("You are a helpful research assistant.")
        .additional_params(
            venice::VeniceParameters::default()
                .enable_web_search(venice::WebSearchMode::Auto)
                .enable_web_citations(true)
                .include_venice_system_prompt(true)
                .into_additional_params(),
        )
        .build();

    let response = agent
        .prompt("Summarize one recent AI development in two short paragraphs.")
        .await?;

    println!("{response}");
    Ok(())
}
