use anyhow::Result;
use llm::{
    completion::{Prompt, ToolDefinition},
    providers,
    tool::Tool,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::env;

#[derive(Deserialize)]
struct OperationArgs {
    x: i32,
    y: i32,
}

#[derive(Deserialize, Serialize)]
struct Adder;
impl Tool for Adder {
    const NAME: &'static str = "add";

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        serde_json::from_value(json!({
            "name": "add",
            "description": "Add x and y together",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {
                        "type": "number",
                        "description": "The first number to add"
                    },
                    "y": {
                        "type": "number",
                        "description": "The second number to add"
                    }
                }
            }
        }))
        .expect("Tool Definition")
    }

    async fn call(&self, args: String) -> Result<String> {
        let args: OperationArgs = serde_json::from_str(&args)?;
        let result = args.x + args.y;
        Ok(format!("{result}"))
    }
}

#[derive(Deserialize, Serialize)]
struct Subtract;
impl Tool for Subtract {
    const NAME: &'static str = "subtract";

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        serde_json::from_value(json!({
            "name": "subtract",
            "description": "Subtract y from x (i.e.: x - y)",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {
                        "type": "number",
                        "description": "The number to substract from"
                    },
                    "y": {
                        "type": "number",
                        "description": "The number to substract"
                    }
                }
            }
        }))
        .expect("Tool Definition")
    }

    async fn call(&self, args: String) -> Result<String> {
        let args: OperationArgs = serde_json::from_str(&args)?;
        let result = args.x - args.y;
        Ok(format!("{result}"))
    }
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create OpenAI client
    let openai_api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
    let openai_client = providers::openai::Client::new(&openai_api_key);

    // Create agent with a single context prompt and two tools
    let gpt4_calculator_agent = openai_client
        .agent("gpt-4")
        .context("You are a calculator here to help the user perform arithmetic operations.")
        .tool(Adder)
        .tool(Subtract)
        .build();

    // Create OpenAI client
    let cohere_api_key = env::var("COHERE_API_KEY").expect("COHERE_API_KEY not set");
    let cohere_client = providers::cohere::Client::new(&cohere_api_key);

    // Create agent with a single context prompt and two tools
    let coral_calculator_agent = cohere_client
        .agent("command-r")
        .preamble("You are a calculator here to help the user perform arithmetic operations.")
        .tool(Adder)
        .tool(Subtract)
        .build();

    // Prompt the agent and print the response
    println!("Calculate 2 - 5");
    println!(
        "GPT-4: {}",
        gpt4_calculator_agent
            .prompt("Calculate 2 - 5", vec![])
            .await?
    );
    println!(
        "Coral: {}",
        coral_calculator_agent
            .prompt("Calculate 2 - 5", vec![])
            .await?
    );

    Ok(())
}
