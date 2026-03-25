//! Venice API client and Rig integration
//!
//! # Example
//! ```no_run
//! use rig::prelude::*;
//! use rig::providers::venice;
//!
//! let client = venice::Client::from_env();
//!
//! let agent = client
//!     .agent(venice::ZAI_ORG_GLM_4_7)
//!     .additional_params(
//!         venice::VeniceParameters::default()
//!             .enable_web_search(venice::WebSearchMode::Auto)
//!             .include_venice_system_prompt(true)
//!             .into_additional_params(),
//!     )
//!     .build();
//! ```

pub mod client;
pub mod completion;
pub mod embedding;

pub use client::*;
pub use completion::*;
pub use embedding::*;
