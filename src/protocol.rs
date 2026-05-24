use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Command {
    Start { src: String, tgt: String },
    Stop,
    Quit,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Event {
    #[serde(rename_all = "camelCase")]
    Partial {
        text: String,
        lang: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        confidence: Option<f32>,
    },
    #[serde(rename_all = "camelCase")]
    Final {
        text: String,
        translated: String,
        lang: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        confidence: Option<f32>,
    },
    Status {
        state: String,
    },
    Error {
        message: String,
    },
}
