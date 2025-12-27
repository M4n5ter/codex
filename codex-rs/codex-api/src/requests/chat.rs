use crate::error::ApiError;
use crate::requests::headers::build_conversation_headers;
use crate::requests::headers::insert_header;
use crate::requests::headers::subagent_header;
use codex_protocol::models::ContentItem;
use codex_protocol::models::FunctionCallOutputContentItem;
use codex_protocol::models::ReasoningItemContent;
use codex_protocol::models::ReasoningSource;
use codex_protocol::models::ResponseItem;
use codex_protocol::protocol::SessionSource;
use http::HeaderMap;
use serde_json::Value;
use serde_json::json;
use std::collections::HashMap;

/// Assembled request body plus headers for Chat Completions streaming calls.
pub struct ChatRequest {
    pub body: Value,
    pub headers: HeaderMap,
}

pub struct ChatRequestBuilder<'a> {
    model: &'a str,
    instructions: &'a str,
    input: &'a [ResponseItem],
    tools: &'a [Value],
    enable_reasoning: bool,
    conversation_id: Option<String>,
    session_source: Option<SessionSource>,
}

#[derive(Clone, Default)]
struct ReasoningAttachment {
    text: String,
    details: Option<Value>,
    source: Option<ReasoningSource>,
}

impl<'a> ChatRequestBuilder<'a> {
    pub fn new(
        model: &'a str,
        instructions: &'a str,
        input: &'a [ResponseItem],
        tools: &'a [Value],
        enable_reasoning: bool,
    ) -> Self {
        Self {
            model,
            instructions,
            input,
            tools,
            enable_reasoning,
            conversation_id: None,
            session_source: None,
        }
    }

    pub fn conversation_id(mut self, id: Option<String>) -> Self {
        self.conversation_id = id;
        self
    }

    pub fn session_source(mut self, source: Option<SessionSource>) -> Self {
        self.session_source = source;
        self
    }

    pub fn build(self) -> Result<ChatRequest, ApiError> {
        let mut messages = Vec::<Value>::new();
        messages.push(json!({"role": "system", "content": self.instructions}));

        let input = self.input;
        let mut reasoning_by_anchor_index: HashMap<usize, ReasoningAttachment> = HashMap::new();
        let mut last_emitted_role: Option<&str> = None;
        for item in input {
            match item {
                ResponseItem::Message { role, .. } => last_emitted_role = Some(role.as_str()),
                ResponseItem::FunctionCall { .. } | ResponseItem::LocalShellCall { .. } => {
                    last_emitted_role = Some("assistant")
                }
                ResponseItem::FunctionCallOutput { .. } => last_emitted_role = Some("tool"),
                ResponseItem::Reasoning { .. } | ResponseItem::Other => {}
                ResponseItem::CustomToolCall { .. } => {}
                ResponseItem::CustomToolCallOutput { .. } => {}
                ResponseItem::WebSearchCall { .. } => {}
                ResponseItem::GhostSnapshot { .. } => {}
                ResponseItem::Compaction { .. } => {}
            }
        }

        let mut last_user_index: Option<usize> = None;
        for (idx, item) in input.iter().enumerate() {
            if let ResponseItem::Message { role, .. } = item
                && role == "user"
            {
                last_user_index = Some(idx);
            }
        }

        if !matches!(last_emitted_role, Some("user")) {
            for (idx, item) in input.iter().enumerate() {
                if let Some(u_idx) = last_user_index
                    && idx <= u_idx
                {
                    continue;
                }

                if let ResponseItem::Reasoning {
                    content,
                    reasoning_details,
                    reasoning_source,
                    ..
                } = item
                {
                    let mut text = String::new();
                    if let Some(items) = content {
                        for entry in items {
                            match entry {
                                ReasoningItemContent::ReasoningText { text: segment }
                                | ReasoningItemContent::Text { text: segment } => {
                                    text.push_str(segment)
                                }
                            }
                        }
                    }
                    if text.trim().is_empty() && reasoning_details.is_none() {
                        continue;
                    }

                    let attachment = ReasoningAttachment {
                        text,
                        details: reasoning_details.clone(),
                        source: reasoning_source.clone().or_else(|| {
                            reasoning_details
                                .as_ref()
                                .map(|_| ReasoningSource::ReasoningDetails)
                        }),
                    };
                    let mut attached = false;
                    if idx > 0
                        && let ResponseItem::Message { role, .. } = &input[idx - 1]
                        && role == "assistant"
                    {
                        reasoning_by_anchor_index
                            .entry(idx - 1)
                            .and_modify(|existing| {
                                merge_reasoning_attachment(existing, &attachment);
                            })
                            .or_insert(attachment.clone());
                        attached = true;
                    }

                    if !attached && idx + 1 < input.len() {
                        match &input[idx + 1] {
                            ResponseItem::FunctionCall { .. }
                            | ResponseItem::LocalShellCall { .. } => {
                                reasoning_by_anchor_index
                                    .entry(idx + 1)
                                    .and_modify(|existing| {
                                        merge_reasoning_attachment(existing, &attachment);
                                    })
                                    .or_insert(attachment.clone());
                            }
                            ResponseItem::Message { role, .. } if role == "assistant" => {
                                reasoning_by_anchor_index
                                    .entry(idx + 1)
                                    .and_modify(|existing| {
                                        merge_reasoning_attachment(existing, &attachment);
                                    })
                                    .or_insert(attachment.clone());
                            }
                            _ => {}
                        }
                    }
                }
            }
        }

        let mut last_assistant_text: Option<String> = None;

        for (idx, item) in input.iter().enumerate() {
            match item {
                ResponseItem::Message { role, content, .. } => {
                    let mut text = String::new();
                    let mut items: Vec<Value> = Vec::new();
                    let mut saw_image = false;

                    for c in content {
                        match c {
                            ContentItem::InputText { text: t }
                            | ContentItem::OutputText { text: t } => {
                                text.push_str(t);
                                items.push(json!({"type":"text","text": t}));
                            }
                            ContentItem::InputImage { image_url } => {
                                saw_image = true;
                                items.push(
                                    json!({"type":"image_url","image_url": {"url": image_url}}),
                                );
                            }
                        }
                    }

                    if role == "assistant" {
                        if let Some(prev) = &last_assistant_text
                            && prev == &text
                        {
                            continue;
                        }
                        last_assistant_text = Some(text.clone());
                    }

                    let content_value = if role == "assistant" {
                        json!(text)
                    } else if saw_image {
                        json!(items)
                    } else {
                        json!(text)
                    };

                    let mut msg = json!({"role": role, "content": content_value});
                    if role == "assistant"
                        && let Some(reasoning) = reasoning_by_anchor_index.get(&idx)
                        && let Some(obj) = msg.as_object_mut()
                    {
                        attach_reasoning_fields(obj, reasoning);
                    }
                    messages.push(msg);
                }
                ResponseItem::FunctionCall {
                    name,
                    arguments,
                    call_id,
                    ..
                } => {
                    let mut msg = json!({
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [{
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": arguments,
                            }
                        }]
                    });
                    if let Some(reasoning) = reasoning_by_anchor_index.get(&idx)
                        && let Some(obj) = msg.as_object_mut()
                    {
                        attach_reasoning_fields(obj, reasoning);
                    }
                    messages.push(msg);
                }
                ResponseItem::LocalShellCall {
                    id,
                    call_id: _,
                    status,
                    action,
                } => {
                    let mut msg = json!({
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [{
                            "id": id.clone().unwrap_or_default(),
                            "type": "local_shell_call",
                            "status": status,
                            "action": action,
                        }]
                    });
                    if let Some(reasoning) = reasoning_by_anchor_index.get(&idx)
                        && let Some(obj) = msg.as_object_mut()
                    {
                        attach_reasoning_fields(obj, reasoning);
                    }
                    messages.push(msg);
                }
                ResponseItem::FunctionCallOutput { call_id, output } => {
                    let content_value = if let Some(items) = &output.content_items {
                        let mapped: Vec<Value> = items
                            .iter()
                            .map(|it| match it {
                                FunctionCallOutputContentItem::InputText { text } => {
                                    json!({"type":"text","text": text})
                                }
                                FunctionCallOutputContentItem::InputImage { image_url } => {
                                    json!({"type":"image_url","image_url": {"url": image_url}})
                                }
                            })
                            .collect();
                        json!(mapped)
                    } else {
                        json!(output.content)
                    };

                    messages.push(json!({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": content_value,
                    }));
                }
                ResponseItem::CustomToolCall {
                    id,
                    call_id: _,
                    name,
                    input,
                    status: _,
                } => {
                    messages.push(json!({
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [{
                            "id": id,
                            "type": "custom",
                            "custom": {
                                "name": name,
                                "input": input,
                            }
                        }]
                    }));
                }
                ResponseItem::CustomToolCallOutput { call_id, output } => {
                    messages.push(json!({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": output,
                    }));
                }
                ResponseItem::GhostSnapshot { .. } => {
                    continue;
                }
                ResponseItem::Reasoning { .. }
                | ResponseItem::WebSearchCall { .. }
                | ResponseItem::Other
                | ResponseItem::Compaction { .. } => {
                    continue;
                }
            }
        }

        let payload = json!({
            "model": self.model,
            "messages": messages,
            "stream": true,
            "tools": self.tools,
        });

        let payload = if self.enable_reasoning {
            attach_reasoning_controls(payload)
        } else {
            payload
        };

        let mut headers = build_conversation_headers(self.conversation_id);
        if let Some(subagent) = subagent_header(&self.session_source) {
            insert_header(&mut headers, "x-openai-subagent", &subagent);
        }

        Ok(ChatRequest {
            body: payload,
            headers,
        })
    }
}

fn merge_reasoning_attachment(target: &mut ReasoningAttachment, incoming: &ReasoningAttachment) {
    if !incoming.text.is_empty() {
        target.text.push_str(&incoming.text);
    }
    if incoming.details.is_some() {
        target.details = incoming.details.clone();
    }
    merge_reasoning_source(&mut target.source, incoming.source.clone());
}

fn merge_reasoning_source(target: &mut Option<ReasoningSource>, incoming: Option<ReasoningSource>) {
    let Some(incoming) = incoming else {
        return;
    };
    let replace = match target {
        None => true,
        Some(existing) => reasoning_source_rank(&incoming) > reasoning_source_rank(existing),
    };
    if replace {
        *target = Some(incoming);
    }
}

fn reasoning_source_rank(source: &ReasoningSource) -> u8 {
    match source {
        ReasoningSource::Reasoning => 0,
        ReasoningSource::ReasoningContent => 1,
        ReasoningSource::ReasoningDetails => 2,
    }
}

fn attach_reasoning_fields(
    obj: &mut serde_json::Map<String, Value>,
    reasoning: &ReasoningAttachment,
) {
    if let Some(details) = &reasoning.details {
        obj.insert("reasoning_details".to_string(), details.clone());
        return;
    }

    if reasoning.text.trim().is_empty() {
        return;
    }

    let field = match reasoning.source {
        Some(ReasoningSource::ReasoningContent) => "reasoning_content",
        _ => "reasoning",
    };
    obj.insert(field.to_string(), json!(reasoning.text));
}

fn attach_reasoning_controls(mut payload: Value) -> Value {
    let Some(obj) = payload.as_object_mut() else {
        return payload;
    };

    obj.insert("reasoning".to_string(), json!({ "enabled": true }));
    obj.insert("reasoning_split".to_string(), json!(true));
    obj.insert(
        "thinking".to_string(),
        json!({
            "type": "enabled",
            "clear_thinking": false,
        }),
    );
    obj.insert(
        "chat_template_kwargs".to_string(),
        json!({
            "thinking": true,
        }),
    );

    payload
}

#[cfg(test)]
mod tests {
    use super::*;
    use codex_protocol::protocol::SessionSource;
    use codex_protocol::protocol::SubAgentSource;
    use http::HeaderValue;
    use pretty_assertions::assert_eq;

    #[test]
    fn attaches_conversation_and_subagent_headers() {
        let prompt_input = vec![ResponseItem::Message {
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::InputText {
                text: "hi".to_string(),
            }],
        }];
        let req = ChatRequestBuilder::new("gpt-test", "inst", &prompt_input, &[], false)
            .conversation_id(Some("conv-1".into()))
            .session_source(Some(SessionSource::SubAgent(SubAgentSource::Review)))
            .build()
            .expect("request");

        assert_eq!(
            req.headers.get("conversation_id"),
            Some(&HeaderValue::from_static("conv-1"))
        );
        assert_eq!(
            req.headers.get("session_id"),
            Some(&HeaderValue::from_static("conv-1"))
        );
        assert_eq!(
            req.headers.get("x-openai-subagent"),
            Some(&HeaderValue::from_static("review"))
        );
    }

    #[test]
    fn enables_reasoning_controls() {
        let prompt_input = vec![ResponseItem::Message {
            id: None,
            role: "user".to_string(),
            content: vec![ContentItem::InputText {
                text: "hi".to_string(),
            }],
        }];
        let req = ChatRequestBuilder::new("gpt-test", "inst", &prompt_input, &[], true)
            .build()
            .expect("request");

        assert_eq!(req.body["reasoning"]["enabled"], Value::Bool(true));
        assert_eq!(req.body["reasoning_split"], Value::Bool(true));
        assert_eq!(
            req.body["thinking"]["type"],
            Value::String("enabled".into())
        );
        assert_eq!(req.body["thinking"]["clear_thinking"], Value::Bool(false));
        assert_eq!(
            req.body["chat_template_kwargs"]["thinking"],
            Value::Bool(true)
        );
    }
}
