//! HTTP client for the Anthropic Messages API (`POST /v1/messages`).
//!
//! Mirrors [`crate::endpoint::chat_completions::ChatCompletionsClient`] but
//! targets the Messages streaming endpoint and feeds the response into
//! [`crate::sse::spawn_messages_stream`]. The endpoint URL is caller-supplied
//! (typically from [`codex_model_provider_info::ModelProviderInfo::messages_endpoint_url`])
//! because Anthropic-compatible providers may expose the Messages wire
//! protocol at custom locations.

use crate::anthropic::AnthropicMessagesRequest;
use crate::auth::SharedAuthProvider;
use crate::common::ResponseStream;
use crate::endpoint::session::EndpointSession;
use crate::error::ApiError;
use crate::provider::Provider;
use crate::sse::spawn_messages_stream;
use crate::telemetry::SseTelemetry;
use codex_client::EncodedJsonBody;
use codex_client::HttpTransport;
use codex_client::RequestTelemetry;
use http::HeaderMap;
use http::HeaderValue;
use http::Method;
use std::sync::Arc;
use std::sync::OnceLock;
use tracing::instrument;

/// HTTP client for the Anthropic Messages streaming endpoint.
///
/// Generic over [`HttpTransport`] so it can be used with both real HTTP
/// clients and test doubles, matching
/// [`ResponsesClient`](crate::endpoint::ResponsesClient).
pub struct MessagesClient<T: HttpTransport> {
    session: EndpointSession<T>,
    sse_telemetry: Option<Arc<dyn SseTelemetry>>,
    /// Full endpoint URL (e.g. `https://api.anthropic.com/v1/messages`).
    endpoint_url: String,
}

/// Per-request options for the Messages API. Lighter than
/// [`ResponsesOptions`](crate::endpoint::ResponsesOptions) because Anthropic
/// doesn't use session/thread IDs or request compression.
#[derive(Default)]
pub struct MessagesOptions {
    pub extra_headers: HeaderMap,
    pub turn_state: Option<Arc<OnceLock<String>>>,
}

impl<T: HttpTransport> MessagesClient<T> {
    pub fn new(
        transport: T,
        provider: Provider,
        auth: SharedAuthProvider,
        endpoint_url: String,
    ) -> Self {
        Self {
            session: EndpointSession::new(transport, provider, auth),
            sse_telemetry: None,
            endpoint_url,
        }
    }

    pub fn with_telemetry(
        self,
        request: Option<Arc<dyn RequestTelemetry>>,
        sse: Option<Arc<dyn SseTelemetry>>,
    ) -> Self {
        Self {
            session: self.session.with_request_telemetry(request),
            sse_telemetry: sse,
            ..self
        }
    }

    #[instrument(
        name = "messages.stream_request",
        level = "info",
        skip_all,
        fields(
            transport = "messages_http",
            http.method = "POST",
            api.path = "messages"
        )
    )]
    pub async fn stream_request(
        &self,
        request: AnthropicMessagesRequest,
        options: MessagesOptions,
    ) -> Result<ResponseStream, ApiError> {
        let body = EncodedJsonBody::encode(&request)
            .map_err(|e| ApiError::Stream(format!("failed to encode messages request: {e}")))?;

        let endpoint_url = self.endpoint_url.clone();
        let stream_response = self
            .session
            .stream_encoded_json_with(
                Method::POST,
                "", // path ignored — URL is overridden below
                options.extra_headers,
                Some(body),
                |req| {
                    req.url = endpoint_url.clone();
                    req.headers.insert(
                        http::header::ACCEPT,
                        HeaderValue::from_static("text/event-stream"),
                    );
                },
            )
            .await?;

        Ok(spawn_messages_stream(
            stream_response,
            self.session.provider().stream_idle_timeout,
            self.sse_telemetry.clone(),
            options.turn_state,
        ))
    }
}
