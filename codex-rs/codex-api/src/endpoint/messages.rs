//! HTTP client for the Anthropic Messages API (`POST /v1/messages`).
//!
//! The endpoint URL is supplied by the caller (typically from
//! `ModelProviderInfo::messages_endpoint_url()`) rather than assembled
//! from a provider base URL + path, so that users can override the full
//! URL for providers that speak the Messages wire protocol at custom
//! locations (e.g. DashScope, Ollama, LM Studio).

use crate::auth::SharedAuthProvider;
use crate::anthropic::AnthropicMessagesRequest;
use crate::common::ResponseStream;
use crate::endpoint::session::EndpointSession;
use crate::error::ApiError;
use crate::provider::Provider;
use crate::sse::spawn_messages_stream;
use crate::telemetry::SseTelemetry;
use codex_client::HttpTransport;
use codex_client::RequestCompression;
use codex_client::RequestTelemetry;
use http::HeaderMap;
use http::HeaderValue;
use http::Method;
use serde_json::Value;
use std::sync::Arc;
use tracing::instrument;

/// HTTP client for the Anthropic Messages API streaming endpoint.
///
/// Unlike `ResponsesClient`, this client POSTs to a caller-supplied URL
/// (typically from `ModelProviderInfo::messages_endpoint_url()`) rather
/// than assembling a path from a provider base URL. This allows users
/// to target arbitrary Messages-API-compatible providers.
///
/// The client is generic over `HttpTransport` so it can be used with
/// both real HTTP clients and test doubles.
pub struct MessagesClient<T: HttpTransport> {
    session: EndpointSession<T>,
    sse_telemetry: Option<Arc<dyn SseTelemetry>>,
    /// Full endpoint URL (e.g. `https://api.anthropic.com/v1/messages`).
    endpoint_url: String,
}

/// Per-request options for the Messages API. Lighter than
/// `ResponsesOptions` because Anthropic doesn't use session/thread
/// IDs or request compression.
pub struct MessagesOptions {
    pub extra_headers: HeaderMap,
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
        let body = serde_json::to_value(&request)
            .map_err(|e| ApiError::Stream(format!("failed to encode messages request: {e}")))?;

        self.stream(body, options.extra_headers).await
    }

    #[instrument(
        name = "messages.stream",
        level = "info",
        skip_all,
        fields(
            transport = "messages_http",
            http.method = "POST",
            api.path = "messages"
        )
    )]
    pub async fn stream(
        &self,
        body: Value,
        extra_headers: HeaderMap,
    ) -> Result<ResponseStream, ApiError> {
        // Use `stream_with` with an empty path; the configure callback
        // overrides the URL to use the full endpoint URL directly.
        let endpoint_url = self.endpoint_url.clone();
        let stream_response = self
            .session
            .stream_with(
                Method::POST,
                "", // path ignored — URL is overridden below
                extra_headers,
                Some(body),
                |req| {
                    req.url = endpoint_url.clone();
                    req.headers.insert(
                        http::header::ACCEPT,
                        HeaderValue::from_static("text/event-stream"),
                    );
                    req.compression = RequestCompression::None;
                },
            )
            .await?;

        Ok(spawn_messages_stream(
            stream_response,
            self.session.provider().stream_idle_timeout,
            self.sse_telemetry.clone(),
            /*turn_state*/ None,
        ))
    }
}