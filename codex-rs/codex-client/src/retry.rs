use crate::error::TransportError;
use crate::request::Request;
use rand::Rng;
use std::future::Future;
use std::time::Duration;
use tokio::time::sleep;

/// Upper bound on how long a single retry will sleep when honoring a
/// server-provided `Retry-After` value, so a misbehaving upstream cannot stall
/// a turn indefinitely.
const MAX_RETRY_AFTER: Duration = Duration::from_secs(60);

#[derive(Debug, Clone)]
pub struct RetryPolicy {
    pub max_attempts: u64,
    pub base_delay: Duration,
    pub retry_on: RetryOn,
}

#[derive(Debug, Clone)]
pub struct RetryOn {
    pub retry_429: bool,
    pub retry_5xx: bool,
    pub retry_transport: bool,
}

impl RetryOn {
    pub fn should_retry(&self, err: &TransportError, attempt: u64, max_attempts: u64) -> bool {
        if attempt >= max_attempts {
            return false;
        }
        match err {
            TransportError::Http { status, .. } => {
                (self.retry_429 && status.as_u16() == 429)
                    || (self.retry_5xx && status.is_server_error())
            }
            TransportError::Timeout | TransportError::Network(_) => self.retry_transport,
            _ => false,
        }
    }
}

pub fn backoff(base: Duration, attempt: u64) -> Duration {
    if attempt == 0 {
        return base;
    }
    let exp = 2u64.saturating_pow(attempt as u32 - 1);
    let millis = base.as_millis() as u64;
    let raw = millis.saturating_mul(exp);
    let jitter: f64 = rand::rng().random_range(0.9..1.1);
    Duration::from_millis((raw as f64 * jitter) as u64)
}

/// Delay requested by the server via a `Retry-After` header on a 429 response,
/// capped at [`MAX_RETRY_AFTER`]. Returns `None` for non-429 errors, missing
/// headers, or values that are not a whole number of seconds; the HTTP-date
/// form is intentionally unsupported.
fn retry_after_delay(err: &TransportError) -> Option<Duration> {
    let TransportError::Http {
        status, headers, ..
    } = err
    else {
        return None;
    };
    if status.as_u16() != 429 {
        return None;
    }
    let headers = headers.as_ref()?;
    let value = headers.get("retry-after")?.to_str().ok()?;
    let secs: u64 = value.trim().parse().ok()?;
    Some(Duration::from_secs(secs).min(MAX_RETRY_AFTER))
}

/// Delay before the next attempt: the larger of the exponential backoff and a
/// server-requested `Retry-After` (when present), so the client never retries
/// sooner than the server asked while still growing the delay across attempts.
fn retry_delay(policy: &RetryPolicy, err: &TransportError, attempt: u64) -> Duration {
    let backoff = backoff(policy.base_delay, attempt + 1);
    retry_after_delay(err)
        .map(|ra| ra.max(backoff))
        .unwrap_or(backoff)
}

pub async fn run_with_retry<T, F, Fut>(
    policy: RetryPolicy,
    mut make_req: impl FnMut() -> Request,
    op: F,
) -> Result<T, TransportError>
where
    F: Fn(Request, u64) -> Fut,
    Fut: Future<Output = Result<T, TransportError>>,
{
    for attempt in 0..=policy.max_attempts {
        let req = make_req();
        match op(req, attempt).await {
            Ok(resp) => return Ok(resp),
            Err(err)
                if policy
                    .retry_on
                    .should_retry(&err, attempt, policy.max_attempts) =>
            {
                sleep(retry_delay(&policy, &err, attempt)).await;
            }
            Err(err) => return Err(err),
        }
    }
    Err(TransportError::RetryLimit)
}

#[cfg(test)]
mod tests {
    use super::*;
    use http::HeaderMap;
    use http::HeaderValue;
    use http::Method;
    use http::StatusCode;

    fn http_429(retry_after: Option<&str>) -> TransportError {
        let mut headers = HeaderMap::new();
        if let Some(value) = retry_after {
            headers.insert("retry-after", HeaderValue::from_str(value).unwrap());
        }
        TransportError::Http {
            status: StatusCode::TOO_MANY_REQUESTS,
            url: None,
            headers: Some(headers),
            body: None,
        }
    }

    fn http_err(status: StatusCode, retry_after: Option<&str>) -> TransportError {
        let mut headers = HeaderMap::new();
        if let Some(value) = retry_after {
            headers.insert("retry-after", HeaderValue::from_str(value).unwrap());
        }
        TransportError::Http {
            status,
            url: None,
            headers: Some(headers),
            body: None,
        }
    }

    fn req() -> Request {
        Request::new(Method::GET, "http://test".to_string())
    }

    #[test]
    fn parses_retry_after_seconds_on_429() {
        assert_eq!(
            retry_after_delay(&http_429(Some("5"))),
            Some(Duration::from_secs(5))
        );
    }

    #[test]
    fn caps_retry_after_at_max() {
        assert_eq!(
            retry_after_delay(&http_429(Some("3600"))),
            Some(MAX_RETRY_AFTER)
        );
    }

    #[test]
    fn missing_retry_after_returns_none() {
        assert_eq!(retry_after_delay(&http_429(None)), None);
    }

    #[test]
    fn no_headers_returns_none() {
        let err = TransportError::Http {
            status: StatusCode::TOO_MANY_REQUESTS,
            url: None,
            headers: None,
            body: None,
        };
        assert_eq!(retry_after_delay(&err), None);
    }

    #[test]
    fn non_429_ignores_retry_after() {
        assert_eq!(
            retry_after_delay(&http_err(StatusCode::INTERNAL_SERVER_ERROR, Some("5"))),
            None
        );
    }

    #[test]
    fn malformed_retry_after_returns_none() {
        assert_eq!(
            retry_after_delay(&http_429(Some("Wed, 21 Oct 2025 07:28:00 GMT"))),
            None
        );
        assert_eq!(retry_after_delay(&http_429(Some("soon"))), None);
    }

    #[test]
    fn retry_delay_takes_max_of_backoff_and_retry_after() {
        let policy = RetryPolicy {
            max_attempts: 4,
            base_delay: Duration::from_millis(100),
            retry_on: RetryOn {
                retry_429: true,
                retry_5xx: false,
                retry_transport: false,
            },
        };
        // backoff at attempt 0 equals base_delay (100ms); retry-after 2s wins.
        assert_eq!(
            retry_delay(&policy, &http_429(Some("2")), 0),
            Duration::from_secs(2)
        );
    }

    #[tokio::test]
    async fn retries_429_then_succeeds() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicU64, Ordering};

        let attempts = Arc::new(AtomicU64::new(0));
        let attempts_inner = Arc::clone(&attempts);
        let policy = RetryPolicy {
            max_attempts: 3,
            base_delay: Duration::from_millis(1),
            retry_on: RetryOn {
                retry_429: true,
                retry_5xx: false,
                retry_transport: false,
            },
        };
        let result: Result<u64, TransportError> =
            run_with_retry(policy, req, |_: Request, _: u64| {
                let attempts_inner = Arc::clone(&attempts_inner);
                async move {
                    let n = attempts_inner.fetch_add(1, Ordering::SeqCst);
                    if n == 0 {
                        Err(http_429(Some("0")))
                    } else {
                        Ok(42u64)
                    }
                }
            })
            .await;

        let value = result.unwrap();
        assert_eq!(value, 42);
        assert_eq!(attempts.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn does_not_retry_429_when_disabled() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicU64, Ordering};

        let attempts = Arc::new(AtomicU64::new(0));
        let attempts_inner = Arc::clone(&attempts);
        let policy = RetryPolicy {
            max_attempts: 3,
            base_delay: Duration::from_millis(1),
            retry_on: RetryOn {
                retry_429: false,
                retry_5xx: false,
                retry_transport: false,
            },
        };
        let result: Result<u64, TransportError> =
            run_with_retry(policy, req, |_: Request, _: u64| {
                let attempts_inner = Arc::clone(&attempts_inner);
                async move {
                    attempts_inner.fetch_add(1, Ordering::SeqCst);
                    Err(http_429(Some("0")))
                }
            })
            .await;

        assert!(matches!(
            result,
            Err(TransportError::Http { status, .. }) if status == StatusCode::TOO_MANY_REQUESTS
        ));
        assert_eq!(attempts.load(Ordering::SeqCst), 1);
    }
}
