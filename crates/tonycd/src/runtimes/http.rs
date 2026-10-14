//! A minimal HTTP/1.1 client that connects to one already-classified socket
//! address. Resolving and classifying first, then connecting only to that
//! exact address, means a DNS change can never move a request to a host the
//! locality policy did not see.

use http_body_util::{BodyExt, Full};
use hyper::body::Bytes;
use hyper::{Request, StatusCode};
use hyper_util::rt::TokioIo;
use std::net::SocketAddr;
use std::time::Duration;
use tokio::net::TcpStream;

pub struct Reply {
    pub status: StatusCode,
    pub body: Bytes,
    pub bytes_sent: u64,
}

pub async fn request(
    addr: SocketAddr,
    host_header: &str,
    method: &str,
    path: &str,
    headers: &[(&str, String)],
    body: Option<Vec<u8>>,
    timeout: Duration,
) -> Result<Reply, String> {
    tokio::time::timeout(timeout, async {
        let stream = TcpStream::connect(addr).await.map_err(|e| format!("connect {addr}: {e}"))?;
        let (mut sender, conn) =
            hyper::client::conn::http1::handshake(TokioIo::new(stream)).await.map_err(|e| format!("handshake: {e}"))?;
        let driver = tokio::spawn(conn);
        let payload = body.unwrap_or_default();
        let bytes_sent = (payload.len() + path.len() + host_header.len() + 64) as u64;
        let mut req = Request::builder()
            .method(method)
            .uri(path)
            .header("host", host_header)
            .header("content-type", "application/json");
        for (k, v) in headers {
            req = req.header(*k, v);
        }
        let req = req.body(Full::new(Bytes::from(payload))).map_err(|e| e.to_string())?;
        let res = sender.send_request(req).await.map_err(|e| format!("request: {e}"))?;
        let status = res.status();
        let body = res.into_body().collect().await.map_err(|e| format!("body: {e}"))?.to_bytes();
        driver.abort();
        Ok(Reply { status, body, bytes_sent })
    })
    .await
    .map_err(|_| format!("timed out after {}s", timeout.as_secs()))?
}
