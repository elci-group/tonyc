//! Locating an HTTP runtime: parsing its URL and classifying where the
//! address it resolves to actually is.

use std::net::{IpAddr, SocketAddr};
use tonyc_core::Exposure;

/// `http://host:port[/]` → (host, port). Only plain HTTP is accepted: the
/// adapter is meant for a runtime on this host or the LAN.
pub(crate) fn parse_url(what: &str, url: &str) -> Result<(String, u16), String> {
    let rest = url.strip_prefix("http://").ok_or_else(|| format!("{what} url `{url}` must start with http://"))?;
    let authority = rest.split('/').next().unwrap_or("");
    if authority.is_empty() {
        return Err(format!("{what} url `{url}` has no host"));
    }
    if let Some(v6) = authority.strip_prefix('[') {
        let (host, after) = v6.split_once(']').ok_or("unterminated IPv6 address")?;
        let port = after.strip_prefix(':').map(|p| p.parse::<u16>()).transpose().map_err(|e| e.to_string())?;
        return Ok((host.to_string(), port.unwrap_or(80)));
    }
    match authority.rsplit_once(':') {
        Some((h, p)) => Ok((h.to_string(), p.parse().map_err(|_| format!("bad port in `{url}`"))?)),
        None => Ok((authority.to_string(), 80)),
    }
}

pub(crate) fn classify(host: &str, port: u16, addrs: &[SocketAddr]) -> Exposure {
    if addrs.iter().all(|a| a.ip().is_loopback()) {
        Exposure::Loopback { endpoint: format!("{host}:{port}") }
    } else if addrs.iter().all(|a| is_private(a.ip())) {
        Exposure::Lan { host: host.to_string() }
    } else {
        Exposure::Remote { host: host.to_string() }
    }
}

pub(crate) fn is_private(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => v4.is_private() || v4.is_link_local(),
        IpAddr::V6(v6) => {
            let seg = v6.segments()[0];
            (seg & 0xfe00) == 0xfc00 || (seg & 0xffc0) == 0xfe80
        }
    }
}

/// Resolves `host:port` and classifies every address it resolves to.
pub(crate) async fn locate(host: &str, port: u16) -> Result<(SocketAddr, Exposure), String> {
    let addrs: Vec<SocketAddr> =
        tokio::net::lookup_host((host, port)).await.map_err(|e| format!("resolving {host}: {e}"))?.collect();
    let addr = *addrs.first().ok_or_else(|| format!("{host} resolved to no addresses"))?;
    Ok((addr, classify(host, port, &addrs)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_urls() {
        assert_eq!(parse_url("x", "http://127.0.0.1:11434"), Ok(("127.0.0.1".into(), 11434)));
        assert_eq!(parse_url("x", "http://localhost/"), Ok(("localhost".into(), 80)));
        assert_eq!(parse_url("x", "http://[::1]:11434"), Ok(("::1".into(), 11434)));
        assert!(parse_url("x", "https://api.example.com").is_err());
    }

    #[test]
    fn classifies_by_resolved_address() {
        let lo: SocketAddr = "127.0.0.1:1".parse().unwrap();
        let lan: SocketAddr = "192.168.1.5:1".parse().unwrap();
        let wan: SocketAddr = "8.8.8.8:1".parse().unwrap();
        assert!(matches!(classify("h", 1, &[lo]), Exposure::Loopback { .. }));
        assert!(matches!(classify("h", 1, &[lan]), Exposure::Lan { .. }));
        assert!(matches!(classify("h", 1, &[wan]), Exposure::Remote { .. }));
        // Mixed resolution takes the most exposed reading.
        assert!(matches!(classify("h", 1, &[lo, wan]), Exposure::Remote { .. }));
    }
}
