//! Locality semantics (directive §7, invariant T1).
//!
//! Locality is enforced structurally: every execution domain declares the
//! [`Exposure`] its use implies, and a plan is admissible only if *both* the
//! job's policy and the daemon's ceiling policy permit that exposure. No
//! score, cost or capability can outweigh a refusal here; the planner applies
//! this filter before it looks at anything else.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::str::FromStr;

/// The locality class a job (or the daemon ceiling) permits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Locality {
    /// In-process or loopback execution on this host only.
    LocalOnly,
    /// Also hosts on private LAN addresses.
    LanAllowed,
    /// Local, plus SSH nodes named in the allowlist.
    SshAllowlist,
    /// Local, plus remote endpoints named in the allowlist.
    RemoteAllowlist,
    /// Anything an enabled execution domain offers.
    Unrestricted,
}

impl Locality {
    pub const ALL: [Locality; 5] = [
        Locality::LocalOnly,
        Locality::LanAllowed,
        Locality::SshAllowlist,
        Locality::RemoteAllowlist,
        Locality::Unrestricted,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Locality::LocalOnly => "LOCAL_ONLY",
            Locality::LanAllowed => "LAN_ALLOWED",
            Locality::SshAllowlist => "SSH_ALLOWLIST",
            Locality::RemoteAllowlist => "REMOTE_ALLOWLIST",
            Locality::Unrestricted => "UNRESTRICTED",
        }
    }
}

impl fmt::Display for Locality {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

impl FromStr for Locality {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        let norm = s.trim().to_ascii_uppercase().replace('-', "_");
        Locality::ALL.into_iter().find(|l| l.as_str() == norm).ok_or_else(|| {
            format!("unknown locality `{s}` (expected one of {})", Locality::ALL.map(|l| l.as_str()).join(", "))
        })
    }
}

impl<'de> Deserialize<'de> for Locality {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        String::deserialize(d)?.parse().map_err(serde::de::Error::custom)
    }
}

/// Where data goes when an execution domain runs a job.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Exposure {
    /// Computed inside the daemon process.
    InProcess,
    /// Sent to another process on this host over a loopback address or a
    /// Unix socket.
    Loopback { endpoint: String },
    /// Sent to a host on a private network address.
    Lan { host: String },
    /// Sent to an authorised SSH compute node. A `shared` node (one other
    /// people also use) is only ever used when a job names it explicitly.
    Ssh {
        node: String,
        #[serde(default, skip_serializing_if = "std::ops::Not::not")]
        shared: bool,
    },
    /// Sent to a host outside the user's network.
    Remote { host: String },
}

impl Exposure {
    /// Whether using this domain moves data off this host.
    pub fn leaves_host(&self) -> bool {
        !matches!(self, Exposure::InProcess | Exposure::Loopback { .. })
    }

    pub fn label(&self) -> String {
        match self {
            Exposure::InProcess => "in-process".into(),
            Exposure::Loopback { endpoint } => format!("loopback ({endpoint})"),
            Exposure::Lan { host } => format!("LAN ({host})"),
            Exposure::Ssh { node, shared: false } => format!("SSH node {node}"),
            Exposure::Ssh { node, shared: true } => format!("shared SSH node {node}"),
            Exposure::Remote { host } => format!("remote ({host})"),
        }
    }
}

/// A locality class together with the allowlists it refers to.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LocalityPolicy {
    pub class: Locality,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ssh_allowlist: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub remote_allowlist: Vec<String>,
}

impl LocalityPolicy {
    pub fn local_only() -> Self {
        LocalityPolicy { class: Locality::LocalOnly, ssh_allowlist: vec![], remote_allowlist: vec![] }
    }

    /// Returns `Err(reason)` if this policy forbids the exposure.
    pub fn permits(&self, exposure: &Exposure) -> Result<(), String> {
        use Locality::*;
        let class = self.class;
        match exposure {
            Exposure::InProcess | Exposure::Loopback { .. } => Ok(()),
            Exposure::Lan { host } => match class {
                LanAllowed | Unrestricted => Ok(()),
                _ => Err(format!("{class} forbids LAN host {host}")),
            },
            Exposure::Ssh { node, shared } => {
                let named = self.ssh_allowlist.iter().any(|n| n == node);
                match class {
                    Unrestricted | SshAllowlist if named => Ok(()),
                    Unrestricted if !shared => Ok(()),
                    Unrestricted => Err(format!("shared SSH node {node} must be named in the ssh allowlist")),
                    SshAllowlist => Err(format!("SSH node {node} is not in the allowlist")),
                    _ => Err(format!("{class} forbids SSH node {node}")),
                }
            }
            Exposure::Remote { host } => match class {
                Unrestricted => Ok(()),
                RemoteAllowlist if self.remote_allowlist.iter().any(|h| h == host) => Ok(()),
                RemoteAllowlist => Err(format!("remote host {host} is not in the allowlist")),
                _ => Err(format!("{class} forbids remote host {host}")),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn remote() -> Exposure {
        Exposure::Remote { host: "api.example.com".into() }
    }

    #[test]
    fn local_only_permits_only_this_host() {
        let p = LocalityPolicy::local_only();
        assert!(p.permits(&Exposure::InProcess).is_ok());
        assert!(p.permits(&Exposure::Loopback { endpoint: "127.0.0.1:11434".into() }).is_ok());
        assert!(p.permits(&Exposure::Lan { host: "10.0.0.2".into() }).is_err());
        assert!(p.permits(&Exposure::Ssh { node: "forge".into(), shared: false }).is_err());
        assert!(p.permits(&remote()).is_err());
    }

    #[test]
    fn shared_nodes_need_naming_even_when_unrestricted() {
        let open = LocalityPolicy { class: Locality::Unrestricted, ssh_allowlist: vec![], remote_allowlist: vec![] };
        let shared = Exposure::Ssh { node: "lab".into(), shared: true };
        assert!(open.permits(&Exposure::Ssh { node: "forge".into(), shared: false }).is_ok());
        assert!(open.permits(&shared).is_err());
        let named = LocalityPolicy { ssh_allowlist: vec!["lab".into()], ..open };
        assert!(named.permits(&shared).is_ok());
    }

    #[test]
    fn allowlists_are_exact() {
        let p = LocalityPolicy {
            class: Locality::SshAllowlist,
            ssh_allowlist: vec!["forge".into()],
            remote_allowlist: vec![],
        };
        assert!(p.permits(&Exposure::Ssh { node: "forge".into(), shared: false }).is_ok());
        assert!(p.permits(&Exposure::Ssh { node: "forge2".into(), shared: false }).is_err());
        assert!(p.permits(&Exposure::Lan { host: "10.0.0.2".into() }).is_err());
        assert!(p.permits(&remote()).is_err());
    }

    #[test]
    fn parses_case_insensitively() {
        assert_eq!("local_only".parse::<Locality>(), Ok(Locality::LocalOnly));
        assert_eq!("ssh-allowlist".parse::<Locality>(), Ok(Locality::SshAllowlist));
        assert!("everywhere".parse::<Locality>().is_err());
    }
}
