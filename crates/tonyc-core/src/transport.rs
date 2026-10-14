//! Where the daemon listens. Shared by `tonycd`, the SDKs and the CLI so
//! that every client finds the daemon the same way.

use std::path::PathBuf;

/// `$TONYC_SOCKET`, else `$XDG_RUNTIME_DIR/tonyc/tonycd.sock`, else
/// `$XDG_STATE_HOME/tonyc/tonycd.sock` (default
/// `$HOME/.local/state/tonyc/tonycd.sock`).
pub fn default_socket() -> PathBuf {
    let var = |name: &str| std::env::var_os(name).filter(|v| !v.is_empty()).map(PathBuf::from);
    if let Some(p) = var("TONYC_SOCKET") {
        return p;
    }
    let base = var("XDG_RUNTIME_DIR")
        .or_else(|| var("XDG_STATE_HOME"))
        .unwrap_or_else(|| var("HOME").unwrap_or_else(|| PathBuf::from("/")).join(".local/state"));
    base.join("tonyc").join("tonycd.sock")
}
