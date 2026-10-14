//! `tonycd`, the Tonyc daemon (directive TONYC/1 §5).
//!
//! It owns the persistent queue, scheduling, model discovery, policy
//! enforcement, provenance and the local API. Applications talk to it over
//! a Unix socket; nothing needs to embed it.

pub mod api;
pub mod bench;
pub mod cognition;
pub mod cogstore;
pub mod config;
pub mod daemon;
pub mod economics;
pub mod fabric;
pub mod host;
pub mod pd;
pub mod protocol;
pub mod refinement;
pub mod runtimes;
pub mod scheduler;
pub mod store;
pub mod trace;

use anyhow::{Context, Result, bail};
use config::Config;
use daemon::Daemon;
use runtimes::Registry;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use store::Store;
use tokio::net::UnixListener;

/// A running daemon. Dropping it does not stop it; call [`Running::stop`].
pub struct Running {
    pub daemon: Arc<Daemon>,
    pub socket: PathBuf,
    tasks: Vec<tokio::task::JoinHandle<()>>,
    _lock: std::fs::File,
}

impl Running {
    /// Graceful stop: in-flight jobs are interrupted and requeued.
    pub async fn stop(self) {
        self.daemon.shutdown.cancel();
        let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
        while self.daemon.running_count() > 0 && tokio::time::Instant::now() < deadline {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        for t in self.tasks {
            let _ = tokio::time::timeout(Duration::from_secs(2), t).await;
        }
        let _ = std::fs::remove_file(&self.socket);
    }
}

pub async fn start(cfg: Config) -> Result<Running> {
    let registry = Registry::from_config(&cfg);
    start_with(cfg, registry).await
}

/// Starts the daemon with an explicit set of runtimes.
pub async fn start_with(cfg: Config, registry: Registry) -> Result<Running> {
    cfg.validate()?;
    let data_dir = cfg.data_dir();
    private_dir(&data_dir)?;
    let lock = exclusive_lock(&data_dir.join("tonycd.lock")).await?;
    let store = Arc::new(Store::open(&data_dir.join("state.db"))?);
    let recovered = store.recover()?;
    if !recovered.is_empty() {
        tracing::info!("requeued {} job(s) interrupted by the last shutdown", recovered.len());
    }

    registry.refresh().await;

    let socket = cfg.socket_path();
    let listener = bind_socket(&socket)?;
    let tcp = match &cfg.daemon.tcp {
        Some(addr) => {
            let addr: std::net::SocketAddr = addr.parse().with_context(|| format!("daemon.tcp `{addr}`"))?;
            if !addr.ip().is_loopback() {
                bail!("daemon.tcp must be a loopback address, not {addr}");
            }
            Some(tokio::net::TcpListener::bind(addr).await.with_context(|| format!("binding {addr}"))?)
        }
        None => None,
    };

    let d = Arc::new(Daemon::new(cfg, store, registry));
    d.load_nodes()?;
    d.load_observed()?;
    let mut tasks = vec![
        tokio::spawn(scheduler::run(d.clone())),
        tokio::spawn(refresher(d.clone())),
        tokio::spawn(cognition_scanner(d.clone())),
        tokio::spawn(pd_loop(d.clone())),
    ];

    let app = api::router(d.clone());
    let shutdown = d.shutdown.clone();
    tasks.push(tokio::spawn(async move {
        let _ = axum::serve(listener, app).with_graceful_shutdown(shutdown.cancelled_owned()).await;
    }));
    if let Some(tcp) = tcp {
        let app = api::router(d.clone());
        let shutdown = d.shutdown.clone();
        tasks.push(tokio::spawn(async move {
            let _ = axum::serve(tcp, app).with_graceful_shutdown(shutdown.cancelled_owned()).await;
        }));
    }
    tracing::info!("listening on {}", socket.display());
    Ok(Running { daemon: d, socket, tasks, _lock: lock })
}

/// Rediscovers runtimes and probes nodes: at once (nodes loaded from the
/// state database have not been probed yet), then periodically, more often
/// while something is down.
async fn refresher(d: Arc<Daemon>) {
    loop {
        tokio::select! {
            _ = d.registry.refresh() => {}
            _ = d.shutdown.cancelled() => break,
        }
        d.wake.notify_one();
        let any_down = d.registry.adapters().iter().any(|a| !a.healthy());
        let every = d.cfg.fabric.probe_interval_secs.clamp(5, 3600);
        let wait = Duration::from_secs(if any_down { 10.min(every) } else { every });
        tokio::select! {
            _ = tokio::time::sleep(wait) => {}
            _ = d.shutdown.cancelled() => break,
        }
    }
}

/// Re-checks the corpus and every fresh object so that staleness is found
/// before anyone asks (a use-time check happens regardless).
async fn cognition_scanner(d: Arc<Daemon>) {
    let every = Duration::from_secs(d.cfg.cognition.scan_interval_secs.max(1));
    loop {
        tokio::select! {
            _ = tokio::time::sleep(every) => {}
            _ = d.shutdown.cancelled() => break,
        }
        let store = d.store.clone();
        match tokio::task::spawn_blocking(move || cognition::scan(&store)).await {
            Ok(Ok(r)) if !r.invalidated.objects.is_empty() => {
                tracing::info!(
                    objects = r.invalidated.objects.len(),
                    jobs = r.invalidated.jobs.len(),
                    "invalidated stale cognition"
                )
            }
            Ok(Ok(_)) => {}
            Ok(Err(e)) => tracing::error!("cognition scan failed: {e:#}"),
            Err(e) => tracing::error!("cognition scan panicked: {e}"),
        }
    }
}

/// Preemptive Dynamism: evaluate demand periodically.
async fn pd_loop(d: Arc<Daemon>) {
    let every = Duration::from_secs(d.cfg.pd.interval_secs.max(1));
    loop {
        tokio::select! {
            _ = tokio::time::sleep(every) => {}
            _ = d.shutdown.cancelled() => break,
        }
        match pd::evaluate(&d) {
            Ok(e) => {
                if !e.submitted.is_empty() {
                    tracing::info!(jobs = e.submitted.len(), "preparing predicted cognition");
                }
                *d.pd_last.lock().unwrap() = Some(e);
            }
            Err(e) => tracing::error!("PD evaluation failed: {e:#}"),
        }
    }
}

fn private_dir(dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700))?;
    Ok(())
}

/// Takes the state directory's lock. It is retried for a short while: a
/// daemon that has just exited — or a process forked from one in the
/// instant before it `exec`s, which shares its descriptors — can still hold
/// it briefly.
async fn exclusive_lock(path: &Path) -> Result<std::fs::File> {
    use std::os::fd::AsRawFd;
    let f = std::fs::OpenOptions::new().create(true).truncate(false).write(true).open(path)?;
    for _ in 0..100 {
        if unsafe { libc::flock(f.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } == 0 {
            return Ok(f);
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    bail!("another tonycd holds {}", path.display())
}

fn bind_socket(path: &Path) -> Result<UnixListener> {
    if let Some(parent) = path.parent() {
        private_dir(parent)?;
    }
    if path.exists() {
        if std::os::unix::net::UnixStream::connect(path).is_ok() {
            bail!("a daemon is already listening on {}", path.display());
        }
        std::fs::remove_file(path).with_context(|| format!("removing stale socket {}", path.display()))?;
    }
    let listener = UnixListener::bind(path).with_context(|| format!("binding {}", path.display()))?;
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600))?;
    Ok(listener)
}
