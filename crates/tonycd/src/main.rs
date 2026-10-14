use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;
use tonycd::config::Config;

/// tonycd — the Tonyc deferred-inference daemon.
#[derive(Parser)]
#[command(version)]
struct Args {
    /// Config file (default: $XDG_CONFIG_HOME/tonyc/tonycd.toml if present).
    #[arg(long)]
    config: Option<PathBuf>,
    /// Override the Unix socket path.
    #[arg(long)]
    socket: Option<PathBuf>,
    /// Override the state directory.
    #[arg(long)]
    data_dir: Option<PathBuf>,
    /// Print the effective configuration as TOML and exit.
    #[arg(long)]
    print_config: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_env("TONYC_LOG").unwrap_or_else(|_| "tonycd=info".into()),
        )
        .with_writer(std::io::stderr)
        .init();
    let args = Args::parse();
    let mut cfg = match &args.config {
        Some(p) => Config::load(p)?,
        None if Config::default_path().exists() => Config::load(&Config::default_path())?,
        None => Config::default(),
    };
    if let Some(s) = args.socket {
        cfg.daemon.socket = Some(s);
    }
    if let Some(d) = args.data_dir {
        cfg.daemon.data_dir = Some(d);
    }
    if args.print_config {
        print!("{}", toml::to_string_pretty(&cfg)?);
        return Ok(());
    }

    let running = tonycd::start(cfg).await?;
    let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {}
        _ = term.recv() => {}
    }
    tracing::info!("shutting down; in-flight jobs will be requeued");
    running.stop().await;
    Ok(())
}
