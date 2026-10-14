//! Host discovery and pressure sampling (directive §20).
//!
//! Linux only for now: everything is read from `/proc` and `/sys`, plus
//! `nvidia-smi` when it is installed. Nothing here needs privileges.

use crate::config::PressureConfig;
use serde::Serialize;
use std::fs;
use std::process::Command;

#[derive(Debug, Clone, Serialize)]
pub struct HostInfo {
    pub hostname: String,
    pub cpu_model: String,
    pub cores: usize,
    pub memory_mb: u64,
    pub gpus: Vec<Gpu>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Gpu {
    pub vendor: String,
    pub name: String,
    pub vram_mb: Option<u64>,
    pub source: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct Pressure {
    pub load_1m: f64,
    pub load_per_core: f64,
    pub memory_available_fraction: f64,
    pub on_battery: bool,
    /// Highest GPU utilisation (0–1), when `nvidia-smi` is available.
    pub gpu_utilisation: Option<f64>,
    /// Hottest thermal zone in °C, when the kernel reports one.
    pub temperature_c: Option<f64>,
    pub busy: bool,
    pub reasons: Vec<String>,
}

pub fn discover() -> HostInfo {
    let cpuinfo = fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
    let cpu_model = cpuinfo
        .lines()
        .find_map(|l| l.strip_prefix("model name").and_then(|r| r.split_once(':')).map(|(_, v)| v.trim().to_string()))
        .unwrap_or_else(|| "unknown".into());
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let memory_mb = meminfo("MemTotal").map(|kb| kb / 1024).unwrap_or(0);
    let hostname = fs::read_to_string("/proc/sys/kernel/hostname").map(|s| s.trim().to_string()).unwrap_or_default();
    HostInfo { hostname, cpu_model, cores, memory_mb, gpus: gpus() }
}

pub fn sample(cfg: &PressureConfig, cores: usize) -> Pressure {
    let load_1m = fs::read_to_string("/proc/loadavg")
        .ok()
        .and_then(|s| s.split_whitespace().next().and_then(|v| v.parse().ok()))
        .unwrap_or(0.0);
    let load_per_core = load_1m / cores.max(1) as f64;
    let memory_available_fraction = match (meminfo("MemAvailable"), meminfo("MemTotal")) {
        (Some(a), Some(t)) if t > 0 => a as f64 / t as f64,
        _ => 1.0,
    };
    let on_battery = on_battery();
    let gpu_utilisation = gpu_utilisation();
    let temperature_c = hottest_zone();
    let mut reasons = Vec::new();
    if let Some(u) = gpu_utilisation
        && u > cfg.gpu_busy_utilisation
    {
        reasons.push(format!("GPU {:.0}% busy > {:.0}%", u * 100.0, cfg.gpu_busy_utilisation * 100.0));
    }
    if let Some(t) = temperature_c
        && t > cfg.max_temperature_c
    {
        reasons.push(format!("{t:.0}°C > {:.0}°C", cfg.max_temperature_c));
    }
    if load_per_core > cfg.busy_load_per_core {
        reasons.push(format!("load {load_per_core:.2}/core > {:.2}", cfg.busy_load_per_core));
    }
    if memory_available_fraction < cfg.min_available_memory {
        reasons.push(format!(
            "available memory {:.0}% < {:.0}%",
            memory_available_fraction * 100.0,
            cfg.min_available_memory * 100.0
        ));
    }
    if on_battery && cfg.battery_is_busy {
        reasons.push("running on battery".into());
    }
    let busy = !cfg.disabled && !reasons.is_empty();
    Pressure {
        load_1m,
        load_per_core,
        memory_available_fraction,
        on_battery,
        gpu_utilisation,
        temperature_c,
        busy,
        reasons,
    }
}

/// Available memory now, in MB.
pub fn available_memory_mb() -> u64 {
    meminfo("MemAvailable").map(|kb| kb / 1024).unwrap_or(0)
}

/// Highest GPU utilisation from `nvidia-smi`, cached for a few seconds
/// because the tool is slow to start.
fn gpu_utilisation() -> Option<f64> {
    use std::sync::Mutex;
    use std::time::{Duration, Instant};
    static CACHE: Mutex<Option<(Instant, Option<f64>)>> = Mutex::new(None);
    let mut cache = CACHE.lock().unwrap_or_else(|p| p.into_inner());
    if let Some((at, v)) = *cache
        && at.elapsed() < Duration::from_secs(5)
    {
        return v;
    }
    let v = Command::new("nvidia-smi")
        .args(["--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| parse_gpu_utilisation(&String::from_utf8_lossy(&o.stdout)));
    *cache = Some((Instant::now(), v));
    v
}

fn parse_gpu_utilisation(text: &str) -> Option<f64> {
    text.lines().filter_map(|l| l.trim().parse::<f64>().ok()).map(|p| p / 100.0).reduce(f64::max)
}

/// The hottest `/sys/class/thermal` zone, in °C.
fn hottest_zone() -> Option<f64> {
    let entries = fs::read_dir("/sys/class/thermal").ok()?;
    entries
        .flatten()
        .filter(|e| e.file_name().to_string_lossy().starts_with("thermal_zone"))
        .filter_map(|e| fs::read_to_string(e.path().join("temp")).ok())
        .filter_map(|t| parse_millidegrees(&t))
        .reduce(f64::max)
}

fn parse_millidegrees(text: &str) -> Option<f64> {
    // Some zones report nonsense (negative, or > 200 °C) when absent.
    text.trim().parse::<f64>().ok().map(|m| m / 1000.0).filter(|c| (0.0..200.0).contains(c))
}

fn meminfo(key: &str) -> Option<u64> {
    let text = fs::read_to_string("/proc/meminfo").ok()?;
    text.lines()
        .find_map(|l| l.strip_prefix(key)?.strip_prefix(':').map(str::trim))
        .and_then(|v| v.split_whitespace().next()?.parse().ok())
}

/// True when a battery reports that it is discharging.
fn on_battery() -> bool {
    let Ok(entries) = fs::read_dir("/sys/class/power_supply") else { return false };
    entries.flatten().any(|e| {
        let p = e.path();
        fs::read_to_string(p.join("type")).unwrap_or_default().trim() == "Battery"
            && fs::read_to_string(p.join("status")).unwrap_or_default().trim() == "Discharging"
    })
}

fn gpus() -> Vec<Gpu> {
    let mut out = Vec::new();
    if let Ok(o) =
        Command::new("nvidia-smi").args(["--query-gpu=name,memory.total", "--format=csv,noheader,nounits"]).output()
        && o.status.success()
    {
        for line in String::from_utf8_lossy(&o.stdout).lines() {
            let mut parts = line.split(',').map(str::trim);
            let name = parts.next().unwrap_or("").to_string();
            let vram_mb = parts.next().and_then(|v| v.parse().ok());
            out.push(Gpu { vendor: "nvidia".into(), name, vram_mb, source: "nvidia-smi".into() });
        }
    }
    // Other display devices from sysfs (covers integrated and AMD/Intel GPUs).
    if let Ok(entries) = fs::read_dir("/sys/class/drm") {
        let mut names: Vec<_> = entries.flatten().map(|e| e.file_name().to_string_lossy().into_owned()).collect();
        names.sort();
        for name in names.into_iter().filter(|n| n.starts_with("card") && !n.contains('-')) {
            let dev = format!("/sys/class/drm/{name}/device");
            let vendor_id = fs::read_to_string(format!("{dev}/vendor")).unwrap_or_default();
            let vendor = match vendor_id.trim() {
                "0x10de" => "nvidia",
                "0x1002" => "amd",
                "0x8086" => "intel",
                "" => continue,
                other => other,
            };
            if vendor == "nvidia" && out.iter().any(|g| g.source == "nvidia-smi") {
                continue;
            }
            let vram_mb = fs::read_to_string(format!("{dev}/mem_info_vram_total"))
                .ok()
                .and_then(|v| v.trim().parse::<u64>().ok())
                .map(|b| b / (1024 * 1024));
            out.push(Gpu { vendor: vendor.into(), name, vram_mb, source: "sysfs".into() });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_gpu_and_thermal_readings() {
        assert_eq!(parse_gpu_utilisation("12\n87\n"), Some(0.87));
        assert_eq!(parse_gpu_utilisation("[N/A]\n"), None);
        assert_eq!(parse_millidegrees("54000\n"), Some(54.0));
        assert_eq!(parse_millidegrees("-273000"), None);
    }

    #[test]
    fn gpu_and_heat_count_as_pressure() {
        let cfg = PressureConfig { gpu_busy_utilisation: -1.0, ..PressureConfig::default() };
        // With a threshold below zero any reading is "busy"; without a GPU
        // there is no reading and so no reason.
        let p = sample(&cfg, 1);
        assert_eq!(p.reasons.iter().any(|r| r.starts_with("GPU")), p.gpu_utilisation.is_some());
    }
}
