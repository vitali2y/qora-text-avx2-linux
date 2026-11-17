//! System awareness: detect RAM, CPU threads, and smart parameter adjustment

/// System information detected at startup
pub struct SystemInfo {
    pub total_ram_mb: u64,
    pub available_ram_mb: u64,
    pub cpu_threads: usize,
}

/// Smart parameter limits based on system capabilities
pub struct SmartLimits {
    pub max_think_budget: usize,
    pub max_tokens: usize,
    pub default_think_budget: usize,
    pub default_max_tokens: usize,
    pub warning: Option<&'static str>,
}

impl SystemInfo {
    /// Detect system info from Linux /proc/meminfo
    /// Falls back to conservative defaults if detection fails
    pub fn detect() -> Self {
        let cpu_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        let (total, available) = detect_ram();

        SystemInfo {
            total_ram_mb: total,
            available_ram_mb: available,
            cpu_threads,
        }
    }

    /// Compute smart limits based on available RAM
    /// Returns both defaults (for when user didn't specify) and hard caps
    /// (to clamp even explicit user values on weak systems)
    pub fn smart_limits(&self) -> SmartLimits {
        let avail = self.available_ram_mb;
        if avail < 4096 {
            SmartLimits {
                max_think_budget: 256,
                max_tokens: 512,
                default_think_budget: 128,
                default_max_tokens: 256,
                warning: Some("Very low RAM — capping generation to 512 tokens"),
            }
        } else if avail < 8192 {
            SmartLimits {
                max_think_budget: 1024,
                max_tokens: 1024,
                default_think_budget: 256,
                default_max_tokens: 512,
                warning: Some("Low RAM — capping generation to 1024 tokens"),
            }
        } else if avail < 12288 {
            SmartLimits {
                max_think_budget: 2048,
                max_tokens: 2048,
                default_think_budget: 1024,
                default_max_tokens: 1024,
                warning: None,
            }
        } else {
            SmartLimits {
                max_think_budget: 8192,
                max_tokens: 8192,
                default_think_budget: 2048,
                default_max_tokens: 2048,
                warning: None,
            }
        }
    }
}

/// Detect total and available RAM in MB
fn detect_ram() -> (u64, u64) {
    #[cfg(target_os = "linux")]
    {
        if let Some((total, avail)) = detect_ram_linux() {
            return (total, avail);
        }
    }

    // Fallback: assume 8 GB total, 4 GB available
    (8192, 4096)
}

#[cfg(target_os = "linux")]
fn detect_ram_linux() -> Option<(u64, u64)> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    let mut total_kb = 0u64;
    let mut available_kb = 0u64;
    for line in text.lines() {
        if line.starts_with("MemTotal:") {
            total_kb = parse_meminfo_kb(line);
        } else if line.starts_with("MemAvailable:") {
            available_kb = parse_meminfo_kb(line);
        }
    }
    (total_kb > 0).then_some((total_kb / 1024, available_kb / 1024))
}

#[cfg(target_os = "linux")]
fn parse_meminfo_kb(line: &str) -> u64 {
    line.split_whitespace()
        .nth(1)
        .and_then(|value| value.parse().ok())
        .unwrap_or(0)
}
