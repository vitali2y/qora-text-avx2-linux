#[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
compile_error!("QORA CPU-only AVX2 release currently supports Linux x86_64 only");

pub mod config;
pub mod gemv;
pub mod generate;
pub mod save;
pub mod simd;
pub mod system;
pub mod tokenizer;
