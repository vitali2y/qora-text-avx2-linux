//! Save/load QORA-4B ModelWeights to compact binary format (.qor4b)
//!
//! File format:
//!   Header: magic "QOR4" + version(u32) + format(u8: 0=F16, 1=Q4)
//!   Config: all architecture params
//!   Per-layer: type byte + layer-specific weights
//!   Global: embedding + final norm + RoPE tables
//!   Legacy v1/v2 files may append a vision section; the text-only loader skips it

use half::f16;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

use crate::config::{LayerType, Qor4bConfig};
use crate::gemv::*;

const MAGIC: &[u8; 4] = b"QOR4";
const VERSION: u32 = 2; // Keep v2 container compatibility; new files contain text weights only
const LEGACY_VISION_LAYERS: usize = 24;

// ============================================================
// I/O helpers
// ============================================================

fn write_u8(w: &mut impl Write, val: u8) -> io::Result<()> {
    w.write_all(&[val])
}
fn write_u32(w: &mut impl Write, val: u32) -> io::Result<()> {
    w.write_all(&val.to_le_bytes())
}
fn write_u64(w: &mut impl Write, val: u64) -> io::Result<()> {
    w.write_all(&val.to_le_bytes())
}
fn write_f32(w: &mut impl Write, val: f32) -> io::Result<()> {
    w.write_all(&val.to_le_bytes())
}
fn write_f64(w: &mut impl Write, val: f64) -> io::Result<()> {
    w.write_all(&val.to_le_bytes())
}

fn write_f16_vec(w: &mut impl Write, data: &[f16]) -> io::Result<()> {
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 2) };
    write_u64(w, data.len() as u64)?;
    w.write_all(bytes)
}

fn write_f32_vec(w: &mut impl Write, data: &[f32]) -> io::Result<()> {
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    write_u64(w, data.len() as u64)?;
    w.write_all(bytes)
}

fn write_bytes(w: &mut impl Write, data: &[u8]) -> io::Result<()> {
    write_u64(w, data.len() as u64)?;
    w.write_all(data)
}

fn read_u8(r: &mut impl Read) -> io::Result<u8> {
    let mut b = [0u8; 1];
    r.read_exact(&mut b)?;
    Ok(b[0])
}
fn read_u32(r: &mut impl Read) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}
fn read_u64(r: &mut impl Read) -> io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}
fn read_f32(r: &mut impl Read) -> io::Result<f32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(f32::from_le_bytes(b))
}
fn read_f64(r: &mut impl Read) -> io::Result<f64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(f64::from_le_bytes(b))
}

fn read_f16_vec(r: &mut impl Read) -> io::Result<Vec<f16>> {
    let len = read_u64(r)? as usize;
    let mut bytes = vec![0u8; len * 2];
    r.read_exact(&mut bytes)?;
    Ok(unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f16, len).to_vec() })
}

fn read_f32_vec(r: &mut impl Read) -> io::Result<Vec<f32>> {
    let len = read_u64(r)? as usize;
    let mut bytes = vec![0u8; len * 4];
    r.read_exact(&mut bytes)?;
    Ok(unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, len).to_vec() })
}

fn read_bytes_vec(r: &mut impl Read) -> io::Result<Vec<u8>> {
    let len = read_u64(r)? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    Ok(buf)
}

fn write_weight(w: &mut impl Write, weight: &Weight) -> io::Result<()> {
    write_u64(w, weight.k() as u64)?;
    write_u64(w, weight.n() as u64)?;
    let fmt = match weight {
        Weight::F16(_) => 0u8,
        Weight::Q4(_) => 1u8,
    };
    write_u8(w, fmt)?;
    match weight {
        Weight::F16(fw) => write_f16_vec(w, &fw.data)?,
        Weight::Q4(qw) => {
            write_bytes(w, &qw.packed)?;
            write_f16_vec(w, &qw.scales)?;
        }
    }
    Ok(())
}

fn read_weight(r: &mut impl Read, _format_id: u8) -> io::Result<Weight> {
    let k = read_u64(r)? as usize;
    let n = read_u64(r)? as usize;
    let per_weight_fmt = read_u8(r)?;
    match per_weight_fmt {
        0 => {
            let data = read_f16_vec(r)?;
            Ok(Weight::F16(F16Weight { data, k, n }))
        }
        1 => {
            let packed = read_bytes_vec(r)?;
            let scales = read_f16_vec(r)?;
            Ok(Weight::Q4(Q4Weight {
                packed,
                scales,
                k,
                n,
            }))
        }
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Unknown weight format {per_weight_fmt}"),
        )),
    }
}

// ============================================================
// Save
// ============================================================

pub fn save_model(weights: &ModelWeights, path: &Path) -> io::Result<()> {
    let mut w = BufWriter::with_capacity(8 * 1024 * 1024, File::create(path)?);
    let config = &weights.config;
    let format_id = weights.format_id();

    // Header
    w.write_all(MAGIC)?;
    write_u32(&mut w, VERSION)?;
    write_u8(&mut w, format_id)?;

    // Config
    write_u32(&mut w, config.vocab_size as u32)?;
    write_u32(&mut w, config.hidden_size as u32)?;
    write_u32(&mut w, config.num_layers as u32)?;
    write_u32(&mut w, config.num_attn_heads as u32)?;
    write_u32(&mut w, config.num_kv_heads as u32)?;
    write_u32(&mut w, config.attn_head_dim as u32)?;
    write_u32(&mut w, config.num_qk_heads as u32)?;
    write_u32(&mut w, config.num_v_heads as u32)?;
    write_u32(&mut w, config.deltanet_head_dim as u32)?;
    write_u32(&mut w, config.conv_kernel_size as u32)?;
    write_u32(&mut w, config.intermediate_size as u32)?;
    write_f64(&mut w, config.rope_theta)?;
    write_f32(&mut w, config.partial_rotary_factor)?;
    write_f64(&mut w, config.rms_norm_eps)?;
    write_u32(&mut w, config.eos_token_id)?;
    write_u8(&mut w, config.tie_word_embeddings as u8)?;
    // New files are strictly text-only; legacy vision metadata remains readable
    write_u8(&mut w, 0)?;

    // Layer types
    for lt in &config.layer_types {
        write_u8(&mut w, *lt as u8)?;
    }

    // Per-layer weights
    for (i, layer) in weights.layers.iter().enumerate() {
        match layer {
            HybridLayerWeights::DeltaNet(lw) => {
                write_weight(&mut w, &lw.in_proj_qkv)?;
                write_weight(&mut w, &lw.in_proj_a)?;
                write_weight(&mut w, &lw.in_proj_b)?;
                write_weight(&mut w, &lw.in_proj_z)?;
                write_weight(&mut w, &lw.out_proj)?;
                write_f32_vec(&mut w, &lw.conv1d_weight)?;
                write_f32_vec(&mut w, &lw.a_log)?;
                write_f32_vec(&mut w, &lw.dt_bias)?;
                write_f16_vec(&mut w, &lw.attn_norm_weight)?;
                write_weight(&mut w, &lw.gate_proj)?;
                write_weight(&mut w, &lw.up_proj)?;
                write_weight(&mut w, &lw.down_proj)?;
                write_f16_vec(&mut w, &lw.input_norm)?;
                write_f16_vec(&mut w, &lw.post_attn_norm)?;
            }
            HybridLayerWeights::FullAttn(lw) => {
                write_weight(&mut w, &lw.q_proj)?;
                write_weight(&mut w, &lw.k_proj)?;
                write_weight(&mut w, &lw.v_proj)?;
                write_weight(&mut w, &lw.o_proj)?;
                write_f16_vec(&mut w, &lw.q_norm)?;
                write_f16_vec(&mut w, &lw.k_norm)?;
                write_weight(&mut w, &lw.gate_proj)?;
                write_weight(&mut w, &lw.up_proj)?;
                write_weight(&mut w, &lw.down_proj)?;
                write_f16_vec(&mut w, &lw.input_norm)?;
                write_f16_vec(&mut w, &lw.post_attn_norm)?;
            }
        }
        if i % 6 == 0 {
            eprintln!("  Saved layer {i}/{}...", config.num_layers);
        }
    }

    // Global
    write_weight(&mut w, &weights.embed)?;
    write_f16_vec(&mut w, &weights.final_norm)?;
    write_f32_vec(&mut w, &weights.rope_cos)?;
    write_f32_vec(&mut w, &weights.rope_sin)?;

    w.flush()?;
    Ok(())
}

// ============================================================
// Load
// ============================================================

/// Skip one serialized vision vector without allocating its contents
///
/// Each vector starts with a u64 element count, followed by either
/// f32 (v1) or f16 (v2) data
fn skip_vision_vec(r: &mut BufReader<File>, element_size: u64) -> io::Result<()> {
    let len = read_u64(r)?;
    let bytes = len
        .checked_mul(element_size)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Vision vector size overflow"))?;

    let current = r.stream_position()?;
    let end = current
        .checked_add(bytes)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Vision offset overflow"))?;
    let file_len = r.get_ref().metadata()?.len();

    if end > file_len {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "Vision weights extend beyond end of model file",
        ));
    }

    r.seek(SeekFrom::Start(end))?;
    Ok(())
}

/// Skip the complete serialized vision encoder
///
/// Layout: 3 initial vectors, 12 vectors per vision block,
/// and 6 merger vectors
fn skip_vision_weights(
    r: &mut BufReader<File>,
    version: u32,
    vision_layers: usize,
) -> io::Result<()> {
    let element_size = if version == 1 { 4 } else { 2 };

    for _ in 0..3 {
        skip_vision_vec(r, element_size)?;
    }

    for _ in 0..vision_layers {
        for _ in 0..12 {
            skip_vision_vec(r, element_size)?;
        }
    }

    for _ in 0..6 {
        skip_vision_vec(r, element_size)?;
    }

    Ok(())
}

/// Load text-model weights. Legacy image/video sections in v1/v2 files are
/// skipped without allocating or interpreting them
pub fn load_model_text(path: &Path) -> io::Result<ModelWeights> {
    load_model_text_with_verbosity(path, false)
}

/// Load the text model, optionally reporting loading progress to stderr
pub fn load_model_text_with_verbosity(path: &Path, verbose: bool) -> io::Result<ModelWeights> {
    let mut r = BufReader::with_capacity(8 * 1024 * 1024, File::open(path)?);

    // Header
    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Invalid magic (expected QOR4)",
        ));
    }
    let version = read_u32(&mut r)?;
    if version != 1 && version != VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Unsupported version {version} (expected 1 or {VERSION})"),
        ));
    }
    let format_id = read_u8(&mut r)?;

    // Config
    let vocab_size = read_u32(&mut r)? as usize;
    let hidden_size = read_u32(&mut r)? as usize;
    let num_layers = read_u32(&mut r)? as usize;
    let num_attn_heads = read_u32(&mut r)? as usize;
    let num_kv_heads = read_u32(&mut r)? as usize;
    let attn_head_dim = read_u32(&mut r)? as usize;
    let num_qk_heads = read_u32(&mut r)? as usize;
    let num_v_heads = read_u32(&mut r)? as usize;
    let deltanet_head_dim = read_u32(&mut r)? as usize;
    let conv_kernel_size = read_u32(&mut r)? as usize;
    let intermediate_size = read_u32(&mut r)? as usize;
    let rope_theta = read_f64(&mut r)?;
    let partial_rotary_factor = read_f32(&mut r)?;
    let rms_norm_eps = read_f64(&mut r)?;
    let eos_token_id = read_u32(&mut r)?;
    let tie_word_embeddings = read_u8(&mut r)? != 0;
    let has_vision = read_u8(&mut r)? != 0;

    // Layer types
    let mut layer_types = Vec::with_capacity(num_layers);
    for _ in 0..num_layers {
        let lt = read_u8(&mut r)?;
        layer_types.push(if lt == 0 {
            LayerType::DeltaNet
        } else {
            LayerType::FullAttn
        });
    }

    let config = Qor4bConfig {
        vocab_size,
        hidden_size,
        num_layers,
        num_attn_heads,
        num_kv_heads,
        attn_head_dim,
        num_qk_heads,
        num_v_heads,
        deltanet_head_dim,
        conv_kernel_size,
        intermediate_size,
        rope_theta,
        partial_rotary_factor,
        rms_norm_eps,
        eos_token_id,
        tie_word_embeddings,
        layer_types: layer_types.clone(),
    };

    // Per-layer weights
    let mut layers = Vec::with_capacity(num_layers);
    for (i, lt) in layer_types.iter().enumerate() {
        if verbose && i % 6 == 0 {
            eprintln!("  Loading layer {i}/{num_layers}...");
        }
        match lt {
            LayerType::DeltaNet => {
                layers.push(HybridLayerWeights::DeltaNet(DeltaNetLayerWeights {
                    in_proj_qkv: read_weight(&mut r, format_id)?,
                    in_proj_a: read_weight(&mut r, format_id)?,
                    in_proj_b: read_weight(&mut r, format_id)?,
                    in_proj_z: read_weight(&mut r, format_id)?,
                    out_proj: read_weight(&mut r, format_id)?,
                    conv1d_weight: read_f32_vec(&mut r)?,
                    a_log: read_f32_vec(&mut r)?,
                    dt_bias: read_f32_vec(&mut r)?,
                    attn_norm_weight: read_f16_vec(&mut r)?,
                    gate_proj: read_weight(&mut r, format_id)?,
                    up_proj: read_weight(&mut r, format_id)?,
                    down_proj: read_weight(&mut r, format_id)?,
                    input_norm: read_f16_vec(&mut r)?,
                    post_attn_norm: read_f16_vec(&mut r)?,
                }));
            }
            LayerType::FullAttn => {
                layers.push(HybridLayerWeights::FullAttn(FullAttnLayerWeights {
                    q_proj: read_weight(&mut r, format_id)?,
                    k_proj: read_weight(&mut r, format_id)?,
                    v_proj: read_weight(&mut r, format_id)?,
                    o_proj: read_weight(&mut r, format_id)?,
                    q_norm: read_f16_vec(&mut r)?,
                    k_norm: read_f16_vec(&mut r)?,
                    gate_proj: read_weight(&mut r, format_id)?,
                    up_proj: read_weight(&mut r, format_id)?,
                    down_proj: read_weight(&mut r, format_id)?,
                    input_norm: read_f16_vec(&mut r)?,
                    post_attn_norm: read_f16_vec(&mut r)?,
                }));
            }
        }
    }

    // Global
    let embed = read_weight(&mut r, format_id)?;
    let final_norm = read_f16_vec(&mut r)?;
    let rope_cos = read_f32_vec(&mut r)?;
    let rope_sin = read_f32_vec(&mut r)?;

    // Strictly text-only runtime: preserve backward compatibility by skipping
    // the legacy serialized vision section using its original v1/v2 layout
    if has_vision {
        if verbose {
            eprintln!("  Skipping legacy vision section (text-only runtime)...");
        }
        skip_vision_weights(&mut r, version, LEGACY_VISION_LAYERS)?;
    }

    let format_name = match format_id {
        0 => "F16",
        1 => "Q4",
        _ => "unknown",
    };
    if verbose {
        eprintln!(
            "  Loaded {format_name} model: {num_layers} layers, vocab={vocab_size}, hidden={hidden_size}"
        );
    }

    Ok(ModelWeights {
        layers,
        embed,
        vocab: vocab_size,
        hidden: hidden_size,
        final_norm,
        rope_cos,
        rope_sin,
        config,
        format_name,
    })
}
