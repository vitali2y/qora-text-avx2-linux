//! Runtime-dispatched AVX2 CPU kernels for QORA text inference
//! Scalar reference implementations remain in gemv.rs

#![allow(unsafe_op_in_unsafe_fn)]

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use half::f16;

const Q4_GROUP_SIZE: usize = 32;

// ============================================================
// AVX2 + FMA kernels (with F16C where needed)
// ============================================================

/// Runtime check for the instruction set used by the AVX2 kernels
/// FMA is checked separately because AVX2 does not imply FMA
pub fn has_avx2_fma() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Runtime check for AVX2 + FMA + F16C, required by the half-precision kernel
pub fn has_avx2_f16c_fma() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("f16c")
            && is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// AVX2 Q4 GEMV. Q4 values are unpacked into eight-element lanes, converted
/// to f32, scaled, and accumulated with FMA. SIMD unpacking preserves the
/// existing on-disk low-nibble/high-nibble ordering exactly
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn gemv_q4_avx2(
    input: &[f32],
    packed: &[u8],
    scales: &[f16],
    n: usize,
    k_start: usize,
    k_end: usize,
) -> Vec<f32> {
    let groups_per_row = n / Q4_GROUP_SIZE;
    let packed_per_row = groups_per_row * (Q4_GROUP_SIZE / 2);
    let mut output = vec![0.0f32; n];

    let nibble_mask = _mm256_set1_epi32(0x0f);
    let offset = _mm256_set1_epi32(8);
    for ki in k_start..k_end {
        let x = input[ki];
        if x == 0.0 {
            continue;
        }
        let scale_base = ki * groups_per_row;
        let packed_base = ki * packed_per_row;
        for group in 0..groups_per_row {
            let scale = scales[scale_base + group].to_f32() * x;
            if scale == 0.0 {
                continue;
            }
            let byte_base = packed_base + group * 16;
            let out_base = group * Q4_GROUP_SIZE;
            let scale_vec = _mm256_set1_ps(scale);

            // Each 8-byte load contains eight packed bytes -> sixteen Q4 values
            // Unpack low/high nibbles and interleave them into the original order
            for half_group in 0..2 {
                let bytes = _mm_loadl_epi64(
                    packed.as_ptr().add(byte_base + half_group * 8) as *const __m128i
                );
                let values = _mm256_cvtepu8_epi32(bytes);
                let lo = _mm256_sub_epi32(_mm256_and_si256(values, nibble_mask), offset);
                let hi = _mm256_sub_epi32(_mm256_srli_epi32(values, 4), offset);
                let lo_f = _mm256_cvtepi32_ps(lo);
                let hi_f = _mm256_cvtepi32_ps(hi);

                let interleave_lo = _mm256_unpacklo_ps(lo_f, hi_f);
                let interleave_hi = _mm256_unpackhi_ps(lo_f, hi_f);
                let first = _mm256_permute2f128_ps(interleave_lo, interleave_hi, 0x20);
                let second = _mm256_permute2f128_ps(interleave_lo, interleave_hi, 0x31);

                let out = out_base + half_group * 16;
                let acc0 = _mm256_loadu_ps(output.as_ptr().add(out));
                let acc1 = _mm256_loadu_ps(output.as_ptr().add(out + 8));
                _mm256_storeu_ps(
                    output.as_mut_ptr().add(out),
                    _mm256_fmadd_ps(first, scale_vec, acc0),
                );
                _mm256_storeu_ps(
                    output.as_mut_ptr().add(out + 8),
                    _mm256_fmadd_ps(second, scale_vec, acc1),
                );
            }
        }
    }
    output
}

/// AVX2/F16C/FMA F16 GEMV: output = input[k] @ weight[k,n]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c,fma")]
pub unsafe fn gemv_f16_avx2(input: &[f32], weight_data: &[f16], k: usize, n: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; n];
    let n8 = n / 8 * 8;
    for ki in 0..k {
        let x = input[ki];
        if x == 0.0 {
            continue;
        }
        let row = ki * n;
        let xv = _mm256_set1_ps(x);
        let mut j = 0;
        while j < n8 {
            let h = _mm_loadu_si128(weight_data.as_ptr().add(row + j) as *const __m128i);
            let w = _mm256_cvtph_ps(h);
            let acc = _mm256_loadu_ps(output.as_ptr().add(j));
            _mm256_storeu_ps(output.as_mut_ptr().add(j), _mm256_fmadd_ps(xv, w, acc));
            j += 8;
        }
        while j < n {
            output[j] += x * weight_data[row + j].to_f32();
            j += 1;
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn avx2_q4_matches_scalar_reference() {
        if !has_avx2_fma() {
            return;
        }
        let k = 3usize;
        let n = 64usize;
        let mut packed = vec![0u8; k * n / 2];
        for (i, byte) in packed.iter_mut().enumerate() {
            *byte = ((i * 7 + 3) as u8 & 0x0f) | ((((i * 11 + 5) as u8) & 0x0f) << 4);
        }
        let scales: Vec<f16> = (0..k * (n / 32))
            .map(|i| f16::from_f32(0.125 + i as f32 * 0.0625))
            .collect();
        let input = [0.75f32, -1.25, 0.5];
        let mut expected = vec![0.0f32; n];
        for row in 0..k {
            for group in 0..(n / 32) {
                let scale = scales[row * (n / 32) + group].to_f32() * input[row];
                for byte_index in 0..16 {
                    let byte = packed[row * (n / 2) + group * 16 + byte_index];
                    let out = group * 32 + byte_index * 2;
                    expected[out] += ((byte & 0x0f) as i32 - 8) as f32 * scale;
                    expected[out + 1] += ((byte >> 4) as i32 - 8) as f32 * scale;
                }
            }
        }
        let actual = unsafe { gemv_q4_avx2(&input, &packed, &scales, n, 0, k) };
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-5, "actual={a}, expected={e}");
        }
    }

    #[test]
    fn avx2_f16_matches_scalar_reference() {
        if !has_avx2_f16c_fma() {
            return;
        }
        let k = 4usize;
        let n = 13usize;
        let weights: Vec<f16> = (0..k * n)
            .map(|i| f16::from_f32((i as f32 - 17.0) * 0.03125))
            .collect();
        let input = [0.25f32, -0.5, 1.25, 0.75];
        let mut expected = vec![0.0f32; n];
        for row in 0..k {
            for col in 0..n {
                expected[col] += input[row] * weights[row * n + col].to_f32();
            }
        }
        let actual = unsafe { gemv_f16_avx2(&input, &weights, k, n) };
        for (a, e) in actual.iter().zip(expected.iter()) {
            assert!((a - e).abs() < 1e-5, "actual={a}, expected={e}");
        }
    }
}
