use numpy::{IntoPyArray, PyArray2, PyArray3, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Fast parallel Chroma Key matte calculation in Rust (Rayon).
/// Takes RGB uint8 image (H, W, 3) and returns float32 alpha matte (H, W) in [0.0, 1.0].
#[pyfunction]
#[pyo3(signature = (rgb_u8, screen_type="green", screen_color=(0.0, 177.0, 64.0), strength=1.0, clip_black=0.08, clip_white=0.92))]
fn chroma_key_matte_rust<'py>(
    py: Python<'py>,
    rgb_u8: PyReadonlyArray3<'py, u8>,
    screen_type: &str,
    screen_color: (f32, f32, f32),
    strength: f32,
    clip_black: f32,
    clip_white: f32,
) -> Bound<'py, PyArray2<f32>> {
    let view = rgb_u8.as_array();
    let shape = view.shape();
    let h = shape[0];
    let w = shape[1];
    let raw = view.as_slice();

    let s = strength.clamp(0.05, 5.0);
    let cb = clip_black.clamp(0.0, 0.98);
    let cw = clip_white.clamp(cb + 0.01, 1.0);
    let inv_range = 1.0 / (cw - cb);

    let mode = match screen_type.to_ascii_lowercase().as_str() {
        "blue" => 1u8,
        "custom" => 2u8,
        _ => 0u8,
    };

    let kr = (screen_color.0 / 255.0).clamp(0.0, 1.0);
    let kg = (screen_color.1 / 255.0).clamp(0.0, 1.0);
    let kb = (screen_color.2 / 255.0).clamp(0.0, 1.0);

    let mut out = vec![0.0f32; h * w];

    py.allow_threads(|| {
        if let Some(slice) = raw {
            out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
                let row_offset = y * w * 3;
                for x in 0..w {
                    let idx = row_offset + x * 3;
                    let r = (slice[idx] as f32) * (1.0 / 255.0);
                    let g = (slice[idx + 1] as f32) * (1.0 / 255.0);
                    let b = (slice[idx + 2] as f32) * (1.0 / 255.0);

                    let raw_alpha = if mode == 0 {
                        // Green screen difference key
                        let max_rb = if r > b { r } else { b };
                        let diff = (g - max_rb) * (1.65 * s);
                        (1.0 - diff).clamp(0.0, 1.0)
                    } else if mode == 1 {
                        // Blue screen difference key
                        let max_rg = if r > g { r } else { g };
                        let diff = (b - max_rg) * (1.65 * s);
                        (1.0 - diff).clamp(0.0, 1.0)
                    } else {
                        // Custom color Euclidean distance in RGB
                        let dr = r - kr;
                        let dg = g - kg;
                        let db = b - kb;
                        let dist = (dr * dr + dg * dg + db * db).sqrt();
                        (dist * (2.2 / s)).clamp(0.0, 1.0)
                    };

                    row[x] = ((raw_alpha - cb) * inv_range).clamp(0.0, 1.0);
                }
            });
        } else {
            for y in 0..h {
                for x in 0..w {
                    let r = (view[[y, x, 0]] as f32) * (1.0 / 255.0);
                    let g = (view[[y, x, 1]] as f32) * (1.0 / 255.0);
                    let b = (view[[y, x, 2]] as f32) * (1.0 / 255.0);
                    let max_rb = if r > b { r } else { b };
                    let raw_alpha = (1.0 - (g - max_rb) * (1.65 * s)).clamp(0.0, 1.0);
                    out[y * w + x] = ((raw_alpha - cb) * inv_range).clamp(0.0, 1.0);
                }
            }
        }
    });

    let arr = numpy::ndarray::Array2::from_shape_vec((h, w), out).unwrap();
    arr.into_pyarray_bound(py)
}

/// Fast parallel despill + checkerboard/backdrop compositing in Rust (Rayon).
/// Takes FG float32 [H, W, 3] in [0, 1], Alpha float32 [H, W] in [0, 1], returns uint8 RGB [H, W, 3].
#[pyfunction]
#[pyo3(signature = (fg_f32, alpha_f32, despill_strength=1.0, screen_color="green", checker_size=16))]
fn despill_and_composite_rust<'py>(
    py: Python<'py>,
    fg_f32: PyReadonlyArray3<'py, f32>,
    alpha_f32: PyReadonlyArray2<'py, f32>,
    despill_strength: f32,
    screen_color: &str,
    checker_size: usize,
) -> Bound<'py, PyArray3<u8>> {
    let fg_view = fg_f32.as_array();
    let a_view = alpha_f32.as_array();
    let h = fg_view.shape()[0];
    let w = fg_view.shape()[1];
    let fg_slice = fg_view.as_slice();
    let a_slice = a_view.as_slice();

    let is_blue = screen_color.eq_ignore_ascii_case("blue");
    let ds = despill_strength.clamp(0.0, 1.0);
    let c_size = checker_size.max(4);

    let mut out = vec![0u8; h * w * 3];

    py.allow_threads(|| {
        if let (Some(f_raw), Some(a_raw)) = (fg_slice, a_slice) {
            out.par_chunks_mut(w * 3).enumerate().for_each(|(y, row)| {
                let cy = y / c_size;
                let f_off = y * w * 3;
                let a_off = y * w;
                for x in 0..w {
                    let cx = x / c_size;
                    let bg_val = if (cx + cy) % 2 == 0 { 0.18f32 } else { 0.28f32 };

                    let idx3 = f_off + x * 3;
                    let mut r = f_raw[idx3].clamp(0.0, 1.0);
                    let mut g = f_raw[idx3 + 1].clamp(0.0, 1.0);
                    let mut b = f_raw[idx3 + 2].clamp(0.0, 1.0);
                    let a = a_raw[a_off + x].clamp(0.0, 1.0);

                    if ds > 0.0 {
                        if is_blue {
                            let limit = 0.5 * (r + g);
                            if b > limit {
                                let spill = (b - limit) * ds;
                                b -= spill;
                                r = (r + spill * 0.25).clamp(0.0, 1.0);
                                g = (g + spill * 0.25).clamp(0.0, 1.0);
                            }
                        } else {
                            let limit = 0.5 * (r + b);
                            if g > limit {
                                let spill = (g - limit) * ds;
                                g -= spill;
                                r = (r + spill * 0.30).clamp(0.0, 1.0);
                                b = (b + spill * 0.15).clamp(0.0, 1.0);
                            }
                        }
                    }

                    let comp_r = r * a + bg_val * (1.0 - a);
                    let comp_g = g * a + bg_val * (1.0 - a);
                    let comp_b = b * a + bg_val * (1.0 - a);

                    let out_idx = x * 3;
                    row[out_idx] = (comp_r * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
                    row[out_idx + 1] = (comp_g * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
                    row[out_idx + 2] = (comp_b * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
                }
            });
        }
    });

    let arr = numpy::ndarray::Array3::from_shape_vec((h, w, 3), out).unwrap();
    arr.into_pyarray_bound(py)
}

/// Fast bilinear resize of uint8 RGB image (H, W, 3) for responsive zero-lag viewport rendering.
#[pyfunction]
fn resize_rgb_u8_rust<'py>(
    py: Python<'py>,
    rgb_u8: PyReadonlyArray3<'py, u8>,
    target_w: usize,
    target_h: usize,
) -> Bound<'py, PyArray3<u8>> {
    let view = rgb_u8.as_array();
    let src_h = view.shape()[0];
    let src_w = view.shape()[1];
    let dst_w = target_w.max(1);
    let dst_h = target_h.max(1);
    let raw = view.as_slice();

    let mut out = vec![0u8; dst_h * dst_w * 3];

    py.allow_threads(|| {
        if let Some(slice) = raw {
            let scale_y = (src_h as f32) / (dst_h as f32);
            let scale_x = (src_w as f32) / (dst_w as f32);
            out.par_chunks_mut(dst_w * 3).enumerate().for_each(|(dy, row)| {
                let sy = ((dy as f32 + 0.5) * scale_y - 0.5).clamp(0.0, (src_h - 1) as f32);
                let y0 = sy.floor() as usize;
                let y1 = (y0 + 1).min(src_h - 1);
                let wy = sy - (y0 as f32);

                for dx in 0..dst_w {
                    let sx = ((dx as f32 + 0.5) * scale_x - 0.5).clamp(0.0, (src_w - 1) as f32);
                    let x0 = sx.floor() as usize;
                    let x1 = (x0 + 1).min(src_w - 1);
                    let wx = sx - (x0 as f32);

                    let i00 = (y0 * src_w + x0) * 3;
                    let i01 = (y0 * src_w + x1) * 3;
                    let i10 = (y1 * src_w + x0) * 3;
                    let i11 = (y1 * src_w + x1) * 3;

                    let w00 = (1.0 - wy) * (1.0 - wx);
                    let w01 = (1.0 - wy) * wx;
                    let w10 = wy * (1.0 - wx);
                    let w11 = wy * wx;

                    let out_idx = dx * 3;
                    for c in 0..3 {
                        let val = (slice[i00 + c] as f32) * w00
                            + (slice[i01 + c] as f32) * w01
                            + (slice[i10 + c] as f32) * w10
                            + (slice[i11 + c] as f32) * w11;
                        row[out_idx + c] = (val + 0.5).clamp(0.0, 255.0) as u8;
                    }
                }
            });
        }
    });

    let arr = numpy::ndarray::Array3::from_shape_vec((dst_h, dst_w, 3), out).unwrap();
    arr.into_pyarray_bound(py)
}

#[pymodule]
fn ck2_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(chroma_key_matte_rust, m)?)?;
    m.add_function(wrap_pyfunction!(despill_and_composite_rust, m)?)?;
    m.add_function(wrap_pyfunction!(resize_rgb_u8_rust, m)?)?;
    Ok(())
}
