// Spectrogram fragment shader (spec §9).
//
// Maps screen UV -> (time, log-frequency), samples the R32Float history texture
// (with linear interpolation across frequency bins), normalizes dB to [0,1]
// against the floor/ceiling window, and colors via a 256x1 LUT texture.

struct Params {
    cursor_offset : f32,  // history write cursor / width, in [0,1)
    width         : f32,  // history columns
    num_bins      : f32,  // fft_size/2 + 1
    fft_size      : f32,
    sample_rate   : f32,
    db_floor      : f32,
    db_ceiling    : f32,
    freq_min      : f32,
    freq_max      : f32,
    _pad0         : f32,
    _pad1         : f32,
    _pad2         : f32,
};

@group(0) @binding(0) var history : texture_2d<f32>;   // R32Float dB values
@group(0) @binding(1) var lut      : texture_2d<f32>;  // Rgba8Unorm colormap
@group(0) @binding(2) var lut_samp : sampler;
@group(0) @binding(3) var<uniform> p : Params;

struct VsOut {
    @builtin(position) pos : vec4<f32>,
    @location(0) uv : vec2<f32>,
};

// Full-screen triangle; no vertex buffer needed.
@vertex
fn vs_main(@builtin(vertex_index) vi : u32) -> VsOut {
    var out : VsOut;
    // (0,0),(2,0),(0,2) in UV -> covers the [0,1] quad.
    let x = f32((vi << 1u) & 2u);
    let y = f32(vi & 2u);
    out.uv = vec2<f32>(x, y);
    // Clip space: map UV x [0,2] -> [-1,3]. For y, uv.y=0 is the TOP of the
    // screen (NDC +1); the fragment shader inverts row_norm so bass ends up at
    // the bottom.
    out.pos = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    return out;
}

fn db_at(col : i32, bin : i32) -> f32 {
    let b = clamp(bin, 0, i32(p.num_bins) - 1);
    return textureLoad(history, vec2<i32>(col, b), 0).x;
}

@fragment
fn fs_main(in : VsOut) -> @location(0) vec4<f32> {
    // --- time axis (horizontal): newest column at the right edge ---
    // uv.x=0 -> oldest (the column about to be overwritten, = cursor),
    // uv.x=1 -> newest (cursor-1, mod width).
    let w = p.width;
    var col_f = p.cursor_offset * w + in.uv.x * (w - 1.0);
    col_f = col_f - floor(col_f / w) * w; // modulo width
    let col = i32(col_f);

    // --- frequency axis (vertical, logarithmic): bass at the bottom ---
    // uv.y=0 is the top of the screen (see vs_main), so invert it: the bottom
    // row maps to row_norm=0 -> freq_min (bass), the top to freq_max (treble).
    let row_norm = clamp(1.0 - in.uv.y, 0.0, 1.0);
    let freq = p.freq_min * pow(p.freq_max / p.freq_min, row_norm);
    let bin_f = freq * p.fft_size / p.sample_rate;

    // Linear interpolation between adjacent bins.
    let b0 = i32(floor(bin_f));
    let frac = bin_f - floor(bin_f);
    let db = mix(db_at(col, b0), db_at(col, b0 + 1), frac);

    // --- normalize + colormap ---
    let span = max(p.db_ceiling - p.db_floor, 1e-6);
    let t = clamp((db - p.db_floor) / span, 0.0, 1.0);
    let rgb = textureSample(lut, lut_samp, vec2<f32>(t, 0.5)).rgb;
    return vec4<f32>(rgb, 1.0);
}
