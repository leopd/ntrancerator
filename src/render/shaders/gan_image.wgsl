// Full-screen image display shader for the GAN output.
// Samples an Rgba8Unorm texture and displays it stretched to fill the window.

@group(0) @binding(0) var img_tex  : texture_2d<f32>;
@group(0) @binding(1) var img_samp : sampler;

struct VsOut {
    @builtin(position) pos : vec4<f32>,
    @location(0) uv : vec2<f32>,
};

// Full-screen triangle; no vertex buffer needed.
@vertex
fn vs_main(@builtin(vertex_index) vi : u32) -> VsOut {
    var out : VsOut;
    let x = f32((vi << 1u) & 2u);
    let y = f32(vi & 2u);
    out.uv = vec2<f32>(x, y);
    out.pos = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    return out;
}

@fragment
fn fs_main(in : VsOut) -> @location(0) vec4<f32> {
    return textureSample(img_tex, img_samp, in.uv);
}
