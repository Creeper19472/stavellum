struct Frame { size: vec2<f32> }
var<push_constant> frame: Frame;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex: u32,
           @location(0) rect: vec4<f32>,
           @location(1) uv_rect: vec4<f32>,
           @location(2) color: vec4<f32>) -> VertexOutput {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0));
    let corner = corners[vertex];
    let pixel = rect.xy + corner * rect.zw;
    var output: VertexOutput;
    // Pure Vulkan NDC (y grows downward with pixel rows). build.rs disables
    // naga's automatic coordinate-space adjustment so this mapping is exact.
    output.position = vec4<f32>(pixel.x / frame.size.x * 2.0 - 1.0,
                                pixel.y / frame.size.y * 2.0 - 1.0, 0.0, 1.0);
    output.uv = mix(uv_rect.xy, uv_rect.zw, corner);
    output.color = color;
    return output;
}

@group(0) @binding(0) var image: texture_2d<f32>;
@group(0) @binding(1) var image_sampler: sampler;

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return textureSample(image, image_sampler, input.uv) * input.color;
}
