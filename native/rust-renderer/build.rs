//! Compile the WGSL quad shader to two SPIR-V entry points with naga.
use std::{env, fs, path::PathBuf};

fn main() {
    let out = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let source = fs::read_to_string("src/shaders/quad.wgsl")
        .unwrap_or_else(|error| panic!("read quad.wgsl: {error}"));
    let module = naga::front::wgsl::parse_str(&source)
        .unwrap_or_else(|errors| panic!("WGSL parse errors: {errors}"));
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap_or_else(|errors| panic!("validate quad.wgsl: {errors:?}"));
    let mut options = naga::back::spv::Options::default();
    // The quad shader maps pixel rows onto Vulkan's y-down NDC itself;
    // naga's automatic y-flip epilogue must not be applied on top of it.
    options.flags -= naga::back::spv::WriterFlags::ADJUST_COORDINATE_SPACE;
    for (stage, entry) in [
        (naga::ShaderStage::Vertex, "vs_main"),
        (naga::ShaderStage::Fragment, "fs_main"),
    ] {
        let words = naga::back::spv::write_vec(
            &module,
            &info,
            &options,
            Some(&naga::back::spv::PipelineOptions {
                shader_stage: stage,
                entry_point: entry.to_owned(),
            }),
        )
        .unwrap_or_else(|error| panic!("write {entry} SPIR-V: {error:?}"));
        fs::write(
            out.join(format!("quad.{entry}.words")),
            format!("{words:?}"),
        )
        .unwrap_or_else(|error| panic!("write words file: {error}"));
    }
    println!("cargo:rerun-if-changed=src/shaders/quad.wgsl");
}
