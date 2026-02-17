use anyhow::{bail, Context, Result};
use burn::backend::autodiff::Autodiff;
use burn::module::Module;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn_import::onnx::ModelGen;
use burn_ndarray::{NdArray, NdArrayDevice};
use burn_store::{BurnpackStore, ModuleSnapshot};
use clap::Parser;
use std::path::PathBuf;

use rust_rl::common::model::models::Agent;

#[derive(Debug, Parser)]
#[command(name = "convert_onnx")]
#[command(about = "Offline ONNX -> Burn source/state artifact generation")]
struct Args {
    /// Path to the source ONNX file.
    #[arg(long)]
    onnx_path: PathBuf,

    /// Output directory where generated Burn artifacts will be written.
    #[arg(long, default_value = "generated/onnx")]
    out_dir: PathBuf,

    /// Emit additional debug graph dumps for inspection.
    #[arg(long, default_value_t = false)]
    development: bool,

    /// Optional path for exporting a model-only native checkpoint (.mpk).
    ///
    /// This performs a best-effort tensor-name mapping by loading generated .bpk
    /// snapshots into a typed rust_ppo Agent.
    #[arg(long)]
    to_mpk: Option<PathBuf>,

    /// Agent observation dimension used when --to-mpk is set.
    #[arg(long)]
    obs_dim: Option<usize>,

    /// Agent hidden dimension used when --to-mpk is set.
    #[arg(long)]
    hidden_dim: Option<usize>,

    /// Agent action dimension used when --to-mpk is set.
    #[arg(long)]
    action_dim: Option<usize>,

    /// Whether to initialize BinPack architecture when --to-mpk is set.
    #[arg(long, default_value_t = false)]
    use_binpack_architecture: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let extension = args
        .onnx_path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_ascii_lowercase())
        .unwrap_or_default();

    if extension != "onnx" {
        bail!(
            "--onnx-path must point to a .onnx file, got '{}'",
            args.onnx_path.display()
        );
    }

    if !args.onnx_path.exists() {
        bail!("onnx file '{}' does not exist", args.onnx_path.display());
    }

    std::fs::create_dir_all(&args.out_dir).with_context(|| {
        format!(
            "failed to create output directory '{}'",
            args.out_dir.display()
        )
    })?;

    let onnx_path = args
        .onnx_path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("onnx path contains invalid UTF-8"))?;
    let out_dir = args
        .out_dir
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("out-dir path contains invalid UTF-8"))?;

    let mut model_gen = ModelGen::new();
    model_gen
        .input(onnx_path)
        .out_dir(out_dir)
        .development(args.development)
        .run_from_cli();

    let model_stem = args
        .onnx_path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .ok_or_else(|| anyhow::anyhow!("failed to infer ONNX model stem"))?;
    let generated_bpk = args.out_dir.join(format!("{model_stem}.bpk"));

    println!(
        "Generated Burn ONNX artifacts in '{}'. This produces model source (.rs) and state (.bpk).",
        args.out_dir.display()
    );

    if let Some(to_mpk_path) = args.to_mpk.as_ref() {
        let obs_dim = args
            .obs_dim
            .ok_or_else(|| anyhow::anyhow!("--obs-dim is required with --to-mpk"))?;
        let hidden_dim = args
            .hidden_dim
            .ok_or_else(|| anyhow::anyhow!("--hidden-dim is required with --to-mpk"))?;
        let action_dim = args
            .action_dim
            .ok_or_else(|| anyhow::anyhow!("--action-dim is required with --to-mpk"))?;

        if !generated_bpk.exists() {
            bail!(
                "expected generated burnpack file '{}' was not found",
                generated_bpk.display()
            );
        }

        let backend_device = NdArrayDevice::Cpu;
        type B = Autodiff<NdArray<f32>>;
        let mut agent = Agent::<B>::new(
            obs_dim,
            hidden_dim,
            action_dim,
            args.use_binpack_architecture,
            &backend_device,
        );

        let mut store = BurnpackStore::from_file(&generated_bpk)
            .allow_partial(true)
            .validate(true);
        let apply_result = agent
            .load_from(&mut store)
            .with_context(|| format!("failed to apply burnpack snapshots from '{}'", generated_bpk.display()))?;

        let mpk_base = strip_mpk_extension(to_mpk_path);
        if let Some(parent) = mpk_base.parent() {
            std::fs::create_dir_all(parent).with_context(|| {
                format!("failed to create output directory '{}'", parent.display())
            })?;
        }

        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();
        agent
            .save_file(mpk_base, &recorder)
            .context("failed to export native .mpk model checkpoint")?;

        println!(
            "Exported model-only .mpk checkpoint to '{}'.",
            to_mpk_path.display()
        );
        println!(
            "Burnpack apply summary: applied={}, missing={}, skipped={}, unused={}, errors={}",
            apply_result.applied.len(),
            apply_result.missing.len(),
            apply_result.skipped.len(),
            apply_result.unused.len(),
            apply_result.errors.len()
        );
        if !apply_result.errors.is_empty() {
            println!(
                "Warning: some tensors failed to map. Check apply summary before using this checkpoint."
            );
        }
    }

    Ok(())
}

fn strip_mpk_extension(path: &std::path::Path) -> PathBuf {
    let is_mpk = path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("mpk"))
        .unwrap_or(false);

    if is_mpk {
        path.with_extension("")
    } else {
        path.to_path_buf()
    }
}
