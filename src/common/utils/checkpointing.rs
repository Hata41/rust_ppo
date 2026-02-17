use anyhow::{bail, Context, Result};
use burn::module::Module;
use burn::optim::Optimizer;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
use burn::tensor::backend::AutodiffBackend;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use crate::algorithms::spo::loss::MpoDuals;
use crate::common::config::Args;
use crate::common::model::models::{Actor, Agent, Critic};

const STEP_PREFIX: &str = "step_";
const AGENT_FILE: &str = "agent";
const AGENT_ONLINE_FILE: &str = "agent_online";
const AGENT_TARGET_FILE: &str = "agent_target";
const ACTOR_OPTIM_FILE: &str = "actor_optim";
const CRITIC_OPTIM_FILE: &str = "critic_optim";
const DUAL_FILE: &str = "duals";
const DUAL_OPTIM_FILE: &str = "dual_optim";
const METADATA_FILE: &str = "metadata.json";

#[derive(Clone, Debug)]
pub struct CheckpointConfig {
    pub save_dir: PathBuf,
    pub save_interval: usize,
    pub load_path: Option<PathBuf>,
    pub keep_last_n: Option<usize>,
}

impl CheckpointConfig {
    pub fn from_args(args: &Args) -> Self {
        Self {
            save_dir: args.checkpoint_save_dir.clone(),
            save_interval: args.checkpoint_save_interval,
            load_path: args.checkpoint_load_path.clone(),
            keep_last_n: args.checkpoint_keep_last_n,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Checkpointer {
    config: CheckpointConfig,
    run_dir: PathBuf,
    recorder: NamedMpkFileRecorder<FullPrecisionSettings>,
}

#[derive(Debug, Serialize, Deserialize)]
struct CheckpointMetadata {
    step: usize,
    created_at_utc: String,
    algorithm: String,
}

impl Checkpointer {
    pub fn new(config: CheckpointConfig, algorithm: &str) -> Result<Self> {
        let run_dir = Self::resolve_run_dir(&config, algorithm)?;
        fs::create_dir_all(&run_dir)
            .with_context(|| format!("failed to create checkpoint run dir {}", run_dir.display()))?;

        Ok(Self {
            config,
            run_dir,
            recorder: NamedMpkFileRecorder::<FullPrecisionSettings>::default(),
        })
    }

    pub fn should_save(&self, update: usize) -> bool {
        let interval = self.config.save_interval;
        interval > 0 && (update + 1) % interval == 0
    }

    pub fn save_ppo<B, OA, OC>(
        &self,
        step: usize,
        agent: &Agent<B>,
        actor_optim: &OA,
        critic_optim: &OC,
    ) -> Result<()>
    where
        B: AutodiffBackend,
        OA: Optimizer<Actor<B>, B>,
        OC: Optimizer<Critic<B>, B>,
    {
        let step_dir = self.step_dir(step);
        fs::create_dir_all(&step_dir)
            .with_context(|| format!("failed to create checkpoint dir {}", step_dir.display()))?;

        agent
            .clone()
            .save_file(step_dir.join(AGENT_FILE), &self.recorder)
            .context("failed to save PPO agent checkpoint")?;

        self.recorder
            .record(actor_optim.to_record(), step_dir.join(ACTOR_OPTIM_FILE))
            .context("failed to save PPO actor optimizer checkpoint")?;
        self.recorder
            .record(critic_optim.to_record(), step_dir.join(CRITIC_OPTIM_FILE))
            .context("failed to save PPO critic optimizer checkpoint")?;

        self.write_metadata(&step_dir, step, "ppo")?;
        self.rotate_if_needed()?;
        Ok(())
    }

    pub fn load_ppo<B, OA, OC>(
        &self,
        path: &Path,
        agent: &mut Agent<B>,
        actor_optim: &mut OA,
        critic_optim: &mut OC,
        device: &B::Device,
    ) -> Result<Option<usize>>
    where
        B: AutodiffBackend,
        OA: Optimizer<Actor<B>, B>,
        OC: Optimizer<Critic<B>, B>,
    {
        Self::validate_native_checkpoint_path(path)?;
        let load_layout = Self::resolve_load_layout(path, AGENT_FILE)?;

        *agent = agent
            .clone()
            .load_file(load_layout.model_base_path.clone(), &self.recorder, device)
            .context("failed to load PPO agent checkpoint")?;

        let actor_optim_path = load_layout.step_dir.join(ACTOR_OPTIM_FILE);
        let critic_optim_path = load_layout.step_dir.join(CRITIC_OPTIM_FILE);
        if Self::mpk_file_exists(&actor_optim_path) && Self::mpk_file_exists(&critic_optim_path) {
            let actor_record = self
                .recorder
                .load(actor_optim_path, device)
                .context("failed to load PPO actor optimizer checkpoint")?;
            let critic_record = self
                .recorder
                .load(critic_optim_path, device)
                .context("failed to load PPO critic optimizer checkpoint")?;
            *actor_optim = actor_optim.clone().load_record(actor_record);
            *critic_optim = critic_optim.clone().load_record(critic_record);
        }

        Ok(self.read_metadata_step(&load_layout.step_dir)?)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn save_spo<B, OA, OC, OD>(
        &self,
        step: usize,
        agent_online: &Agent<B>,
        agent_target: &Agent<B>,
        duals: &MpoDuals<B>,
        actor_optim: &OA,
        critic_optim: &OC,
        dual_optim: &OD,
    ) -> Result<()>
    where
        B: AutodiffBackend,
        OA: Optimizer<Actor<B>, B>,
        OC: Optimizer<Critic<B>, B>,
        OD: Optimizer<MpoDuals<B>, B>,
    {
        let step_dir = self.step_dir(step);
        fs::create_dir_all(&step_dir)
            .with_context(|| format!("failed to create checkpoint dir {}", step_dir.display()))?;

        agent_online
            .clone()
            .save_file(step_dir.join(AGENT_ONLINE_FILE), &self.recorder)
            .context("failed to save SPO online agent checkpoint")?;
        agent_target
            .clone()
            .save_file(step_dir.join(AGENT_TARGET_FILE), &self.recorder)
            .context("failed to save SPO target agent checkpoint")?;
        duals
            .clone()
            .save_file(step_dir.join(DUAL_FILE), &self.recorder)
            .context("failed to save SPO dual state checkpoint")?;

        self.recorder
            .record(actor_optim.to_record(), step_dir.join(ACTOR_OPTIM_FILE))
            .context("failed to save SPO actor optimizer checkpoint")?;
        self.recorder
            .record(critic_optim.to_record(), step_dir.join(CRITIC_OPTIM_FILE))
            .context("failed to save SPO critic optimizer checkpoint")?;
        self.recorder
            .record(dual_optim.to_record(), step_dir.join(DUAL_OPTIM_FILE))
            .context("failed to save SPO dual optimizer checkpoint")?;

        self.write_metadata(&step_dir, step, "spo")?;
        self.rotate_if_needed()?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn load_spo<B, OA, OC, OD>(
        &self,
        path: &Path,
        agent_online: &mut Agent<B>,
        agent_target: &mut Agent<B>,
        duals: &mut MpoDuals<B>,
        actor_optim: &mut OA,
        critic_optim: &mut OC,
        dual_optim: &mut OD,
        device: &B::Device,
    ) -> Result<Option<usize>>
    where
        B: AutodiffBackend,
        OA: Optimizer<Actor<B>, B>,
        OC: Optimizer<Critic<B>, B>,
        OD: Optimizer<MpoDuals<B>, B>,
    {
        Self::validate_native_checkpoint_path(path)?;
        let load_layout = Self::resolve_load_layout(path, AGENT_ONLINE_FILE)?;

        *agent_online = agent_online
            .clone()
            .load_file(load_layout.model_base_path.clone(), &self.recorder, device)
            .context("failed to load SPO online agent checkpoint")?;

        let target_path = load_layout.step_dir.join(AGENT_TARGET_FILE);
        if Self::mpk_file_exists(&target_path) {
            *agent_target = agent_target
                .clone()
                .load_file(target_path, &self.recorder, device)
                .context("failed to load SPO target agent checkpoint")?;
        } else {
            *agent_target = agent_online.clone();
        }

        let dual_path = load_layout.step_dir.join(DUAL_FILE);
        if Self::mpk_file_exists(&dual_path) {
            *duals = duals
                .clone()
                .load_file(dual_path, &self.recorder, device)
                .context("failed to load SPO dual state checkpoint")?;
        }

        let actor_optim_path = load_layout.step_dir.join(ACTOR_OPTIM_FILE);
        let critic_optim_path = load_layout.step_dir.join(CRITIC_OPTIM_FILE);
        let dual_optim_path = load_layout.step_dir.join(DUAL_OPTIM_FILE);
        if Self::mpk_file_exists(&actor_optim_path)
            && Self::mpk_file_exists(&critic_optim_path)
            && Self::mpk_file_exists(&dual_optim_path)
        {
            let actor_record = self
                .recorder
                .load(actor_optim_path, device)
                .context("failed to load SPO actor optimizer checkpoint")?;
            let critic_record = self
                .recorder
                .load(critic_optim_path, device)
                .context("failed to load SPO critic optimizer checkpoint")?;
            let dual_record = self
                .recorder
                .load(dual_optim_path, device)
                .context("failed to load SPO dual optimizer checkpoint")?;

            *actor_optim = actor_optim.clone().load_record(actor_record);
            *critic_optim = critic_optim.clone().load_record(critic_record);
            *dual_optim = dual_optim.clone().load_record(dual_record);
        }

        Ok(self.read_metadata_step(&load_layout.step_dir)?)
    }

    pub fn load_path(&self) -> Option<&Path> {
        self.config.load_path.as_deref()
    }

    fn step_dir(&self, step: usize) -> PathBuf {
        self.run_dir.join(format!("{STEP_PREFIX}{step}"))
    }

    fn validate_native_checkpoint_path(path: &Path) -> Result<()> {
        let ext = path
            .extension()
            .and_then(|s| s.to_str())
            .map(|s| s.to_ascii_lowercase());

        match ext.as_deref() {
            Some("mpk") => Ok(()),
            Some("onnx") => bail!(
                "runtime ONNX loading is not supported; convert the ONNX model offline to a native .mpk checkpoint and pass that path"
            ),
            Some(other) => bail!("unsupported checkpoint extension '.{other}', expected '.mpk'"),
            None => bail!("checkpoint load path must be a .mpk file"),
        }
    }

    fn step_dir_from_load_path(path: &Path) -> Result<PathBuf> {
        let parent = path
            .parent()
            .ok_or_else(|| anyhow::anyhow!("checkpoint path '{}' has no parent directory", path.display()))?;
        Ok(parent.to_path_buf())
    }

    fn resolve_load_layout(path: &Path, default_model_file_name: &str) -> Result<LoadLayout> {
        let step_dir = Self::step_dir_from_load_path(path)?;

        let file_stem = path.file_stem().and_then(|v| v.to_str());
        let model_base_path = match file_stem {
            Some(stem) if !stem.is_empty() => {
                if stem == default_model_file_name {
                    step_dir.join(default_model_file_name)
                } else {
                    path.with_extension("")
                }
            }
            _ => step_dir.join(default_model_file_name),
        };

        Ok(LoadLayout {
            step_dir,
            model_base_path,
        })
    }

    fn mpk_file_exists(base_path: &Path) -> bool {
        let mut path = base_path.to_path_buf();
        path.set_extension("mpk");
        path.exists()
    }

    fn resolve_run_dir(config: &CheckpointConfig, algorithm: &str) -> Result<PathBuf> {
        if let Some(load_path) = config.load_path.as_ref() {
            if let Some(step_dir) = load_path.parent() {
                if let Some(step_name) = step_dir.file_name().and_then(|v| v.to_str()) {
                    if step_name.starts_with(STEP_PREFIX) {
                        if let Some(run_dir) = step_dir.parent() {
                            return Ok(run_dir.to_path_buf());
                        }
                    }
                }
            }
        }

        let run_id = Utc::now().format("run_%Y%m%d_%H%M%S").to_string();
        Ok(config.save_dir.join(algorithm).join(run_id))
    }

    fn write_metadata(&self, step_dir: &Path, step: usize, algorithm: &str) -> Result<()> {
        let metadata = CheckpointMetadata {
            step,
            created_at_utc: Utc::now().to_rfc3339(),
            algorithm: algorithm.to_string(),
        };

        let path = step_dir.join(METADATA_FILE);
        let json = serde_json::to_string_pretty(&metadata)
            .context("failed to serialize checkpoint metadata")?;
        fs::write(&path, json)
            .with_context(|| format!("failed to write checkpoint metadata {}", path.display()))?;
        Ok(())
    }

    fn read_metadata_step(&self, step_dir: &Path) -> Result<Option<usize>> {
        let path = step_dir.join(METADATA_FILE);
        if !path.exists() {
            return Ok(None);
        }

        let content = fs::read_to_string(&path)
            .with_context(|| format!("failed to read checkpoint metadata {}", path.display()))?;
        let metadata: CheckpointMetadata = serde_json::from_str(&content)
            .with_context(|| format!("failed to parse checkpoint metadata {}", path.display()))?;
        Ok(Some(metadata.step))
    }

    fn rotate_if_needed(&self) -> Result<()> {
        let Some(keep_last_n) = self.config.keep_last_n else {
            return Ok(());
        };

        let Some(algorithm_dir) = self.run_dir.parent() else {
            return Ok(());
        };

        let mut step_dirs: Vec<(SystemTime, PathBuf)> = Vec::new();

        for run_entry in fs::read_dir(algorithm_dir)
            .with_context(|| format!("failed to read algorithm checkpoint dir {}", algorithm_dir.display()))?
        {
            let run_entry = run_entry?;
            let run_path = run_entry.path();
            if !run_path.is_dir() {
                continue;
            }

            for step_entry in fs::read_dir(&run_path).with_context(|| {
                format!("failed to read checkpoint run dir {}", run_path.display())
            })? {
                let step_entry = step_entry?;
                let step_path = step_entry.path();
                if !step_path.is_dir() {
                    continue;
                }

                let Some(name) = step_path.file_name().and_then(|v| v.to_str()) else {
                    continue;
                };
                if !name.starts_with(STEP_PREFIX) {
                    continue;
                }

                let modified = fs::metadata(&step_path)
                    .and_then(|meta| meta.modified())
                    .unwrap_or(SystemTime::UNIX_EPOCH);
                step_dirs.push((modified, step_path));
            }
        }

        if step_dirs.len() <= keep_last_n {
            return Ok(());
        }

        step_dirs.sort_by_key(|(modified, _)| *modified);
        let to_remove = step_dirs.len().saturating_sub(keep_last_n);
        for (_, path) in step_dirs.into_iter().take(to_remove) {
            fs::remove_dir_all(&path).with_context(|| {
                format!("failed to remove rotated checkpoint directory {}", path.display())
            })?;

            if let Some(run_dir) = path.parent() {
                if run_dir.read_dir().map(|mut it| it.next().is_none()).unwrap_or(false) {
                    let _ = fs::remove_dir(run_dir);
                }
            }
        }

        Ok(())
    }
}

struct LoadLayout {
    step_dir: PathBuf,
    model_base_path: PathBuf,
}
