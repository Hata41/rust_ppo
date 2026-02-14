use anyhow::{bail, Context, Result};
use clap::{parser::ValueSource, ArgMatches, CommandFactory, Parser, ValueEnum};
use serde::Deserialize;
use std::path::{Path, PathBuf};

#[derive(Copy, Clone, Debug, ValueEnum, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum DeviceType {
    Cuda,
    Cpu,
}

#[derive(Parser, Debug, Clone)]
#[command(name = "rust_ppo")]
pub struct Args {
    /// Optional YAML config file path.
    #[arg(long)]
    pub config: Option<PathBuf>,

    /// rustpool task id: "Maze-v0", "BinPack-v0"
    #[arg(long, default_value = "BinPack-v0")]
    pub task_id: String,

    /// Number of parallel envs (Anakin-style throughput comes mostly from large N here).
    #[arg(long, default_value_t = 64)]
    pub num_envs: usize,

    /// PPO rollout length (T)
    #[arg(long, default_value_t = 128)]
    pub rollout_length: usize,

    /// Total PPO updates
    #[arg(long, default_value_t = 2000)]
    pub num_updates: usize,

    /// Number of updates between deterministic evaluations
    #[arg(long, default_value_t = 20)]
    pub eval_interval: usize,

    /// Number of environments to use for deterministic evaluation
    #[arg(long, default_value_t = 32)]
    pub num_eval_envs: usize,

    /// Total completed episodes to collect per deterministic evaluation
    #[arg(long, default_value_t = 32)]
    pub num_eval_episodes: usize,

    /// PPO epochs per update
    #[arg(long, default_value_t = 4)]
    pub epochs: usize,

    /// Number of minibatches per epoch (total batch size must be divisible by this)
    #[arg(long, default_value_t = 32)]
    pub num_minibatches: usize,

    /// Discount gamma
    #[arg(long, default_value_t = 0.99)]
    pub gamma: f32,

    /// GAE lambda
    #[arg(long, default_value_t = 0.95)]
    pub gae_lambda: f32,

    /// PPO clip epsilon
    #[arg(long, default_value_t = 0.2)]
    pub clip_eps: f32,

    /// Entropy coefficient
    #[arg(long, default_value_t = 0.01)]
    pub ent_coef: f32,

    /// Value loss coefficient
    #[arg(long, default_value_t = 0.5)]
    pub vf_coef: f32,

    /// Actor LR (Adam)
    #[arg(long, default_value_t = 3e-4)]
    pub actor_lr: f64,

    /// Critic LR (Adam)
    #[arg(long, default_value_t = 1e-3)]
    pub critic_lr: f64,

    /// Global gradient clipping threshold (L2 norm)
    #[arg(long, default_value_t = 0.5)]
    pub max_grad_norm: f32,

    /// Linearly decay actor/critic learning rates over updates
    #[arg(long, default_value_t = true)]
    pub decay_learning_rates: bool,

    /// Reward scaling (multiply env reward by this before GAE)
    #[arg(long, default_value_t = 1.0)]
    pub reward_scale: f32,

    /// Standardize advantages per update
    #[arg(long, default_value_t = true)]
    pub standardize_advantages: bool,

    /// Hidden dim of MLP
    #[arg(long, default_value_t = 256)]
    pub hidden_dim: usize,

    /// RNG seed
    #[arg(long, default_value_t = 0)]
    pub seed: u64,

    /// CUDA device index (0 = cuda:0)
    #[arg(long, default_value_t = 0)]
    pub cuda_device: usize,

    /// Device backend to use: cuda or cpu
    #[arg(long, value_enum, default_value_t = DeviceType::Cuda)]
    pub device_type: DeviceType,

    /// Global rank (used if RANK env var is not set)
    #[arg(long, default_value_t = 0)]
    pub rank: usize,

    /// World size (used if WORLD_SIZE env var is not set)
    #[arg(long, default_value_t = 1)]
    pub world_size: usize,

    /// Local rank / local GPU index (used if LOCAL_RANK env var is not set)
    #[arg(long, default_value_t = 0)]
    pub local_rank: usize,

    #[arg(long, default_value_t = 20)]
    pub max_items: usize,

    #[arg(long, default_value_t = 40)]
    pub max_ems: usize,

    /// Max episode steps (used by some envs internally; rustpool auto-resets on done)
    #[arg(long, default_value_t = 200)]
    pub max_episode_steps: usize,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            config: None,
            task_id: "BinPack-v0".to_string(),
            num_envs: 64,
            rollout_length: 128,
            num_updates: 2000,
            eval_interval: 20,
            num_eval_envs: 32,
            num_eval_episodes: 32,
            epochs: 4,
            num_minibatches: 32,
            gamma: 0.99,
            gae_lambda: 0.95,
            clip_eps: 0.2,
            ent_coef: 0.01,
            vf_coef: 0.5,
            actor_lr: 3e-4,
            critic_lr: 1e-3,
            max_grad_norm: 0.5,
            decay_learning_rates: true,
            reward_scale: 1.0,
            standardize_advantages: true,
            hidden_dim: 256,
            seed: 0,
            cuda_device: 0,
            device_type: DeviceType::Cuda,
            rank: 0,
            world_size: 1,
            local_rank: 0,
            max_items: 20,
            max_ems: 40,
            max_episode_steps: 200,
        }
    }
}

#[derive(Debug, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
struct FileConfig {
    environment: EnvironmentConfig,
    ppo_core: PpoCoreConfig,
    optimization: OptimizationConfig,
    architecture: ArchitectureConfig,
    evaluation: EvaluationConfig,
    hardware: HardwareConfig,
    distributed: DistributedConfig,
}

#[derive(Debug, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
struct EnvironmentConfig {
    task_id: Option<String>,
    num_envs: Option<usize>,
    max_items: Option<usize>,
    max_ems: Option<usize>,
    max_episode_steps: Option<usize>,
}

#[derive(Debug, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
struct PpoCoreConfig {
    rollout_length: Option<usize>,
    num_updates: Option<usize>,
    epochs: Option<usize>,
    num_minibatches: Option<usize>,
    gamma: Option<f32>,
    gae_lambda: Option<f32>,
}

#[derive(Debug, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
struct OptimizationConfig {
    actor_lr: Option<f64>,
    critic_lr: Option<f64>,
    max_grad_norm: Option<f32>,
    clip_eps: Option<f32>,
    ent_coef: Option<f32>,
    vf_coef: Option<f32>,
    decay_learning_rates: Option<bool>,
    standardize_advantages: Option<bool>,
    reward_scale: Option<f32>,
}

#[derive(Debug, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
struct ArchitectureConfig {
    hidden_dim: Option<usize>,
    seed: Option<u64>,
}

#[derive(Debug, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
struct EvaluationConfig {
    eval_interval: Option<usize>,
    num_eval_envs: Option<usize>,
    num_eval_episodes: Option<usize>,
}

#[derive(Debug, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
struct HardwareConfig {
    device_type: Option<DeviceType>,
    cuda_device: Option<usize>,
}

#[derive(Debug, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
struct DistributedConfig {
    rank: Option<usize>,
    world_size: Option<usize>,
    local_rank: Option<usize>,
}

impl Args {
    pub fn load() -> Result<Self> {
        let argv = std::env::args_os().collect::<Vec<_>>();
        let cli_args = Self::try_parse_from(&argv)
            .map_err(|e| anyhow::anyhow!(e.to_string()))
            .context("failed to parse CLI arguments")?;
        let matches = Self::command()
            .try_get_matches_from(&argv)
            .map_err(|e| anyhow::anyhow!(e.to_string()))
            .context("failed to parse CLI arguments")?;

        let mut merged = Self::default();

        if let Some(config_path) = cli_args.config.as_deref() {
            let file_config = Self::load_file_config(config_path)?;
            merged.apply_config_file(file_config);
        }

        merged.apply_cli_overrides(&cli_args, &matches);
        merged.config = cli_args.config;

        Ok(merged)
    }

    fn load_file_config(path: &Path) -> Result<FileConfig> {
        let resolved = if path.is_absolute() {
            path.to_path_buf()
        } else {
            std::env::current_dir()
                .context("failed to get current working directory")?
                .join(path)
        };

        let content = std::fs::read_to_string(&resolved)
            .with_context(|| format!("failed to read config file at {}", resolved.display()))?;

        serde_yaml::from_str::<FileConfig>(&content)
            .with_context(|| format!("failed to parse YAML config at {}", resolved.display()))
    }

    fn apply_config_file(&mut self, file: FileConfig) {
        macro_rules! set_if_some {
            ($field:ident, $value:expr) => {
                if let Some(value) = $value {
                    self.$field = value;
                }
            };
        }

        set_if_some!(task_id, file.environment.task_id);
        set_if_some!(num_envs, file.environment.num_envs);
        set_if_some!(max_items, file.environment.max_items);
        set_if_some!(max_ems, file.environment.max_ems);
        set_if_some!(max_episode_steps, file.environment.max_episode_steps);

        set_if_some!(rollout_length, file.ppo_core.rollout_length);
        set_if_some!(num_updates, file.ppo_core.num_updates);
        set_if_some!(epochs, file.ppo_core.epochs);
        set_if_some!(num_minibatches, file.ppo_core.num_minibatches);
        set_if_some!(gamma, file.ppo_core.gamma);
        set_if_some!(gae_lambda, file.ppo_core.gae_lambda);

        set_if_some!(actor_lr, file.optimization.actor_lr);
        set_if_some!(critic_lr, file.optimization.critic_lr);
        set_if_some!(max_grad_norm, file.optimization.max_grad_norm);
        set_if_some!(clip_eps, file.optimization.clip_eps);
        set_if_some!(ent_coef, file.optimization.ent_coef);
        set_if_some!(vf_coef, file.optimization.vf_coef);
        set_if_some!(decay_learning_rates, file.optimization.decay_learning_rates);
        set_if_some!(standardize_advantages, file.optimization.standardize_advantages);
        set_if_some!(reward_scale, file.optimization.reward_scale);

        set_if_some!(hidden_dim, file.architecture.hidden_dim);
        set_if_some!(seed, file.architecture.seed);

        set_if_some!(eval_interval, file.evaluation.eval_interval);
        set_if_some!(num_eval_envs, file.evaluation.num_eval_envs);
        set_if_some!(num_eval_episodes, file.evaluation.num_eval_episodes);

        set_if_some!(device_type, file.hardware.device_type);
        set_if_some!(cuda_device, file.hardware.cuda_device);

        set_if_some!(rank, file.distributed.rank);
        set_if_some!(world_size, file.distributed.world_size);
        set_if_some!(local_rank, file.distributed.local_rank);
    }

    fn apply_cli_overrides(&mut self, cli: &Self, matches: &ArgMatches) {
        macro_rules! set_if_cli {
            ($field:ident, $arg_name:literal) => {
                if Self::provided_on_cli(matches, $arg_name) {
                    self.$field = cli.$field.clone();
                }
            };
        }

        set_if_cli!(task_id, "task_id");
        set_if_cli!(num_envs, "num_envs");
        set_if_cli!(max_items, "max_items");
        set_if_cli!(max_ems, "max_ems");
        set_if_cli!(max_episode_steps, "max_episode_steps");

        set_if_cli!(rollout_length, "rollout_length");
        set_if_cli!(num_updates, "num_updates");
        set_if_cli!(epochs, "epochs");
        set_if_cli!(num_minibatches, "num_minibatches");
        set_if_cli!(gamma, "gamma");
        set_if_cli!(gae_lambda, "gae_lambda");

        set_if_cli!(actor_lr, "actor_lr");
        set_if_cli!(critic_lr, "critic_lr");
        set_if_cli!(max_grad_norm, "max_grad_norm");
        set_if_cli!(clip_eps, "clip_eps");
        set_if_cli!(ent_coef, "ent_coef");
        set_if_cli!(vf_coef, "vf_coef");
        set_if_cli!(decay_learning_rates, "decay_learning_rates");
        set_if_cli!(standardize_advantages, "standardize_advantages");
        set_if_cli!(reward_scale, "reward_scale");

        set_if_cli!(hidden_dim, "hidden_dim");
        set_if_cli!(seed, "seed");

        set_if_cli!(eval_interval, "eval_interval");
        set_if_cli!(num_eval_envs, "num_eval_envs");
        set_if_cli!(num_eval_episodes, "num_eval_episodes");

        set_if_cli!(device_type, "device_type");
        set_if_cli!(cuda_device, "cuda_device");

        set_if_cli!(rank, "rank");
        set_if_cli!(world_size, "world_size");
        set_if_cli!(local_rank, "local_rank");
    }

    fn provided_on_cli(matches: &ArgMatches, arg_name: &str) -> bool {
        matches.value_source(arg_name) == Some(ValueSource::CommandLine)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DistInfo {
    pub rank: usize,
    pub world_size: usize,
    pub local_rank: usize,
}

impl DistInfo {
    pub fn from_env_or_args(args: &Args) -> Result<Self> {
        let rank = std::env::var("RANK")
            .ok()
            .map(|v| v.parse::<usize>().context("invalid RANK"))
            .transpose()?
            .unwrap_or(args.rank);

        let world_size = std::env::var("WORLD_SIZE")
            .ok()
            .map(|v| v.parse::<usize>().context("invalid WORLD_SIZE"))
            .transpose()?
            .unwrap_or(args.world_size);

        let local_rank = std::env::var("LOCAL_RANK")
            .ok()
            .map(|v| v.parse::<usize>().context("invalid LOCAL_RANK"))
            .transpose()?
            .unwrap_or(args.local_rank);

        if world_size == 0 {
            bail!("WORLD_SIZE must be > 0");
        }
        if rank >= world_size {
            bail!("RANK ({rank}) must be < WORLD_SIZE ({world_size})");
        }

        Ok(Self {
            rank,
            world_size,
            local_rank,
        })
    }
}