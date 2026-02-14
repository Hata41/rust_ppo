use anyhow::{bail, Context, Result};
use bitvec::prelude::*;
use clap::Parser;
use crossbeam_channel::{unbounded, Receiver, Sender};
use rand::seq::SliceRandom;
use rand::{rngs::StdRng, SeedableRng};
use std::collections::HashMap;
use std::thread;

use burn::backend::autodiff::Autodiff;
use burn::backend::cuda::{Cuda, CudaDevice};
use burn::prelude::Backend;
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::optim::{AdamConfig, GradientsParams, Optimizer};
use burn::tensor::activation::{log_softmax, softmax};
use burn::tensor::{Distribution, Tensor, TensorData};
use burn::tensor::Int;

use rustpool::core::rl_env::RlEnv;
use rustpool::core::types::{ArrayData, GenericObs};
use rustpool::core::worker::{worker_loop, WorkerAction, WorkerMessage};

use rustpool::envs::binpack::{BinPackConfig, BinPackEnv, RewardFnType};
use rustpool::envs::maze::{MazeConfig, MazeEnv};

/// Burn backend: Autodiff on CUDA (NVIDIA GPU).
type InnerBackend = Cuda<f32, i32>;
type B = Autodiff<InnerBackend>;

#[derive(Parser, Debug, Clone)]
#[command(name = "rust_ppo")]
struct Args {
    /// rustpool task id: "Maze-v0", "BinPack-v0"
    #[arg(long, default_value = "BinPack-v0")]
    task_id: String,

    /// Number of parallel envs (Anakin-style throughput comes mostly from large N here).
    #[arg(long, default_value_t = 64)]
    num_envs: usize,

    /// PPO rollout length (T)
    #[arg(long, default_value_t = 128)]
    rollout_length: usize,

    /// Total PPO updates
    #[arg(long, default_value_t = 2000)]
    num_updates: usize,

    /// PPO epochs per update
    #[arg(long, default_value_t = 4)]
    epochs: usize,

    /// Number of minibatches per epoch (total batch size must be divisible by this)
    #[arg(long, default_value_t = 32)]
    num_minibatches: usize,

    /// Discount gamma
    #[arg(long, default_value_t = 0.99)]
    gamma: f32,

    /// GAE lambda
    #[arg(long, default_value_t = 0.95)]
    gae_lambda: f32,

    /// PPO clip epsilon
    #[arg(long, default_value_t = 0.2)]
    clip_eps: f32,

    /// Entropy coefficient
    #[arg(long, default_value_t = 0.01)]
    ent_coef: f32,

    /// Value loss coefficient
    #[arg(long, default_value_t = 0.5)]
    vf_coef: f32,

    /// Actor LR (Adam)
    #[arg(long, default_value_t = 3e-4)]
    actor_lr: f64,

    /// Critic LR (Adam)
    #[arg(long, default_value_t = 1e-3)]
    critic_lr: f64,

    /// Reward scaling (multiply env reward by this before GAE)
    #[arg(long, default_value_t = 1.0)]
    reward_scale: f32,

    /// Standardize advantages per update
    #[arg(long, default_value_t = true)]
    standardize_advantages: bool,

    /// Hidden dim of MLP
    #[arg(long, default_value_t = 256)]
    hidden_dim: usize,

    /// RNG seed
    #[arg(long, default_value_t = 0)]
    seed: u64,

    /// CUDA device index (0 = cuda:0)
    #[arg(long, default_value_t = 0)]
    cuda_device: usize,

    // --- Common rustpool knobs (used for BinPack) ---
    #[arg(long, default_value_t = 20)]
    max_items: usize,

    #[arg(long, default_value_t = 40)]
    max_ems: usize,

    /// Max episode steps (used by some envs internally; rustpool auto-resets on done)
    #[arg(long, default_value_t = 200)]
    max_episode_steps: usize,
}

#[derive(Clone, Debug)]
struct StepOut {
    obs: GenericObs,
    reward: f32,
    done: bool,
    action_mask: Vec<bool>,
}

/// Minimal Rust envpool (pure Rust) built on rustpool's worker_loop. 
struct AsyncEnvPool {
    num_envs: usize,
    num_threads: usize,
    action_txs: Vec<Sender<WorkerAction>>,
    state_rx: Receiver<WorkerMessage>,
    worker_handles: Vec<thread::JoinHandle<()>>,
}

impl AsyncEnvPool {
    fn new<F>(num_envs: usize, base_seed: u64, mut factory: F) -> Result<Self>
    where
        F: FnMut(u64) -> Box<dyn RlEnv> + Send + 'static,
    {
        if num_envs == 0 {
            bail!("num_envs must be > 0");
        }

        let num_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .min(num_envs);

        let mut shards: Vec<HashMap<usize, Box<dyn RlEnv>>> =
            (0..num_threads).map(|_| HashMap::new()).collect();

        for env_id in 0..num_envs {
            let shard_id = env_id % num_threads;
            let env = factory(base_seed + env_id as u64);
            shards[shard_id].insert(env_id, env);
        }

        let (state_tx, state_rx) = unbounded::<WorkerMessage>();
        let mut action_txs = Vec::with_capacity(num_threads);
        let mut worker_handles = Vec::with_capacity(num_threads);

        for shard_id in 0..num_threads {
            let (action_tx, action_rx) = unbounded::<WorkerAction>();
            action_txs.push(action_tx);

            let envs_for_worker = std::mem::take(&mut shards[shard_id]);
            let state_tx_clone = state_tx.clone();

            let handle = thread::spawn(move || worker_loop(shard_id, envs_for_worker, action_rx, state_tx_clone));
            worker_handles.push(handle);
        }

        Ok(Self {
            num_envs,
            num_threads,
            action_txs,
            state_rx,
            worker_handles,
        })
    }

    fn route_tx(&self, env_id: usize) -> &Sender<WorkerAction> {
        let shard = env_id % self.num_threads;
        &self.action_txs[shard]
    }

    fn async_reset(&self, seed: Option<u64>) -> Result<()> {
        let seed_map = seed.map(|s| {
            (0..self.num_envs)
                .map(|env_id| (env_id, s + env_id as u64))
                .collect::<Vec<_>>()
        });

        // Send the same seed map to all shards; worker filters to its envs.
        for tx in &self.action_txs {
            tx.send(WorkerAction::Reset(seed_map.clone()))
                .context("failed to send reset")?;
        }
        Ok(())
    }

    fn recv_n(&self, n: usize) -> Result<Vec<StepOut>> {
        let mut out = vec![
            StepOut {
                obs: vec![],
                reward: 0.0,
                done: false,
                action_mask: vec![],
            };
            n
        ];

        let mut got = 0usize;
        while got < n {
            match self.state_rx.recv().context("envpool recv failed")? {
                WorkerMessage::StepResult { env_id, obs, reward, done, action_mask } => {
                    if env_id < n {
                        out[env_id] = StepOut { obs, reward, done, action_mask };
                    }
                    got += 1;
                }
                WorkerMessage::SnapshotResult { .. } => {
                    // ignore in PPO
                }
            }
        }
        Ok(out)
    }

    fn reset_all(&self, seed: Option<u64>) -> Result<Vec<StepOut>> {
        self.async_reset(seed)?;
        self.recv_n(self.num_envs)
    }

    fn step_all(&self, actions: &[i32]) -> Result<Vec<StepOut>> {
        if actions.len() != self.num_envs {
            bail!("actions length must match num_envs");
        }

        for env_id in 0..self.num_envs {
            self.route_tx(env_id)
                .send(WorkerAction::Step(env_id, actions[env_id]))
                .context("failed to send step")?;
        }

        self.recv_n(self.num_envs)
    }
}

impl Drop for AsyncEnvPool {
    fn drop(&mut self) {
        for tx in &self.action_txs {
            let _ = tx.send(WorkerAction::Close);
        }
        for h in self.worker_handles.drain(..) {
            let _ = h.join();
        }
    }
}

// ---------------------- Observation flattening ----------------------

fn flatten_obs(obs: &GenericObs) -> Vec<f32> {
    let mut out = Vec::new();
    for a in obs {
        match a {
            ArrayData::Float32(v) => out.extend_from_slice(v),
            ArrayData::Int32(v) => out.extend(v.iter().map(|x| *x as f32)),
            ArrayData::Bool(v) => out.extend(v.iter().map(|b| if *b { 1.0 } else { 0.0 })),
        }
    }
    out
}

// ---------------------- Packed action masks ----------------------
#[derive(Clone)]
struct PackedMasks {
    action_dim: usize,
    words_per_mask: usize,
    bits: BitVec<u64, Lsb0>,
}

impl PackedMasks {
    fn new(action_dim: usize, num_samples: usize) -> Self {
        let words_per_mask = (action_dim + 63) / 64;
        let total_bits = num_samples * words_per_mask * 64;
        Self {
            action_dim,
            words_per_mask,
            bits: bitvec![u64, Lsb0; 0; total_bits],
        }
    }

    fn set_mask(&mut self, sample_idx: usize, mask: &[bool]) {
        let base_bit = sample_idx * self.words_per_mask * 64;
        for (i, &m) in mask.iter().enumerate().take(self.action_dim) {
            self.bits.set(base_bit + i, m);
        }
    }

    fn unpack_to_f32(&self, indices: &[usize]) -> Vec<f32> {
        let mut out = vec![0.0f32; indices.len() * self.action_dim];
        for (row, &sample_idx) in indices.iter().enumerate() {
            let base_bit = sample_idx * self.words_per_mask * 64;
            let row_base = row * self.action_dim;
            for a in 0..self.action_dim {
                out[row_base + a] = if self.bits[base_bit + a] { 1.0 } else { 0.0 };
            }
        }
        out
    }
}

// ---------------------- Networks ----------------------

#[derive(Module, Clone, Debug)]
struct Mlp<Bk: burn::tensor::backend::Backend> {
    l1: Linear<Bk>,
    l2: Linear<Bk>,
}

impl<Bk: burn::tensor::backend::Backend> Mlp<Bk> {
    fn new(in_dim: usize, hidden: usize, device: &Bk::Device) -> Self {
        let l1 = LinearConfig::new(in_dim, hidden).init(device);
        let l2 = LinearConfig::new(hidden, hidden).init(device);
        Self { l1, l2 }
    }

    fn forward(&self, x: Tensor<Bk, 2>) -> Tensor<Bk, 2> {
        let x = burn::tensor::activation::relu(self.l1.forward(x));
        burn::tensor::activation::relu(self.l2.forward(x))
    }
}

#[derive(Module, Clone, Debug)]
struct Actor<Bk: burn::tensor::backend::Backend> {
    torso: Mlp<Bk>,
    head: Linear<Bk>,
}

impl<Bk: burn::tensor::backend::Backend> Actor<Bk> {
    fn new(obs_dim: usize, hidden: usize, action_dim: usize, device: &Bk::Device) -> Self {
        let torso = Mlp::new(obs_dim, hidden, device);
        let head = LinearConfig::new(hidden, action_dim).init(device);
        Self { torso, head }
    }

    fn forward(&self, obs: Tensor<Bk, 2>) -> Tensor<Bk, 2> {
        let h = self.torso.forward(obs);
        self.head.forward(h) // logits
    }
}

#[derive(Module, Clone, Debug)]
struct Critic<Bk: burn::tensor::backend::Backend> {
    torso: Mlp<Bk>,
    head: Linear<Bk>,
}

impl<Bk: burn::tensor::backend::Backend> Critic<Bk> {
    fn new(obs_dim: usize, hidden: usize, device: &Bk::Device) -> Self {
        let torso = Mlp::new(obs_dim, hidden, device);
        let head = LinearConfig::new(hidden, 1).init(device);
        Self { torso, head }
    }

    fn forward(&self, obs: Tensor<Bk, 2>) -> Tensor<Bk, 2> {
        let h = self.torso.forward(obs);
        self.head.forward(h) // value
    }
}

#[derive(Module, Clone, Debug)]
struct Agent<Bk: burn::tensor::backend::Backend> {
    actor: Actor<Bk>,
    critic: Critic<Bk>,
}

impl<Bk: burn::tensor::backend::Backend> Agent<Bk> {
    fn new(obs_dim: usize, hidden: usize, action_dim: usize, device: &Bk::Device) -> Self {
        Self {
            actor: Actor::new(obs_dim, hidden, action_dim, device),
            critic: Critic::new(obs_dim, hidden, device),
        }
    }
}

// ---------------------- PPO helpers ----------------------

fn masked_logits<Bk: burn::tensor::backend::Backend>(
    logits: Tensor<Bk, 2>,
    mask_f32: Tensor<Bk, 2>,
) -> Tensor<Bk, 2> {
    // mask_f32 is 1.0 for valid actions, 0.0 for invalid.
    // Add -1e9 to invalid actions: logits + (mask - 1) * 1e9
    let one = Tensor::<Bk, 2>::ones_like(&mask_f32);
    logits + (mask_f32 - one) * 1.0e9
}

fn logprob_and_entropy<Bk: burn::tensor::backend::Backend>(
    logits: Tensor<Bk, 2>,
    mask_f32: Tensor<Bk, 2>,
    actions: Tensor<Bk, 1, Int>,
) -> (Tensor<Bk, 1>, Tensor<Bk, 1>) {
    let masked = masked_logits(logits, mask_f32);

    let lp = log_softmax(masked.clone(), 1); // [B, A]
    let probs = softmax(masked, 1);          // [B, A]

    // gather log prob of chosen action
    let bsz = actions.dims()[0];
    let idx2 = actions.reshape([bsz, 1]);            // [B, 1]
    let chosen_lp = lp.clone().gather(1, idx2).reshape([bsz]); // [B]

    // entropy: -sum(p * logp)
    let ent = (probs * lp).sum_dim(1).neg().squeeze::<1>(); // [B]

    (chosen_lp, ent)
}

fn sample_actions_gumbel<Bk: burn::tensor::backend::Backend>(
    logits: Tensor<Bk, 2>,
    mask_f32: Tensor<Bk, 2>,
    device: &Bk::Device,
) -> Tensor<Bk, 1, Int> {
    let masked = masked_logits(logits, mask_f32);

    let [b, a] = masked.dims();

    // Use Uniform(eps, 1-eps) for stable Gumbel noise. Distribution variants are in Burn.
    let u = Tensor::<Bk, 2>::random([b, a], Distribution::Uniform(1.0e-6, 1.0 - 1.0e-6), device);

    // gumbel = -log(-log(u))
    let g = (u.log().neg()).log().neg();
    let noisy = masked + g;

    // argmax over action dim
    noisy.argmax(1).squeeze::<1>()
}

// ---------------------- Rollout storage ----------------------

struct Rollout {
    t: usize,
    n: usize,
    obs_dim: usize,
    _action_dim: usize,

    obs: Vec<f32>,          // [T*N, obs_dim]
    actions: Vec<i32>,      // [T*N]
    old_logp: Vec<f32>,     // [T*N]
    values: Vec<f32>,       // [T*N]
    rewards: Vec<f32>,      // [T*N]
    dones: Vec<u8>,         // [T*N] 0/1
    masks: PackedMasks,     // packed [T*N, action_dim]

    advantages: Vec<f32>,   // [T*N]
    targets: Vec<f32>,      // [T*N]
}

impl Rollout {
    fn new(t: usize, n: usize, obs_dim: usize, action_dim: usize) -> Self {
        let num_samples = t * n;
        Self {
            t,
            n,
            obs_dim,
            _action_dim: action_dim,
            obs: vec![0.0; num_samples * obs_dim],
            actions: vec![0; num_samples],
            old_logp: vec![0.0; num_samples],
            values: vec![0.0; num_samples],
            rewards: vec![0.0; num_samples],
            dones: vec![0; num_samples],
            masks: PackedMasks::new(action_dim, num_samples),
            advantages: vec![0.0; num_samples],
            targets: vec![0.0; num_samples],
        }
    }

    #[inline]
    fn idx(&self, t: usize, env: usize) -> usize {
        t * self.n + env
    }

    fn store_step(
        &mut self,
        t: usize,
        env: usize,
        obs_flat: &[f32],
        mask: &[bool],
        action: i32,
        logp: f32,
        value: f32,
        reward: f32,
        done: bool,
    ) {
        let i = self.idx(t, env);

        let obs_base = i * self.obs_dim;
        self.obs[obs_base..obs_base + self.obs_dim].copy_from_slice(obs_flat);

        self.masks.set_mask(i, mask);

        self.actions[i] = action;
        self.old_logp[i] = logp;
        self.values[i] = value;
        self.rewards[i] = reward;
        self.dones[i] = if done { 1 } else { 0 };
    }

    fn compute_gae(&mut self, last_values: &[f32], gamma: f32, lam: f32, reward_scale: f32, standardize: bool) {
        for env in 0..self.n {
            let mut gae = 0.0f32;
            let mut next_v = last_values[env];

            for t in (0..self.t).rev() {
                let i = self.idx(t, env);
                let done = self.dones[i] != 0;

                let r = self.rewards[i] * reward_scale;
                let v = self.values[i];

                let not_done = if done { 0.0 } else { 1.0 };
                let delta = r + gamma * not_done * next_v - v;
                gae = delta + gamma * lam * not_done * gae;

                self.advantages[i] = gae;
                self.targets[i] = gae + v;

                next_v = v;
            }
        }

        if standardize {
            let n = (self.t * self.n) as f32;
            let mean = self.advantages.iter().sum::<f32>() / n;
            let var = self
                .advantages
                .iter()
                .map(|a| {
                    let d = *a - mean;
                    d * d
                })
                .sum::<f32>()
                / n;
            let std = (var + 1.0e-8).sqrt();

            for a in &mut self.advantages {
                *a = (*a - mean) / std;
            }
        }
    }

    fn minibatch(&self, indices: &[usize]) -> (Vec<f32>, Vec<i32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let bsz = indices.len();

        // obs
        let mut obs_mb = vec![0.0f32; bsz * self.obs_dim];
        for (row, &idx) in indices.iter().enumerate() {
            let src = idx * self.obs_dim;
            let dst = row * self.obs_dim;
            obs_mb[dst..dst + self.obs_dim].copy_from_slice(&self.obs[src..src + self.obs_dim]);
        }

        // masks -> f32
        let mask_mb = self.masks.unpack_to_f32(indices);

        // scalars
        let mut act_mb = vec![0i32; bsz];
        let mut old_lp_mb = vec![0.0f32; bsz];
        let mut old_v_mb = vec![0.0f32; bsz];
        let mut adv_mb = vec![0.0f32; bsz];
        let mut tgt_mb = vec![0.0f32; bsz];

        for (row, &idx) in indices.iter().enumerate() {
            act_mb[row] = self.actions[idx];
            old_lp_mb[row] = self.old_logp[idx];
            old_v_mb[row] = self.values[idx];
            adv_mb[row] = self.advantages[idx];
            tgt_mb[row] = self.targets[idx];
        }

        (obs_mb, act_mb, mask_mb, old_lp_mb, old_v_mb, adv_mb, tgt_mb)
    }
}

// ---------------------- Env factory ----------------------

fn make_env(task_id: &str, args: &Args, seed: u64) -> Result<Box<dyn RlEnv>> {
    match task_id {
        "Maze-v0" => {
            let cfg = MazeConfig {
                width: 10,
                height: 10,
                max_episode_steps: args.max_episode_steps as i32,
            };
            Ok(Box::new(MazeEnv::new(cfg, seed)))
        }
        "BinPack-v0" => {
            let cfg = BinPackConfig {
                max_items: args.max_items,
                max_ems: args.max_ems,
                split_eps: 0.001,
                prob_split_one_item: 0.3,
                split_num_same_items: 1,
            };
            // BinPackEnv::new signature is in the repo. 
            Ok(Box::new(BinPackEnv::new(cfg, seed, RewardFnType::Dense)))
        }
        other => bail!("unknown task_id: {other}"),
    }
}

// ---------------------- Main training ----------------------

fn main() -> Result<()> {
    let args = Args::parse();

    if args.num_envs == 0 {
        bail!("num_envs must be > 0");
    }

    let total_batch = args.rollout_length * args.num_envs;
    if total_batch % args.num_minibatches != 0 {
        bail!(
            "rollout_length*num_envs = {total_batch} must be divisible by num_minibatches = {}",
            args.num_minibatches
        );
    }

    // CUDA device
    let device = CudaDevice::new(args.cuda_device);

    // Seed Burn RNG (affects Tensor::random).
    B::seed(&device, args.seed);

    // Build env pool
    let env_pool = AsyncEnvPool::new(args.num_envs, args.seed, {
        let args = args.clone();
        move |seed| make_env(&args.task_id, &args, seed).unwrap()
    })?;

    // Reset
    let reset_out = env_pool.reset_all(Some(args.seed + 10_000))?;
    let mut cur_obs: Vec<GenericObs> = reset_out.iter().map(|s| s.obs.clone()).collect();
    let mut cur_mask: Vec<Vec<bool>> = reset_out.iter().map(|s| s.action_mask.clone()).collect();

    // Infer dims from env0
    let obs0 = flatten_obs(&cur_obs[0]);
    let obs_dim = obs0.len();
    let action_dim = cur_mask[0].len();

    println!("Task: {}", args.task_id);
    println!("num_envs={} rollout_length={} total_batch={}", args.num_envs, args.rollout_length, total_batch);
    println!("obs_dim={} action_dim={}", obs_dim, action_dim);

    // Agent + optimizers
    let mut agent: Agent<B> = Agent::new(obs_dim, args.hidden_dim, action_dim, &device);

    let mut actor_optim = AdamConfig::new().init::<B, Actor<B>>();
    let mut critic_optim = AdamConfig::new().init::<B, Critic<B>>();

    // Episode logging
    let mut ep_return = vec![0.0f32; args.num_envs];
    let mut ep_len = vec![0usize; args.num_envs];
    let mut recent_returns: Vec<f32> = Vec::with_capacity(256);

    let mut rng = StdRng::seed_from_u64(args.seed ^ 0xA11CE);

    for update in 0..args.num_updates {
        let mut roll = Rollout::new(args.rollout_length, args.num_envs, obs_dim, action_dim);

        // -------- Rollout collection (no-grad / detached outputs) --------
        for t in 0..args.rollout_length {
            // Build obs batch tensor
            let mut obs_flat_all = vec![0.0f32; args.num_envs * obs_dim];
            for e in 0..args.num_envs {
                let flat = flatten_obs(&cur_obs[e]);
                let base = e * obs_dim;
                obs_flat_all[base..base + obs_dim].copy_from_slice(&flat);
            }
            let obs_t = Tensor::<B, 2>::from_data(
                TensorData::new(obs_flat_all, [args.num_envs, obs_dim]),
                &device,
            );

            // Mask tensor for sampling/logprob calc
            // (constructed on the fly for the current step only)
            let mut mask_f = vec![0.0f32; args.num_envs * action_dim];
            for e in 0..args.num_envs {
                let row = e * action_dim;
                for a in 0..action_dim {
                    mask_f[row + a] = if cur_mask[e][a] { 1.0 } else { 0.0 };
                }
            }
            let mask_t = Tensor::<B, 2>::from_data(
                TensorData::new(mask_f, [args.num_envs, action_dim]),
                &device,
            );

            // Forward (detach so we don't build a long graph during rollout)
            let logits = agent.actor.forward(obs_t.clone()).detach();
            let values2 = agent.critic.forward(obs_t).detach(); // [N, 1]
            let values = values2.reshape([args.num_envs]);      // [N]

            // Sample actions (masked categorical via gumbel-max)
            let actions_t = sample_actions_gumbel::<B>(logits.clone(), mask_t.clone(), &device);

            // logprob + entropy for sampled actions
            let (logp_t, _ent_t) = logprob_and_entropy::<B>(logits, mask_t, actions_t.clone());

            // Pull small tensors back to CPU for env stepping. TensorData supports to_vec/as_slice.
            let actions_vec: Vec<i32> = actions_t.to_data().to_vec().unwrap();
            let logp_vec: Vec<f32> = logp_t.to_data().to_vec().unwrap();
            let values_vec: Vec<f32> = values.to_data().to_vec().unwrap();

            // Store current obs + mask and the sampled action outputs
            // Then step envs
            let step_out = env_pool.step_all(&actions_vec)?;

            for e in 0..args.num_envs {
                let obs_flat = flatten_obs(&cur_obs[e]);

                roll.store_step(
                    t,
                    e,
                    &obs_flat,
                    &cur_mask[e],
                    actions_vec[e],
                    logp_vec[e],
                    values_vec[e],
                    step_out[e].reward,
                    step_out[e].done,
                );

                // Episode stats
                ep_return[e] += step_out[e].reward;
                ep_len[e] += 1;
                if step_out[e].done {
                    recent_returns.push(ep_return[e]);
                    ep_return[e] = 0.0;
                    ep_len[e] = 0;
                }
            }

            // Update current state to next obs/mask
            for e in 0..args.num_envs {
                cur_obs[e] = step_out[e].obs.clone();
                cur_mask[e] = step_out[e].action_mask.clone();
            }
        }

        // Bootstrap values from last obs
        let mut obs_flat_all = vec![0.0f32; args.num_envs * obs_dim];
        for e in 0..args.num_envs {
            let flat = flatten_obs(&cur_obs[e]);
            let base = e * obs_dim;
            obs_flat_all[base..base + obs_dim].copy_from_slice(&flat);
        }
        let obs_last = Tensor::<B, 2>::from_data(
            TensorData::new(obs_flat_all, [args.num_envs, obs_dim]),
            &device,
        );
        let last_v2 = agent.critic.forward(obs_last).detach();
        let last_v = last_v2.reshape([args.num_envs]);
        let last_values: Vec<f32> = last_v.to_data().to_vec().unwrap();

        roll.compute_gae(
            &last_values,
            args.gamma,
            args.gae_lambda,
            args.reward_scale,
            args.standardize_advantages,
        );

        // -------- PPO update (epochs * minibatches) --------
        let mb_size = total_batch / args.num_minibatches;

        let mut all_indices: Vec<usize> = (0..total_batch).collect();

        let mut last_actor_loss = 0.0f32;
        let mut last_critic_loss = 0.0f32;
        let mut last_entropy = 0.0f32;

        for _epoch in 0..args.epochs {
            all_indices.shuffle(&mut rng);

            for mb in 0..args.num_minibatches {
                let start = mb * mb_size;
                let end = start + mb_size;
                let mb_idx = &all_indices[start..end];

                let (obs_mb, act_mb, mask_mb, old_lp_mb, old_v_mb, adv_mb, tgt_mb) =
                    roll.minibatch(mb_idx);

                let obs_t = Tensor::<B, 2>::from_data(
                    TensorData::new(obs_mb, [mb_size, obs_dim]),
                    &device,
                );
                let act_t = Tensor::<B, 1, Int>::from_data(
                    TensorData::new(act_mb, [mb_size]),
                    &device,
                );
                let mask_t = Tensor::<B, 2>::from_data(
                    TensorData::new(mask_mb, [mb_size, action_dim]),
                    &device,
                );
                let old_lp_t = Tensor::<B, 1>::from_data(
                    TensorData::new(old_lp_mb, [mb_size]),
                    &device,
                );
                let old_v_t = Tensor::<B, 1>::from_data(
                    TensorData::new(old_v_mb, [mb_size]),
                    &device,
                );
                let adv_t = Tensor::<B, 1>::from_data(
                    TensorData::new(adv_mb, [mb_size]),
                    &device,
                );
                let tgt_t = Tensor::<B, 1>::from_data(
                    TensorData::new(tgt_mb, [mb_size]),
                    &device,
                );

                // Forward (with grad)
                let logits = agent.actor.forward(obs_t.clone());
                let v2 = agent.critic.forward(obs_t);
                let v = v2.reshape([mb_size]);

                let (new_lp, ent) = logprob_and_entropy::<B>(logits, mask_t, act_t);

                // PPO clipped policy loss:
                // ratio = exp(new - old)
                let ratio = (new_lp.clone() - old_lp_t).exp();
                let clipped = ratio.clone().clamp(1.0 - args.clip_eps, 1.0 + args.clip_eps);

                let surr1 = ratio * adv_t.clone();
                let surr2 = clipped * adv_t;

                // min(surr1, surr2) without a min op: min = (a+b - |a-b|)/2
                let min_surr = (surr1.clone() + surr2.clone() - (surr1 - surr2).abs()) * 0.5;
                let policy_loss = min_surr.mean().neg();

                let entropy_mean = ent.mean();
                let actor_loss = policy_loss - entropy_mean.clone() * args.ent_coef;

                // Clipped value loss:
                let v_clipped = old_v_t.clone() + (v.clone() - old_v_t).clamp(-args.clip_eps, args.clip_eps);
                let l1 = (v.clone() - tgt_t.clone()).powf_scalar(2.0);
                let l2 = (v_clipped - tgt_t).powf_scalar(2.0);

                // max(l1, l2) without max op: max = (a+b + |a-b|)/2
                let max_l = (l1.clone() + l2.clone() + (l1 - l2).abs()) * 0.5;
                let value_loss = max_l.mean() * (0.5 * args.vf_coef);

                let total_loss = actor_loss.clone() + value_loss.clone();

                // Backprop + step (manual training loop pattern in Burn).
                let mut grads = total_loss.backward();
                // let mut grads = GradientsParams::from_grads::<B, Agent<B>>(grads, &agent);

                let grads_actor = GradientsParams::from_module::<B, Actor<B>>(&mut grads, &agent.actor);
                let grads_critic = GradientsParams::from_module::<B, Critic<B>>(&mut grads, &agent.critic);

                agent.actor = actor_optim.step(args.actor_lr, agent.actor, grads_actor);
                agent.critic = critic_optim.step(args.critic_lr, agent.critic, grads_critic);

                last_actor_loss = actor_loss.to_data().to_vec::<f32>().unwrap()[0];
                last_critic_loss = value_loss.to_data().to_vec::<f32>().unwrap()[0];
                last_entropy = entropy_mean.to_data().to_vec::<f32>().unwrap()[0];
            }
        }

        // -------- Logging --------
        if update % 10 == 0 {
            let mean_return = if recent_returns.is_empty() {
                0.0
            } else {
                let k = recent_returns.len().min(100);
                recent_returns[recent_returns.len() - k..].iter().sum::<f32>() / (k as f32)
            };

            let timesteps = (update + 1) * total_batch;
            println!(
                "[upd {:5}] steps={} mean_return(last100)={:8.3} actor_loss={:8.4} value_loss={:8.4} entropy={:8.4}",
                update,
                timesteps,
                mean_return,
                last_actor_loss,
                last_critic_loss,
                last_entropy
            );
        }
    }

    Ok(())
}
