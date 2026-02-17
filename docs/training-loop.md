# PPO Training Loop

This page documents PPO runtime behavior after modular refactor.

For static boundaries, see [architecture.md](architecture.md).
For SPO runtime, see [spo-training-loop.md](spo-training-loop.md).

## Runtime sequence

`src/bin/ppo.rs` -> `PpoArgs::load` -> `TrainingContext::initialize` -> `PpoTrainer::run`.

Core flow per update:

1. rollout collection on `AsyncEnvPool`
2. typed observation adaptation via `ObservationAdapter`
3. action sampling + logprob collection
4. rollout storage (`Rollout`)
5. GAE/targets
6. optimization phase (`PpoOptimizer`)
7. telemetry emit
8. optional deterministic evaluation through shared `evaluation` module

## Key components

- Entrypoint: `src/bin/ppo.rs`
- Trainer wrapper: `src/ppo/train.rs` (`PpoTrainer`)
- Optimizer phase: `src/ppo/train.rs` (`PpoOptimizer`)
- Shared evaluator: `src/evaluation.rs`
- Observation adapter: `src/env_model.rs`
- Rollout buffer: `src/ppo/buffer.rs`
- PPO losses: `src/ppo/loss.rs`

## Determinism and seeds

Keep these offsets unchanged for parity:

- backend seed: `args.seed`
- env-pool base seed: `args.seed`
- initial reset: `args.seed + 10_000`
- eval pool base: `args.seed + 999`
- minibatch shuffle RNG: `StdRng::seed_from_u64(args.seed ^ 0xA11CE)`

## Auto-reset / One-Strike semantics

`done=true` step carries terminal reward but next observation/action mask may already belong to the next episode due to worker auto-reset.

This is expected and must stay consistent in rollout bookkeeping and eval aggregation.

## Performance notes

- Minibatch scalar metrics are accumulated on device, read back once per update.
- Optimization utilities are shared (`training_utils`) to keep behavior aligned with SPO where applicable.

## Edit map

- Sampling/loss behavior: `src/ppo/loss.rs`
- Optimizer phase: `src/ppo/train.rs` (`PpoOptimizer`)
- Eval behavior: `src/evaluation.rs` + PPO eval closure in `src/ppo/train.rs`
- Observation conversion: `src/env_model.rs`
- Rollout storage: `src/ppo/buffer.rs`
