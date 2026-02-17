# rust_rl

High-throughput PPO/SPO training in Rust on top of `rustpool`, with support for `BinPack-v0` and `Maze-v0`.

## Documentation

- [Onboarding](docs/onboarding.md)
- [Architecture](docs/architecture.md)
- [Configuration](docs/configuration.md)
- [PPO Training Loop](docs/training-loop.md)
- [SPO Training Loop](docs/spo-training-loop.md)
- [Telemetry](docs/telemetry.md)
- [Troubleshooting](docs/troubleshooting.md)

## Quickstart

### Prerequisites

- Rust stable toolchain (`cargo`, `rustc`)
- Python tooling reachable by `python3-config` (required by `build.rs`)
- Sibling `rustpool` checkout at `../rustpool`

### Build check

```bash
cargo check
```

### Run PPO

```bash
cargo run --release -- --config ppo_config.yaml
```

### Run SPO

```bash
cargo run --release --bin spo -- --config spo_config.yaml
```

## Current architecture highlights

- Config loaders are per-binary (`PpoArgs`, `SpoArgs`) on top of a shared schema.
- Observation adaptation is trait-based (`ObservationAdapter`) with dense/binpack implementations.
- Evaluation logic is shared in `src/evaluation.rs`.
- Optimization helpers are shared in `src/training_utils.rs`.
- Telemetry bootstrap is centralized in `TrainingContext`.
- Environment creation is registry-driven (`register_env_factory`).

## Config precedence

Runtime precedence:

1. code defaults
2. YAML file (`--config`)
3. explicit CLI flags

Compatibility:

- canonical section: `training_core`
- legacy alias: `ppo_core`
- optional adapter key: `architecture.observation_adapter` (`dense`/`binpack`), with metadata fallback when omitted

## Code pointers

- PPO entrypoint: `src/bin/ppo.rs`
- SPO entrypoint: `src/bin/spo.rs`
- Config + loaders: `src/config.rs`
- Env pool + registry: `src/env.rs`
- Observation adapters: `src/env_model.rs`
- Shared evaluator: `src/evaluation.rs`
- Telemetry context: `src/telemetry.rs`
- PPO trainer: `src/ppo/train.rs`
- SPO trainer: `src/spo/train.rs`

## CUDA behavior

When `device_type=cuda` is requested:

- startup probe validates CUDA/runtime readiness,
- if probe fails, current behavior is to emit diagnostics and panic,
- there is no automatic CPU fallback in current code path.

See troubleshooting for common CUDA checks.
