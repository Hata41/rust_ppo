# SPO Training Loop

This page documents SPO runtime behavior after modular refactor.

For static boundaries, see [architecture.md](architecture.md).
For PPO runtime, see [training-loop.md](training-loop.md).

## Runtime sequence

`src/bin/spo.rs` -> `SpoArgs::load` -> `TrainingContext::initialize` -> `SpoTrainer::run`.

Per update:

1. collect rollout transitions from live envs
2. run SMC search from snapshot roots
3. release root + leaf snapshot ids
4. execute selected root action on live env
5. append transition + search targets to replay
6. optimize through `SpoOptimizer` when replay warmup allows
7. emit telemetry
8. optional deterministic search evaluation via shared `evaluation` module

## Key components

- Entrypoint: `src/bin/spo.rs`
- Trainer wrapper: `src/spo/train.rs` (`SpoTrainer`)
- Optimizer phase: `src/spo/train.rs` (`SpoOptimizer`)
- Search: `src/spo/search.rs`
- Replay: `src/spo/buffer.rs`
- MPO losses/duals: `src/spo/loss.rs`
- Shared evaluator: `src/evaluation.rs`
- Observation adapter: `src/env_model.rs`

## Search and snapshot contract

Search consumes root snapshots and emits:

- `root_actions`
- `root_action_weights`
- `sampled_actions`
- `sampled_advantages`
- `leaf_state_ids`

All root and leaf ids must be released by caller paths.

Ownership boundary remains:

- `rustpool::StateRegistry`: snapshot storage backend
- `rust_rl::AsyncEnvPool`: active-id accounting discipline

## Optimization contract

`SpoOptimizer` performs sampled replay updates:

- actor target from search root action weights
- critic targets from sequence-wise GAE-style construction
- dual updates (`temperature`, `alpha`)
- soft target update with `tau`

## Evaluation contract

SPO deterministic eval is still search-based, but now runs through shared evaluator infrastructure with a policy closure that can use `&AsyncEnvPool` for search snapshots/simulations.

## Reproducibility notes

Seed offsets and ordering are preserved in runtime code; when editing update ordering, keep eval RNG and search RNG usage consistent.

## Edit map

- Search behavior: `src/spo/search.rs`
- Replay storage/sampling: `src/spo/buffer.rs`
- Optimization details: `src/spo/train.rs` (`SpoOptimizer`)
- Evaluation closure behavior: `src/spo/train.rs` + `src/evaluation.rs`
- Observation conversion: `src/env_model.rs`
