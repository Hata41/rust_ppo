# Troubleshooting

This guide covers shared runtime contracts plus PPO/SPO-specific failure modes.

Source guides:

- [src/common/runtime/README.md](../src/common/runtime/README.md)
- [src/common/model/README.md](../src/common/model/README.md)
- [src/algorithms/README.md](../src/algorithms/README.md)

## One-Strike policy / auto-reset confusion

Invalid actions can terminate episodes immediately depending on env semantics.

Worker contract:

- when `done=true`, worker resets immediately,
- returned reward belongs to terminal step,
- returned observation/mask can already belong to next episode.

If rollout/eval appears to “jump episodes,” this is expected under auto-reset semantics.

## Shape mismatch issues

Symptoms:

- BinPack parsing failures
- tensor shape mismatch during model forward
- mask/action dimension mismatch

Checks:

- `max_items` / `max_ems` consistency across config + env
- action dim matches model output space
- adapter path uses expected strategy (`dense` / `binpack`)

Adapter-specific checks:

- verify explicit `architecture.observation_adapter` value if set,
- if omitted, verify metadata fallback selects the intended adapter,
- verify train/eval/search all use the same adapter object in a run.

## SPO replay warmup appears stuck

Symptom:

- replay grows but optimizer metrics do not move.

Cause:

- optimization gated by replay warmup (`sample_sequence_length` and replay fill).

Checks:

- confirm replay length increases per rollout,
- reduce `sample_sequence_length` for smoke tests,
- confirm optimization branch is entered.

## Snapshot lifecycle regressions

Symptoms:

- long-run memory growth,
- active state counter anomalies.

Checks:

- root snapshot ids are released after each search/eval call,
- leaf ids are released,
- eval path also releases search ids.

Ownership rule remains:

- `rustpool` stores snapshot clones,
- `rust_rl::AsyncEnvPool` owns active-id accounting discipline.

## Telemetry mismatches

Expected:

- both binaries initialize via shared `TrainingContext`,
- categories remain `TRAINER`, `ACTOR`, `EVALUATOR`, `MISC`,
- evaluator schema keys are compatible across PPO and SPO.

If mismatch appears:

- check binary initialization path,
- check trainer emit fields,
- check formatter/category normalization in `src/common/runtime/telemetry.rs`.

## CUDA startup failures

Current behavior in `src/common/runtime/backend.rs`:

- if CUDA is requested and probe fails, process panics with diagnostic warning.
- there is no automatic CPU fallback in current code path.

Checks:

- verify `CUDA_PATH`/`CUDA_HOME`,
- verify `LD_LIBRARY_PATH` contains driver/toolkit libraries,
- verify requested device index exists.

## Checkpoint load failures

Symptoms:

- startup fails when `--checkpoint-load-path` is provided,
- model initializes but optimizer continuity is missing.

Checks:

- runtime loader accepts `.mpk` only,
- ensure path points to a Burn-native checkpoint file (not `.onnx`),
- for full resume, ensure optimizer files are present in same `step_*` directory,
- for model-only initialization, missing optimizer files are expected and optimizers restart fresh.

## ONNX conversion workflow confusion

Expected workflow:

1. run `convert_onnx` offline,
2. produce Burn artifacts (`.rs` + `.bpk`) and optionally model-only `.mpk`,
3. start trainer with `--checkpoint-load-path` pointing to `.mpk`.

Notes:

- ONNX is not loaded directly inside trainer runtime,
- if tensor mapping is partial during conversion, inspect converter apply summary before training.
