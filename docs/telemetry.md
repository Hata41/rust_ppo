# Telemetry

This project uses a dual telemetry pipeline for PPO and SPO:

- **Tracing (OTLP):** exports spans/events to MLflow OTLP (`/v1/traces`) for execution timelines.
- **Metrics (REST):** exports scalar metrics to MLflow REST (`/api/2.0/mlflow/runs/log-batch`) for line charts.

Both pipelines are active in parallel from the same `tracing` events.

Shared modules:

- formatter and category rendering: [src/telemetry.rs](../src/telemetry.rs)
- PPO wiring: [src/bin/ppo.rs](../src/bin/ppo.rs)
- SPO wiring: [src/bin/spo.rs](../src/bin/spo.rs)

## Categories

Structured metrics are emitted under these categories:

- `TRAINER`
- `ACTOR`
- `EVALUATOR`
- `MISC`

Category names are normalized by formatter-level mapping.

## MLflow metrics key mapping

Metrics are automatically prefixed from the `category` field before being sent to MLflow:

- `TRAINER` / `TRAIN` → `trainer/`
- `ACTOR` / `ACT` → `actor/`
- `EVALUATOR` / `EVAL` → `evaluator/`
- default → `misc/`

Example:

- `tracing::info!(category="TRAINER", critic_loss=0.5, timesteps=100, "train")`
- exported metric key: `trainer/critic_loss` (step `100`)

Notes:

- only numeric fields are exported as metrics,
- `category`, `message`, `telemetry` are never exported as metric keys,
- events without an explicit step field are skipped by the metrics exporter.

Accepted step fields:

- `timesteps`
- `policy_version`
- `step`
- `global_step`
- `update`

## Non-blocking metrics architecture

Metrics export never performs HTTP inside `on_event`:

- `MlflowMetricsLayer` extracts numeric fields synchronously.
- Metrics are pushed into a `crossbeam_channel`.
- A background worker batches and POSTs to MLflow REST.

Batching behavior:

- flush every ~1 second, or
- flush when batch size reaches 50.

Reliability behavior:

- retry transient failures up to 3 attempts,
- request timeout is short (3s),
- warning logs are rate-limited.

## Run ID behavior (metrics)

Metrics require an MLflow `run_id`.

- if `--mlflow-run-id` is provided, it is used,
- otherwise the binary attempts `runs/create` automatically,
- if auto-create fails, metrics export is disabled (tracing still works).

Environment override used for run creation:

- `MLFLOW_EXPERIMENT_ID` (default: `0`).

## Dynamic metric labels

Metric labels come from `MetricRegistry` defaults plus optional runtime overrides.

Environment variable format:

```bash
RUST_RL_METRIC_LABELS="global_grad_norm=Grad Norm,steps_per_second=SPS"
```

Rules:

- comma-separated `key=Label` pairs
- unknown keys fall back to title-cased rendering

## Emission producers

- PPO train/eval emissions: [src/ppo/train.rs](../src/ppo/train.rs)
- SPO train/eval emissions: [src/spo/train.rs](../src/spo/train.rs)

## PPO/SPO schema compatibility contract

PPO and SPO are expected to remain log-schema compatible for shared monitoring.

Note:

- some keys can be emitted as compatibility placeholders in one algorithm (for example when a metric is not semantically central to that method), but key presence and category contracts should remain stable for downstream consumers.

### Shared evaluation schema

Both should emit in `EVALUATOR` category:

- `phase`
- `policy_version`
- `episodes`
- `mean_return`
- `max_return`
- `min_return`
- `episode_length_mean`
- `episode_length_max`
- `episode_length_min`
- `duration_ms`

### Shared runtime categories per update

Per update, both algorithms should emit:

- `TRAINER` (optimization metrics)
- `ACTOR` (episodic performance aggregates)
- `MISC` (throughput/runtime metrics)

## Cadence expectations

- PPO: emits train/update logs per update and deterministic eval on configured cadence.
- SPO: emits train/update logs per update and deterministic search-based eval on configured cadence.

Eval cadence contract:

- run when `eval_interval > 0`
- run when `update % eval_interval == 0`
- require `num_eval_episodes > 0`

See runtime sources for exact emission points:

- [training-loop.md](training-loop.md) (PPO)
- [spo-training-loop.md](spo-training-loop.md) (SPO)

## Filtering

Filtering precedence:

- If `RUST_LOG` is set, it is used directly.
- Otherwise, defaults come from YAML `logging` section (`log_level`, `backend_logs_visible`).

Default config behavior keeps CubeCL/CUDA backend context logs hidden to reduce dashboard noise.

## MLflow over reverse SSH (repeatable runbook)

Use this when training runs on a remote host (e.g. `BareMetal`) and MLflow runs on your laptop.

### Required runtime wiring

- OTLP endpoint must be `http://localhost:5000/v1/traces` on the remote training host.
- MLflow OTLP header must be present: `x-mlflow-experiment-id` (configured in `src/telemetry.rs`).
- Metrics endpoint is derived automatically from the base URI: `http://localhost:5000/api/2.0/mlflow/runs/log-batch`.

### SSH config (laptop)

Example `~/.ssh/config` entry:

```ssh_config
Host REMOTE_TRAINING_HOST
	HostName YOUR_REMOTE_HOST_OR_IP
	IdentityFile ~/.ssh/YOUR_PRIVATE_KEY
	User YOUR_REMOTE_USER
	Port YOUR_SSH_PORT
	RemoteForward 5000 localhost:5000
```

### Every-run command order

1) **Laptop / Terminal A**: start MLflow first

```bash
uv run mlflow server --host 127.0.0.1 --port 5000
```

2) **Laptop / Terminal B**: keep reverse tunnel open

```bash
ssh -N BareMetal
```

Optional debug mode:

```bash
ssh -vvv -N BareMetal
```

3) **Remote (`BareMetal`)**: verify tunnel endpoint

```bash
curl -i http://localhost:5000/
```

Expected: `HTTP/1.1 200 OK`.

4) **Remote (`BareMetal`)**: run training

```bash
cargo run --bin ppo -- --config ppo_config.yaml
cargo run --bin spo -- --config spo_config.yaml
```

### Quick diagnostics

- `405 Method Not Allowed` at `http://localhost:5000/`:
	endpoint path is wrong; use `/v1/traces`.
- `Connection refused` to `http://localhost:5000/v1/traces`:
	tunnel or local MLflow is down.
- `ssh -N BareMetal` prints `connect_to localhost port 5000: failed`:
	local MLflow is not listening yet; start MLflow first, then reconnect SSH.

### Optional direct probe of OTLP route

From `BareMetal`, this checks that POST reaches the traces route and header is accepted:

```bash
curl -X POST 'http://localhost:5000/v1/traces' \
	-H 'x-mlflow-experiment-id: 0' \
	-H 'Content-Type: application/x-protobuf' \
	--data-binary '' -i
```

Expected: not `404`/`405`; a `400` with empty payload is acceptable for this probe.

## Maintenance checklist for telemetry changes

When modifying telemetry:

- update both trainer emitters if schema keys are shared,
- keep category names stable,
- keep formatter in shared module,
- update this document and run a quick PPO+SPO smoke check.
