# RustingStrata

A pure-Rust inference engine for the Qwen3.5/3.6 hybrid model family (`qwen35moe`, `qwen35`) in GGUF. Qwen3-Next
(`qwen3next`), Qwen3.8-Flash-Next (`qwen4exp`), and Qwen3 (`qwen3`, `qwen3moe`) also run. CUDA kernels are compiled
at run time with NVRTC through cudarc (cached as a cubin); the same kernels run on Vulkan (`--backend vulkan`,
GLSL compiled to SPIR-V and committed). There is no C++ build step or ggml dependency. Experts that do not fit in
VRAM run on the CPU while the GPU works, and expert placement adapts to the routing it sees.

Status: v1.2.1. Milestones M0-M6 are done (M6: server, chat, calibration, and bench); the v1 gate run is in
[`docs/m6-results.md`](docs/m6-results.md).

## Numbers

Reference machine: RTX 3060 12 GB, Ryzen 5 7600X, 32 GB RAM. Model: `Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf` (11.7 GiB,
does not fit in VRAM). llama.cpp is the 2026-10-28 build 11876, tuned for the same file
(`-ncmoe 12 -ub 2048 -b 2048`, flash attention; its default ubatch gives ~1550 tok/s 4K prefill).

| | CUDA | Vulkan | llama.cpp |
|---|---|---|---|
| Prefill, 4K prompt | **2216 tok/s** (2169-2252) | 2154 (2140-2162) | ~2236 (2183-2310) |
| Decode after 4K prompt | **113.4 tok/s** (110.9-114.8) | 113.6 | 66.5 |
| Decode, short prompt (like tg128) | **118.2 tok/s** (117.0-118.8) | 117.3 | 68.4 |
| Decode with MTP drafts (`--mtp 1`, greedy), short / after 4K | **up to 160 / 152 tok/s** | 158 / 154 | (its MTP is slower) |
| 32K: prefill / decode | **1800 / 91.9 tok/s** | 1582 / 90.7 | |
| Greedy top-1 agreement with llama.cpp (20 prompts x 32 tokens), either llama.cpp mode | 98-99% | 98-99% | 98.3% (its batch vs token-by-token modes) |

Measured with `strata bench --gpu --mtp 0 --pp 512,4096,32768 --tg 128 --reps 3 --ctx 32896`, mean of 3 rounds
(range in brackets); llama.cpp figures are from `llama-server` timings, with a 4047-token prompt plus 128 tokens,
over 3 runs. MTP gains depend on the text.

Agreement counts a token when it matches llama.cpp run token by token or with the context as one batch; llama.cpp's
own two modes agree on only 98.3%. The remaining misses are near-ties where float rounding order picks a different
token. Expert placement adapts between runs, and the CPU and GPU expert kernels round differently, so greedy output
can change at such near-ties from one run to the next. MTP on and off give identical greedy tokens.

With less VRAM, simulated on the 3060 with `--vram-margin` (not measured on real 8 GB or 6 GB cards): ~8 GB
keeps 116 / 110 tok/s decode (short / after 4K) and 2128 tok/s 4K prefill, ~6 GB 105 / 97 and 2056.

Other models, greedy "The capital of France is" with `--gpu` (CUDA / Vulkan, second run):

| Model | Types | Decode |
|---|---|---|
| Qwen3.5-2B (fine-tune) Q4_K_M | Q4_K, Q6_K, ... | 208 / 208 tok/s |
| Qwen3-4B-Instruct-2507 Q4_K_M (dense) | Q4_K, Q6_K | 104 / 110 tok/s |
| Qwen3.5-9B Q5_K_S | Q5_K, Q6_K, Q8_0 | 51 / 53 tok/s |
| Qwen3-30B-A3B-Instruct-2507 UD-Q2_K_XL | Q2_K to Q6_K | 114 / 123 tok/s |
| Qwen3.8-27B UD-IQ3_XXS (dense, 10.8 GB on GPU) | IQ1_S to IQ4_XS, Q2_K to Q8_0 | 20.6 / 20.5 tok/s |
| Qwen3.6-35B-A3B UD-IQ4_XS (57% of experts fit in VRAM) | IQ3_S, IQ4_XS, Q6_K, Q8_0 | 94 / 96 tok/s |
| Qwen3-Next-80B-A3B-Instruct UD-IQ1_S (most experts on the CPU) | IQ1_S to IQ4_NL, Q4_K to Q8_0, BF16 | 92 / 90 tok/s |
| Qwen3.8-Flash-Next GSQ-RCO IQ2_XS (64 GB in two files; pass the `-00001-of-00002` shard; 18% of experts fit in VRAM, and with 32 GB RAM the rest are partly read from disk) | Q2_0, IQ1_M to IQ4_XS, Q6_K, Q8_0, F16, BF16 | 22 / 20 tok/s |

Supported GGUF types: F32, F16, BF16, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, Q2_0, Q2_K-Q6_K, IQ1_S, IQ1_M, IQ2_XXS,
IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, and IQ4_NL.

## Build

Needs Rust (2024 edition) and either an NVIDIA driver with the CUDA NVRTC library (loaded at run time; CUDA 13
dropped Volta/sm_70, so a V100 needs the 12.x toolkit) or a Vulkan 1.3 driver with shaderFloat64, int8/16/64, and
subgroup size 32 or a fixed size 64 (tested on RTX 3060, Radeon VII, and V100; Intel Arc lacks shaderFloat64).
CUDA kernels run on Volta (sm_70, `m8n8k4` tensor cores), Turing (sm_75), and Ampere or newer (sm_80+,
`cp.async` and int8 tensor cores); Pascal and older use `--backend vulkan`. `--gpu` picks CUDA when it loads, else
Vulkan; `--backend` selects a backend. The first CUDA run compiles the kernels (~15 s) and caches them. For AMD GPU
issues and reporting guidance, see [`docs/AMD.md`](docs/AMD.md). Every GPU that has been tried is listed in
[`docs/SUPPORTED_GPUS.md`](docs/SUPPORTED_GPUS.md).

```sh
cargo build --release
```

Windows: install the Visual Studio C++ Build Tools first (one dependency compiles C code).

## Use

Without a terminal, run `strata` (double-click `strata.exe` on Windows; release binaries are on the GitHub releases
page). It starts the server on `http://127.0.0.1:8080` and opens a chat page in the browser. The page keeps chats in
the browser, downloads models from Hugging Face (Models and settings; set `HF_TOKEN` for gated repos), switches
between models in the models folder (`%LOCALAPPDATA%\rusting_strata\models` on Windows,
`~/.local/share/rusting_strata/models` elsewhere, or the folder set with `serve --models-dir`), and shows context
use, speed, totals, and CPU, memory, and graphics memory use. Other apps that accept an OpenAI-compatible address
(Open WebUI, Jan, SillyTavern, code editors) can use `http://127.0.0.1:8080/v1`.

From a terminal (replace the example model path with your GGUF path). If `strata` is not on `PATH`, run
`target/release/strata` (or `target\release\strata.exe` on Windows) instead:

```sh
M=path/to/Qwen3.6-35B-A3B-UD-Q2_K_XL.gguf

strata calibrate --model "$M"
strata generate --model "$M" --gpu --prompt "Write a haiku about rivers." -n 128
strata chat --model "$M" --gpu --think --temp 0.7
strata bench --model "$M" --gpu --pp 512,4096 --tg 128
strata serve --model "$M" --gpu --addr 127.0.0.1:8080
```

For a split GGUF, point `--model` at the first shard (for example, `model-00001-of-00002.gguf`) and keep all shards
together in the same directory. Strata opens the remaining shards automatically. Set the model path variable to
that first shard, then use it like any other model:

```sh
M=path/to/model-00001-of-00002.gguf
strata generate --model "$M" --gpu --prompt "Hello" -n 128
```

`serve` speaks the OpenAI chat completions API (`POST /v1/chat/completions`, streamed or not, and `GET /v1/models`).
It handles one request at a time by default; `--parallel N` (GPU) decodes up to N requests together (4 requests:
164 tok/s total on the 3060, 41 each; see [`docs/HOSTING.md`](docs/HOSTING.md)). Request `temperature`, `top_k`,
`top_p`, and `seed` are honored; decoding is greedy when they are omitted. `max_tokens` (or
`max_completion_tokens`) and up to 16 `stop` strings work as in the OpenAI API; `n` must be 1. Thinking is off
unless a request sets `chat_template_kwargs.enable_thinking` (Anthropic: `thinking: {"type": "enabled"}`). If a
request extends the previous request's messages, only the new part is prefilled
(`usage.prompt_tokens_details.cached_tokens`). `tools` and `tool_choice` work as in the OpenAI API (calls return as
`tool_calls` with `finish_reason: "tool_calls"`). The prompt follows the model's own chat template, rendered like
llama.cpp's `--jinja`. `tool_choice: "required"` does not force a call.

`POST /v1/messages` speaks the Anthropic Messages API (text, `tool_use` and `tool_result` blocks, streamed or not,
the key in `x-api-key`, and `POST /v1/messages/count_tokens`), so Claude Code can use the server:

```sh
ANTHROPIC_BASE_URL=http://127.0.0.1:8080 ANTHROPIC_API_KEY=secret ANTHROPIC_MODEL=local ANTHROPIC_SMALL_FAST_MODEL=local claude
```

OpenCode and other agents can use the OpenAI address. Qwen3.6's template leaves the think block out of replies
before the last user message, so each new user message changes earlier text and the whole conversation is prefilled
again. Strata keeps those blocks by default, so a long agent session prefills only what was added;
`--preserve-thinking false` (or `chat_template_kwargs.preserve_thinking` in a request) follows the template.

A request with another history (for example, an agent's side request for a title or summary) parks the conversation
state in host memory (`--cache-ram`, 4096 MB by default), so a later request that continues it can reuse the state
instead of prefilling again. For a server with customer keys, a usage log, `/health`, TLS through a reverse proxy,
or systemd, see [`docs/HOSTING.md`](docs/HOSTING.md). Each customer key reuses only its own cached conversations;
all requests under one key count as one user.

### Command-line reference

Run `strata --help` for the command list, or `strata <command> --help` for that command's options. `-h` /
`--help` show help; `-V` / `--version` print the version. Running `strata` without a command starts `serve`, opens
the chat page, and falls back to CPU inference if no GPU opens.

| Command | Purpose |
|---|---|
| `info` | Print model configuration and tensor-type counts. |
| `score` | Score token IDs from a JSON prompt/continuation pair. |
| `chat` | Interactive chat in the terminal. |
| `serve` | Start the HTTP API and chat page; `--model` is optional. |
| `calibrate` | Measure and save GPU tuning values for a model. |
| `check` | Check Vulkan shared-memory requirements without loading weights. |
| `bench` | Measure prefill and decode speed. |
| `perplexity` | Measure perplexity over a text file. |
| `generate` | Generate text from a prompt. |
| `help` | Print help for the program or a subcommand. |

#### Options shared by inference commands

These options are accepted by `score`, `chat`, `serve`, `bench`, `perplexity`, and `generate`.

| Option | Default | Description |
|---|---|---|
| `--gpu` | Off | Run on a GPU; experts that do not fit in VRAM run on the CPU. Without it, use the CPU reference. |
| `--no-mmap` | Off | Read GGUF tensors on demand instead of retaining file mappings; lowers working-set memory. With `--gpu`, CPU-routed decode experts and GPU-prefill staging experts share a RAM cache that rechecks available physical memory and commit headroom as it fills, reserving 5% of each. |
| `--ctx N` | `4096` | GPU context length / KV-cache size, in tokens. |
| `--chunk N` | Calibrated, else `4096` | GPU prefill chunk size in tokens; `0` processes tokens one at a time. |
| `--vram-margin MB` | `512` | VRAM to leave free after placing expert layers. |
| `--mtp [N]` | Calibrated, else `0` | Use N MTP drafts per step on GPU. Bare `--mtp` means `--mtp 1`. |
| `--temp T` | `0` | Sampling temperature; `0` is greedy decoding. |
| `--top-k K` | `0` | Sample from only the K most likely tokens; `0` uses all tokens. |
| `--top-p P` | `1.0` | Nucleus-sampling probability threshold. |
| `--seed N` | Clock | Random seed for sampling. |
| `--threads N` | Calibrated, else logical core count | CPU threads for expert work. |
| `--llama-numerics` | Off | Use llama.cpp-matching GPU numerics for parity checks; attention is much slower. |
| `--fixed-experts` | Off | Keep experts in layer order and do not adapt placement; useful for repeatable parity runs. |
| `--cache-ram MB` | `4096` | Host memory for cached GPU conversation states; `0` disables this cache. |
| `--lora PATH[:SCALE]` | None | Apply a GGUF LoRA adapter; repeat for multiple adapters. Scale defaults to `1`. |
| `--preserve-thinking [BOOL]` | `true` | Keep earlier think blocks in prompts. Use `--preserve-thinking false` to follow the template default. |
| `--kv FORMAT` | `f16` | GPU KV-cache format: `f16` or `q8` (`q8` uses about half the KV memory). |
| `--backend NAME` | CUDA if available, else Vulkan | Select GPU backend: `cuda` or `vulkan`. |
| `--device IDS` | `0` | GPU index or comma-separated indices (for example, `0,1`) to split model layers across GPUs. |

Defaults for `--chunk`, `--mtp`, and `--threads` use the calibration file when available. Run `calibrate` to create
or refresh it; explicitly passing the corresponding option overrides the saved value. MTP is disabled on CPU and
when multiple GPUs split the model. Multiple GPUs split layers by free VRAM; they provide more memory, not twice
the speed. MTP drafting is unavailable for split models.

#### Command-specific options

- `info` — `--model PATH` (required).
- `score` — `--model PATH` and `--input PATH` (required; use `-` to read JSON from stdin), plus shared inference
  options. Input JSON has `prompt` and `continuation` arrays of token IDs.
- `chat` — `--model PATH` (required), `--system TEXT`, `-n N` (reply-token limit, default `2048`), and `--think`
  (enable thinking; off by default), plus shared inference options.
- `serve` — `--model PATH` (optional), `--models-dir PATH`, `--addr HOST:PORT` (default `127.0.0.1:8080`),
  `--open`, `--api-key KEY`, shared inference options, and the hosting options below.
- `calibrate` — `--model PATH` (required), `--ctx N` (default `4096`), `--vram-margin MB` (default `512`),
  `--no-mmap`, `--backend NAME`, `--device N` (default `0`), repeatable `--lora PATH[:SCALE]` (accepted, but calibration uses
  the base model), and `--kv FORMAT` (`f16` or `q8`; default `f16`). Calibration always runs on the GPU.
- `check` — `--model PATH` (required), `--backend NAME`, and `--device N` (default `0`).
- `bench` — `--model PATH` (required), `--pp N[,N...]` (prompt lengths, default `512,4096`), `--tg N` (generated
  tokens per run, default `128`), `--reps N` (runs, default `3`), and `--json-out PATH` (write results as JSON;
  `-` is not allowed), plus shared inference options.
- `perplexity` — `--model PATH` and `--file PATH` (required), `--ctx-len N` (default `512`), `--chunks N` (score
  at most the first N chunks), and `--kld-base PATH` (write llama-perplexity-compatible log probabilities), plus
  shared inference options.
- `generate` — `--model PATH` and `--prompt TEXT` (required; `-` reads the prompt from stdin), `-n N`
  (generated-token limit, default `32`), plus shared inference options.

`serve` also accepts these hosting options:

| Option | Default | Description |
|---|---|---|
| `--api-keys PATH` | None | File of `name key` pairs, one per line; lines beginning with `#` are comments. |
| `--usage-log PATH` | None | Append one JSON record per chat request. |
| `--max-waiting N` | `16` | Maximum open, reading, generating, or queued connections. Excess requests receive `503`. |
| `--max-per-key N` | No per-key limit | Maximum simultaneous requests for each customer key; excess requests receive `429`. |
| `--allowed-hosts HOSTS` | `$STRATA_ALLOWED_HOSTS` | Comma-separated extra `Host` header names accepted by the chat page, model list, and stats. |
| `--max-tokens N` | Context remaining | Maximum reply length, also used when a request does not specify a limit. |
| `--parallel N` | `1` | GPU requests decoded together (valid range `1`-`255`); each uses its own `--ctx` KV-cache slot. |

`--api-key KEY` requires authorization on every request; if omitted, the server reads `STRATA_API_KEY`. The default
models directory is `%LOCALAPPDATA%\rusting_strata\models` on Windows, `$XDG_DATA_HOME/rusting_strata/models` when
set, or `~/.local/share/rusting_strata/models` otherwise. Set `--allowed-hosts` (or `STRATA_ALLOWED_HOSTS`) when
clients reach a server bound to `0.0.0.0` through a LAN hostname. For customer-key isolation, usage logging,
`/health`, TLS, and service setup, see [`docs/HOSTING.md`](docs/HOSTING.md).

On a GPU with no display, `--vram-margin 256` can keep more experts resident (+1-2% prefill); `128` may run out of
memory. `--kv q8` halves KV-cache memory (345 instead of 650 MB at 32K), at about 6% slower 32K decode and 14%
slower 32K prefill. Several GPUs run one after another, so a request gets more memory rather than twice the speed.
AMD cards can clock down while another card works; raising the per-card power level may improve multi-GPU results.

The experts that do not fit in VRAM are read from the GGUF through the page cache. A file system that keeps file data
out of the page cache (ZFS with `primarycache=metadata`) makes every run read them from disk again; load the model
from a file system that caches data, or from a tmpfs copy.

## Tests

```sh
cargo test --release
STRATA_MODEL=path/to/model.gguf cargo test --release
python -I tools/parity.py path/to/model.gguf --extra "--gpu" --tag gpu
```

GPU tests need `STRATA_MODEL` set to a model file. The parity script compares against llama.cpp and needs
`llama-server` available.

## Layout

- [`src/gguf.rs`](src/gguf.rs) — GGUF file reader.
- [`src/quant/`](src/quant/) — Dequantization.
- [`src/model.rs`](src/model.rs) — CPU reference forward.
- [`src/cuda/`](src/cuda/) — CUDA kernels.
- [`src/vulkan/`](src/vulkan/) — Vulkan shaders and kernels.
- [`src/backend.rs`](src/backend.rs) — Backend interface.
- [`src/gpu_model.rs`](src/gpu_model.rs) — GPU forward, MTP, and verification.
- [`src/gpu_model/prefill.rs`](src/gpu_model/prefill.rs) — Chunked prefill.
- [`src/expert_cache.rs`](src/expert_cache.rs) — Expert placement.
- [`src/cpu_moe.rs`](src/cpu_moe.rs) — CPU expert compute.
- [`src/cpu_worker.rs`](src/cpu_worker.rs) — CPU expert worker.
- [`src/serve.rs`](src/serve.rs) — HTTP server.
- [`src/chat.rs`](src/chat.rs) — Chat templates and tool calls.
- [`src/main.rs`](src/main.rs) — CLI.

## License

RustingStrata is released under the [RustingStrata License 1.0](LICENSE.md).
You may use, modify and distribute it, including for commercial services, as
long as you keep the license and credit Vasyl Trefilov as the original author.
You may not sell RustingStrata itself.
