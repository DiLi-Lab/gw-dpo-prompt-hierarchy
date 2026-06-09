# Prompt Hierarchy

Code accompanying the paper *"Prompt Hierarchy: Training LLMs to Enforce Multi-Level Instruction Hierarchies via Gravity-Weighted DPO"*.

This repository reproduces every dataset, training run, and evaluation reported in the paper. The pipeline trains Llama-3.1-8B-Instruct to enforce a five-level instruction hierarchy

```
L0  Platform Governance      (highest privilege)
L1  Developer System Prompt
L2  Per-User Configuration
L3  User Messages
L4  Data / Tool Outputs      (lowest privilege)
```

using twelve special delimiter tokens, Instructional Segment Embeddings (ISE), and Gravity-Weighted DPO (GW-DPO) with a curriculum that escalates the per-pair margin proportionally to the hierarchy gap between attacker and victim levels.

## Repository Layout

```
prompt-hierarchy-code/
├── bin/            CLI entry points (one per pipeline stage)
├── configs/        Training / evaluation YAML configurations
├── data/           Datasets — content libraries and SFT / DPO / eval
│                    splits + token-length stats are tracked
├── src/            Library code (data construction, model
│                    modifications, training, evaluation)
├── tests/          Unit and integration tests
├── requirements.txt
├── .env.example
└── README.md
```

Locations that appear under the repo root after running the pipeline but are **git-ignored** (regenerated or user-fetched, never committed):

| Path | Populated by | Contents |
|---|---|---|
| `models/` | Steps 1, 6, 7 | Base-with-tokens checkpoint, SFT / DPO training runs (`models/runs/`), HP sweep (`models/hp_search/`), merged production / ablation checkpoints |
| `evaluation/` | Steps 8, 9 | 5-level suite outputs (`evaluation/runs/`) and external-benchmark outputs (`evaluation/external/`) |
| `data/splits/` | Step 0 | HuggingFace Alpaca / Dolly download caches |
| `data/external/` | Step 9 | XSTest / SEP / MT-Bench / TensorTrust upstream data |
| `vendor/` | Step 9 | Third-party benchmark code — currently just the IHEval clone |

Only the entry points to regenerate or fetch these are tracked.

## Setup

### 1. Create a Python environment

```bash
conda create -n prompt-hierarchy python=3.11 -y
conda activate prompt-hierarchy
```

### 2. Install PyTorch

Install via conda so the correct CUDA / MPS backend is selected:

```bash
conda install pytorch torchvision -c pytorch -y
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Hugging Face access

The base model is Meta's [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct), which is gated. Request access on the model page, then authenticate locally:

```bash
python -c "from huggingface_hub import login; login()"
```

### 5. API keys

Dataset construction and judge-based evaluation use third-party LLM APIs. Copy `.env.example` to `.env` and provide:

| Variable | Used by |
|---|---|
| `ANTHROPIC_API_KEY` | L0 / L1 / DPO Phase 3 / repair passes (Claude Sonnet 4) |
| `OPENAI_API_KEY` | DPO Phase 2, eval-suite construction, PPA / MT-Bench / XSTest judge (GPT-4o, GPT-4o-mini) |
| `GOOGLE_CLOUD_PROJECT` | Dual-judge QC during DPO and eval-suite construction (Gemini 2.5 Pro via Vertex AI) |
| `TOGETHER_API_KEY` | Optional fallback for Llama serving |

### 6. Google Cloud (Vertex AI)

Gemini judge calls use Vertex AI. Configure Application Default Credentials once:

```bash
gcloud auth application-default login
```

Enable the Vertex AI API on the project named in `GOOGLE_CLOUD_PROJECT`.

### 7. External benchmarks (deferred to Step 9)

This repository **does not redistribute** any external benchmark data or third-party benchmark code. Before you run Step 9, you will need to fetch IHEval, XSTest, MT-Bench, SEP, and TensorTrust from their respective upstream sources. The complete set of download / build instructions lives in [Step 9 — External benchmarks](#step-9--external-benchmarks); you can skip ahead to that section now if external benchmarking is your primary interest.

## Hardware

| Stage | Practical minimum | Used in paper |
|---|---|---|
| Dataset construction | CPU + internet | CPU |
| Model setup (Step 1) | 1× A100 40 GB | 1× A100 80 GB |
| SFT (Step 6) | 1× A100 80 GB | 1–2× A100 80 GB, ~2 GPU-h |
| GW-DPO (Step 7) | 2× A100 80 GB | 2–4× A100 80 GB, ~8–12 GPU-h per configuration |
| Evaluation suite (Step 8) | 1× A100 40 GB | 1× A100 80 GB |
| External benchmarks (Step 9) | 1× A100 40 GB | 1× A100 80 GB |

The full HP search (Step 7a) is the dominant cost: ~21 GPU-hours under the axial design, ~100 GPU-hours under the full 12-config grid.

## Configuration model

The project ships **two headline production configurations**, corresponding to the two margin schedules reported in the paper:

| Production config | Margin schedule | Per-pair offset δ(i, j) |
|---|---|---|
| `configs/base_linear.yaml` | linear (`dpo.margin_schedule: "gap"`) | δ = j − i |
| `configs/base_bilateral.yaml` | bilateral (`dpo.margin_schedule: "bilateral"`) | δ = (j − i)·(k − 1 − i), k = 5 |

`configs/base_linear.yaml` is the default config that every CLI loads when `--config` is omitted; benchmark configs (`xstest.yaml`, `iheval.yaml`, `sep.yaml`, `mt_bench.yaml`, `tensortrust.yaml`) chain-load it for `ModelConfig`/`SFTConfig`/etc. `configs/base_bilateral.yaml` overrides only `dpo.margin_schedule` (and is otherwise identical to the linear baseline for apples-to-apples comparison).

Ablation variants override only the fields that differ from `configs/base_linear.yaml`:

| Ablation config | Difference from `base_linear.yaml` |
|---|---|
| `ablation_c_standard_dpo.yaml` | `dpo.gravity_alpha: 0.0` (standard DPO, no gravity weighting) |
| `ablation_e_3level.yaml` | 3-level dataset + `num_curriculum_stages: 2` |
| `ablation_f_tokens_only.yaml` | Tokens added but ISE disabled |

Every CLI script accepts `--override section.key=value` (repeatable). For example:

```bash
python bin/train_sft.py --override sft.num_epochs=1 sft.learning_rate=1e-5
```

The override path is validated against the typed configuration dataclasses, so a typo fails fast at startup.

## Files that get overwritten

Dataset construction scripts **overwrite their output files in place** on each run. In particular:

* `bin/build_libraries.py l1 / l4` overwrite `data/libraries/l1_library.json` / `l4_library.json`.
* `bin/build_sft_dataset.py` overwrites `data/sft/{train,val}/sft_combined.jsonl`.
* `bin/build_dpo_dataset.py` overwrites the phase outputs and `data/dpo/{train,val}/dpo_combined.jsonl`.
* `bin/build_eval_suite.py` overwrites `data/eval/eval_{conflicts,aligned,reference}.jsonl`.

Several phases support a `--resume` flag that keeps cached generations; without `--resume` a re-run starts from scratch and overwrites prior artefacts. Back up any dataset you want to keep before re-running.

---

## Pipeline overview

```
Step 0   Download base datasets (Alpaca, Dolly)
Step 1   Add 12 delimiter tokens, initialise ISE
Step 2   Build L0–L4 content libraries + injection templates
Step 3   Construct SFT dataset (train + val)
Step 3.5 Validate token lengths, set max_seq_length
Step 4   Construct DPO dataset (five phases, train + val)
Step 4e  Validate DPO token lengths
Step 5   Build evaluation suite (1 752 instances)
Step 6   SFT training (LoRA + trainable token rows + ISE)
Step 7   Gravity-Weighted DPO training (3-stage curriculum + sDPO)
Step 8   Evaluation on the 5-level suite (PPA / WHS / ORR / UtilityΔ)
Step 9   External benchmarks (XSTest, IHEval, SEP, MT-Bench, TensorTrust)
```

Each step writes a self-contained artefact that the next step consumes; intermediate files can be inspected without running the rest of the pipeline. Most scripts accept `--dry-run` (preview costs and counts), `--split {train,val}`, `--resume`, and `--override section.key=value`.

---

## Step 0 — Base datasets

Downloads Alpaca-Cleaned (~52 K examples) and Dolly-15K from Hugging Face and writes 85 / 15 train / eval splits with seed 42:

```bash
python bin/download_base_datasets.py            # download + split
python bin/download_base_datasets.py --validate # validate existing splits
```

Output:

| Path | Examples |
|---|---|
| `data/splits/alpaca_train/` | ~44 K |
| `data/splits/alpaca_eval/` | ~7.8 K |
| `data/splits/dolly_train/` | ~12.7 K |
| `data/splits/dolly_eval/` | ~2.3 K |

## Step 1 — Model setup

Loads Llama-3.1-8B-Instruct, adds twelve hierarchy delimiter tokens (`<|L0_START|>`, `<|L0_END|>`, …, `<|L4_END|>`, `<|RESP_START|>`, `<|RESP_END|>`), resizes the embedding matrix with mean-of-existing initialisation, and writes ISE weight tensors:

```bash
CUDA_VISIBLE_DEVICES=0 python bin/setup_model.py
```

If model loading fails with `Cannot allocate memory`, raise the virtual memory limit (per shell):

```bash
ulimit -v unlimited
```

Output: `models/tokenizer-5level/`, `models/base-with-tokens/`, `models/ise_weights_init.pt`.

## Step 2 — Content libraries

Build the level-specific content pools that downstream dataset construction draws from. All sub-steps live behind one CLI:

```bash
python bin/build_libraries.py <level> [--validate] [...]
```

with `<level>` one of `l0`, `l1`, `l2`, `l3`, `l4`, `injection`. L2 and L3 are pure samplers (no persistent file). The L0 conflict-scenario library is generated by a separate script.

Approximate runtime (each step records the actual numbers in stdout):

| Sub-step | Output | Notes |
|---|---|---|
| 2a L0 rules | `data/libraries/l0_rules.json` | LLM-assisted expansion + manual curation; ~150–170 rules across 5 categories |
| 2b L1 prompts | `data/libraries/l1_library.json` | Claude Sonnet 4, 15 domains; dedup by sentence-transformer cosine > 0.85 |
| 2c L2 configs | (in-memory) | Template instantiation across 7 attribute categories |
| 2d L3 messages | (in-memory) | Filtered from Alpaca / Dolly instructions (5 ≤ word count ≤ 500) |
| 2e L4 tool outputs | `data/libraries/l4_library.json` | Wrap existing Alpaca/Dolly fields + synthesise ~15 K missing entries via GPT-4o-mini |
| 2f Injection templates | (validate only) | `data/libraries/injection_templates.json` is handcrafted and version-controlled |
| 2g L0 conflict scenarios | `data/libraries/l0_conflict_scenarios.json` | Run `python bin/generate_l0_conflict_scenarios.py`; manual review required |

Important flags (apply across sub-steps):

| Flag | Effect |
|---|---|
| `--expand` | Trigger LLM expansion for L0 seed rules |
| `--validate` | Run schema / coverage / diversity checks without API calls |
| `--skip-dedup` | Skip embedding-based deduplication (inspect raw outputs) |
| `--skip-synthesis` | L4 only: wrap existing fields without GPT-4o-mini synthesis |
| `--max-synthesis N` | L4 only: cap synthesis to N examples |
| `--domains <list>` | L1 only: restrict to specific task domains |
| `--batches-per-domain N` | L1 only: short iteration runs |
| `--seed-file PATH` / `--rules-file PATH` / `--library-file PATH` / `--templates-file PATH` | Override default input paths |

Recommended execution order:

1. Hand-author `data/libraries/L0_seed_rules.json` (~20 seeds).
2. `python bin/build_libraries.py l0 --expand` → curate `L0_rules_expanded.json` → save as `data/libraries/l0_rules.json` → `--validate`.
3. `python bin/build_libraries.py l1 --domains coding --batches-per-domain 2` for a small pilot, inspect, then full `l1` run.
4. `python bin/build_libraries.py l2 --validate` and `python bin/build_libraries.py l3 --validate`.
5. `python bin/build_libraries.py l4 --skip-synthesis` to sanity-check wrapping, then `--max-synthesis 15000` for the production run.
6. `python bin/build_libraries.py injection` to validate the handcrafted templates.
7. `python bin/generate_l0_conflict_scenarios.py` to produce the L0-conflict library used by DPO Phase 3.

## Step 3 — SFT dataset

Builds train (~10 K) and val (~1.7 K) splits stratified across three categories — *aligned*, *partial*, *misaligned* — using disjoint partitions of the base datasets so no row appears in both splits or in both SFT and DPO.

```bash
python bin/build_sft_dataset.py             # both splits
python bin/build_sft_dataset.py --dry-run   # preview, no API calls
```

Flags:

| Flag | Effect |
|---|---|
| `--split {train,val}` | Build only one split |
| `--skip-synthesis` | Use simple assembly for aligned examples (no GPT-4o context synthesis) |
| `--aligned-count / --synthesis-count / --partial-count / --misaligned-count` | Override per-category counts |

Output: `data/sft/{train,val}/sft_combined.jsonl`. Train split composition: 7 000 aligned (5 000 simple + 2 000 synthesised) / 2 000 partial / 1 000 misaligned; val mirrors the same ratios at ~17 % of the size.

## Step 3.5 — Token-length validation

Truncation that breaks delimiter pairs corrupts ISE segment IDs. This validator tokenises every example, reports the length distribution, verifies that each `<|Li_START|>` has a matching `<|Li_END|>`, and recommends `max_seq_length`:

```bash
python bin/validate_lengths.py sft
python bin/validate_lengths.py sft --split train
python bin/validate_lengths.py dpo --split val
```

Stats are written to `data/stats/{train,val}/{sft,dpo}_length_stats.json`. Update `configs/base_linear.yaml` (`sft.max_seq_length` / `dpo.max_seq_length`) before training; the bilateral config inherits these values implicitly when callers chain-load. If a small number of outliers push the maximum past the next power of two, prefer removing those examples to doubling `max_seq_length` for the whole run.

## Step 4 — DPO dataset

The DPO dataset (~10 K train pairs, ~1 K val pairs) covers all ten pairwise conflicts, over-refusal calibration, and cascading multi-level conflicts. Construction proceeds in five phases plus two repair passes; some prerequisites require separate scripts that are not driven by `build_dpo_dataset.py`.

Required libraries (all produced in Step 2):

| Library | Needed by |
|---|---|
| `l0_rules.json` | All phases |
| `l1_library.json` | All phases |
| `l4_library.json` | All phases |
| `injection_templates.json` | Phases 1–3 |
| `l0_conflict_scenarios.json` | Phase 3 (Step 4a) |
| `l0_adversarial_instructions.json` | Phase 2 (Step 4b) |
| `cascading_families_generated.json` | Phase 3 (Step 4c) |

### Step 4a — L0 conflict scenarios

```bash
python bin/generate_l0_conflict_scenarios.py            # generates ~100 + ~25 scenarios
python bin/generate_l0_conflict_scenarios.py --dry-run  # preview only
```

Open the JSON file and verify each scenario — the adversarial L1 should clearly contradict its targeted L0 rules, and L3 templates should be specific requests that trigger the conflict. Remove or edit ambiguous scenarios before continuing.

### Step 4b — L0 adversarial instructions

```bash
python bin/generate_l0_adversarial_instructions.py --count-per-category 50
```

Produces ~200–300 genuinely L0-violating instructions across all categories. Inspect for quality and coverage before continuing.

### Step 4c — Cascading conflict families

Cascading conflicts involve three or more hierarchy levels interacting at once. Seven *seed* families are hard-coded in `src/data/dpo/cascading.py`; an additional 13–15 are generated iteratively:

```bash
# Round 1 — generate 15 candidate families
python bin/generate_cascading_families.py

# Round 2+ — reviewed families are kept verbatim; new candidates are appended
python bin/generate_cascading_families.py \
    --accepted data/libraries/cascading_families_generated.json

# Validate the final set
python bin/generate_cascading_families.py \
    --validate data/libraries/cascading_families_generated.json
```

`--accepted` preserves already-reviewed families at the top of the output; review only the new entries appended at the bottom and repeat until 13–15 families pass. The validator catches chain / template mismatches, insufficient variable diversity, and chain duplication against seed families.

### Step 4d — Run DPO phases

```bash
python bin/build_dpo_dataset.py             # all five phases, both splits
python bin/build_dpo_dataset.py --dry-run   # print pair table and exit
```

Phases can be run individually with `--phase N` and they write to disjoint output files; Phase 4 requires Phases 1–3 to be complete.

| Phase | Pair types | Required keys |
|---|---|---|
| 1 | L1-vs-L3 (1 500 pairs) | `OPENAI_API_KEY` (L2 generation, skippable with `--skip-l2-generation`) |
| 2 | L0/L1/L2/L3/L4 pairwise + over-refusal calibration (~7 500 pairs) | `OPENAI_API_KEY` |
| 2.5 | Repair Phase 2 outputs via Claude Sonnet 4 | `ANTHROPIC_API_KEY` |
| 3 | L0-vs-L1 / L0-vs-L2 / L0-vs-L4 / cascading (~2 500 pairs) | `OPENAI_API_KEY` + `ANTHROPIC_API_KEY` |
| 3.5 | Repair Phase 3 outputs | `ANTHROPIC_API_KEY` |
| 4 | Combine + deduplicate | — |
| 5 | Dual-judge QC on a 15 % stratified sample | `OPENAI_API_KEY` + `GOOGLE_CLOUD_PROJECT` |

Repair phases are invoked separately when running an individual phase:

```bash
python bin/fix_yl_refusals.py --phase 2 --split train --fix-weak-yl --skip-calibration
python bin/fix_yl_refusals.py --phase 3 --split train --fix-weak-yl --fix-yw
```

The full pipeline (`python bin/build_dpo_dataset.py` without `--phase`) runs Phases 2.5 and 3.5 automatically. After Phase 3.5, review the `*_needs_regeneration.jsonl` and `*_needs_manual_review.jsonl` files before continuing to Phase 4.

Other flags worth knowing:

| Flag | Effect |
|---|---|
| `--split {train,val}` | Build only one split (val skips Phase 5) |
| `--resume` | Reuse caches from a previous run |
| `--pair <PAIR> --count N` | Build a single pair type (debugging) |
| `--skip-qc` | Skip Phase 5 dual-judge evaluation |
| `--skip-l2-generation` | Skip GPT-4o-mini L2 attribute generation in Phase 1 |

Output: `data/dpo/{train,val}/dpo_combined.jsonl`, plus per-phase JSONL files in the same directory.

Train pair budget (val mirrors at ~10 %):

| Pair | Count | | Pair | Count |
|---|---|---|---|---|
| L0_vs_L1 | 500 | | L2_vs_L3 | 500 |
| L0_vs_L2 | 500 | | L2_vs_L4 | 500 |
| L0_vs_L3 | 500 | | L3_vs_L4 | 1 000 |
| L0_vs_L4 | 500 | | Calibration | 2 000 |
| L1_vs_L2 | 500 | | Cascading | 1 000 |
| L1_vs_L3 | 1 500 | | | |
| L1_vs_L4 | 1 000 | | **Total** | **~10 000** |

### Step 4e — Validate DPO token lengths

```bash
python bin/validate_lengths.py dpo
```

Same semantics as Step 3.5; update `dpo.max_seq_length` in `configs/base_linear.yaml` (and `configs/base_bilateral.yaml` if it ever needs to diverge) from the recommended value.

## Step 5 — Build the evaluation suite

Six-phase pipeline that produces ~2 300 instances: 1 000 conflict scenarios, 1 000 matched aligned controls, and 300 flat-text reference baselines.

```bash
python bin/build_eval_suite.py              # all phases
python bin/build_eval_suite.py --dry-run    # preview counts and exit
python bin/build_eval_suite.py --resume     # resume from cached progress
python bin/build_eval_suite.py --phase N    # run a single phase
```

| Phase | What it does | Required keys |
|---|---|---|
| 1 | Generate 1 000 conflict scenarios (GPT-4o) + gold responses (Claude Sonnet 4) | `OPENAI_API_KEY` + `ANTHROPIC_API_KEY` |
| 3 | Build 1 000 aligned controls | `OPENAI_API_KEY` + `ANTHROPIC_API_KEY` |
| 4 | Build 300 reference baselines (flat text, zero API cost) | — |
| 5 | Dual-judge QC on conflict scenarios | `OPENAI_API_KEY` + `GOOGLE_CLOUD_PROJECT` |
| 6 | Final assembly, near-dedup, output write | — |

Flags: `--skip-qc`, `--skip-near-dedup`, `--count N` (per pair, useful for tests), `--seed 42`.

Output:

| File | Count | Description |
|---|---|---|
| `data/eval/eval_conflicts.jsonl` | ~1 000 | Conflict scenarios (100 × 10 pairs) |
| `data/eval/eval_aligned.jsonl` | ~1 000 | Matched controls (1:1 with conflicts) |
| `data/eval/eval_reference.jsonl` | 300 | Flat-text reference baselines |
| `data/eval/eval_stats.json` | — | Counts, discard rates, judge score distribution |

## Step 6 — SFT training (LoRA + ISE)

Teaches the model to recognise the delimiter tokens and integrate ISE segment information. Hierarchy compliance is *not* learned here; that is GW-DPO's job.

```bash
python bin/train_sft.py                                # 3 epochs from configs/base_linear.yaml
python bin/train_sft.py --config path/to/custom.yaml   # alternative config
python bin/train_sft.py --override sft.num_epochs=1 sft.per_device_batch_size=2
CUDA_VISIBLE_DEVICES=0 python bin/train_sft.py         # pin to one GPU
```

The SFT stage is shared between the two production runs — both `configs/base_linear.yaml` and `configs/base_bilateral.yaml` carry identical `sft:` sections, so a single SFT run feeds both DPO variants.

What happens during training:

1. Loads the model from `models/base-with-tokens/` and tokenizer from `models/tokenizer-5level/`.
2. Applies LoRA (rank 64) to every linear projection and marks the 12 special-token rows as trainable via `trainable_token_indices`.
3. Wraps the model with the ISE layer initialised from `models/ise_weights_init.pt`.
4. Trains with completion-only loss — only response tokens after `<|RESP_START|>` contribute.
5. Evaluates on the validation split every 50 steps, saves checkpoints, and persists the best checkpoint (by `eval_loss`) to `models/runs/sft_<ts>/best-checkpoint/`.

Default hyperparameters (in `configs/base_linear.yaml`, section `sft`; identical in `configs/base_bilateral.yaml`):

| Parameter | Value |
|---|---|
| Learning rate | 2 × 10⁻⁵ (cosine schedule, 3 % warmup) |
| Epochs | 3 |
| Effective batch size | 32 (2 per device × 16 gradient accumulation) |
| Max sequence length | 4 096 |
| Precision | bf16 |
| LoRA rank / α | 64 / 128 |
| LoRA targets | q/k/v/o, gate/up/down projections |
| Eval / checkpoint interval | every 50 steps |

After training, merge the best checkpoint (LoRA adapter + the 12 trainable special-token rows) into the base model:

```bash
python bin/merge_sft.py                                   # latest sft_*/best-checkpoint
python bin/merge_sft.py --sft-checkpoint <explicit/path>
```

The merge is idempotent and matches what `bin/train_dpo.py` performs internally between Phases 1 and 2. Output: `models/llama-3.1-8b-sft-merged/`. This artefact serves three roles: (i) DPO policy initialisation, (ii) DPO reference policy, and (iii) the SFT-only ablation row of the 5-level evaluation.

## Step 7 — Gravity-Weighted DPO

Three-stage curriculum DPO with sDPO reference updates between stages. The per-sample margin is `δ = α · (j − i)` by default, where `i` and `j` are the victim and attacker levels.

### Step 7a — Hyperparameter search (ρ × β)

Sweeps `ρ = α / β` and `β` to identify the strongest configuration on a 1 000-pair held-out cut of the validation set. Two modes are supported:

* **Fast axial mode (`dpo.curriculum_enabled=false`, recommended for ranking).** Six configs total: fix `ρ = 1.0`, sweep `β ∈ {0.05, 0.1, 0.2}`; then fix `β` at the winner and sweep `ρ ∈ {0.5, 2.0, 3.0}`. Approximate cost: ~21 GPU-hours.
* **Full grid under the production curriculum.** All 12 (ρ, β) combinations under the 3-stage curriculum; matches the deployment regime exactly. Approximate cost: ~100 GPU-hours.

```bash
# 1. Build the held-out HP-select cut (idempotent)
python bin/build_hp_select_split.py

# 2. Run the sweep
python bin/train_dpo_hp_search.py                                # full grid
python bin/train_dpo_hp_search.py --configs "4-6" \              # axial sweep 1
    --override dpo.curriculum_enabled=false
python bin/train_dpo_hp_search.py --configs "2,8,11"             # axial sweep 2 (β fixed)

# 3. Inspect the winner
cat models/hp_search/best_config.json
cat models/hp_search/results_summary.md
```

The sweep is resumable: configs with an existing `hp_eval.json` are skipped on re-run; mid-training crashes resume from the final-stage `best-checkpoint/`. Results carry a `curriculum_enabled` field so fast-mode and full-curriculum rows are distinguishable in `results.jsonl`. Pass `--configs N-M` or `--configs "i,j,k"` to run disjoint subsets in parallel across GPUs.

Artefacts land under `models/hp_search/` and do not touch production training outputs.

### Step 7b — Production runs with the winning hyperparameters

The paper reports **two production GW-DPO runs** trained from the same SFT checkpoint and the same DPO data, differing only in the margin schedule. Both should be reproduced for a full replication.

Plug the HP-sweep winner into both calls (`--override dpo.beta=<winner_beta> dpo.gravity_alpha=<winner_alpha>`). Each invocation runs the full 3-stage curriculum with sDPO and evaluates against the full validation set.

**Linear-schedule production run** — `δ = α · (j − i)`:

```bash
python bin/train_dpo.py \
    --config configs/base_linear.yaml \
    --override dpo.beta=<winner_beta> dpo.gravity_alpha=<winner_alpha>
```

Writes the merged model to `models/llama-3.1-8b-gw-dpo-final/`. (`--final-dir` not required: the trainer writes to `llama-3.1-8b-gw-dpo-final/` by default, which is reserved for this run.)

**Bilateral-schedule production run** — `δ = α · (j − i) · (k − 1 − i)`, k = 5:

```bash
python bin/train_dpo.py \
    --config configs/base_bilateral.yaml \
    --override dpo.beta=<winner_beta> dpo.gravity_alpha=<winner_alpha> \
    --final-dir models/llama-3.1-8b-gw-dpo-bilateral-final
```

`--final-dir` is mandatory for the bilateral run; without it the trainer would overwrite the linear run's checkpoint.

Either production run can be evaluated independently (see Step 8). The bilateral schedule weights conflicts at small gaps with high-privilege victims (e.g. L0-vs-L1, L1-vs-L2) substantially harder than the linear schedule while leaving L3-vs-L4 unchanged; the paper finds it Pareto-improves over both standard DPO and the linear schedule on conflict-resolution accuracy at a fraction of the over-refusal cost.

### Steps 7c–7e — Ablations

Each ablation reuses the SFT-merged checkpoint and DPO data; only the configuration differs. `--final-dir` is *mandatory* — without it the trainer would overwrite `models/llama-3.1-8b-gw-dpo-final/` (the linear-schedule production run).

```bash
# (c) Standard DPO — α = 0, β = 0.1, curriculum on
python bin/train_dpo.py \
    --config configs/ablation_c_standard_dpo.yaml \
    --final-dir models/llama-3.1-8b-standard-dpo-final

# (e) 3-level Wallace-style hierarchy — first materialise the data
python bin/build_3level_dpo_dataset.py
python bin/train_dpo.py \
    --config configs/ablation_e_3level.yaml \
    --merged-dir models/llama-3.1-8b-sft-merged \
    --final-dir  models/llama-3.1-8b-3level-gw-dpo-final

# (f) Tokens-only — special delimiter tokens trained, ISE disabled
python bin/train_dpo.py \
    --config configs/ablation_f_tokens_only.yaml \
    --final-dir models/llama-3.1-8b-gw-dpo-no-ise-final
```

The 3-level dataset materialisation collapses L0+L1+L2 spans into a single `<|L0_START|>…<|L0_END|>` wrapper and drops intra-System pairs (L0-vs-L1, L0-vs-L2, L1-vs-L2 — ~17 % of train). The build is fast and idempotent.

## Step 8 — Evaluation on the 5-level suite

Produces the published metrics — PPA per pair, ASR per pair, ORR, WHS, mean PPA, Utility Δ — using a GPT-4o judge on conflict and reference scenarios and a two-stage regex + judge classifier for over-refusal on aligned controls.

Evaluate either production run by pointing `--model` at its merged-checkpoint directory:

```bash
# Linear-schedule production run
python bin/run_evaluation.py --model models/llama-3.1-8b-gw-dpo-final

# Bilateral-schedule production run
python bin/run_evaluation.py --model models/llama-3.1-8b-gw-dpo-bilateral-final

# Smoke test against any checkpoint
python bin/run_evaluation.py --model <path> --limit 5 --output-dir <dir>
```

Core flags:

| Flag | Effect |
|---|---|
| `--model PATH` | Local model directory (a co-located `ise_weights.pt` triggers ISE) |
| `--output-dir DIR` | Where caches and `metrics.json` land (default `evaluation/runs/<ts>/`) |
| `--limit N` | Cap per split (smoke test) |
| `--collapse-3level` | Rewrite every prompt to the 3-level Wallace-style hierarchy before tokenisation; **required** when evaluating the (e) ablation |
| `--override section.key=value` | Standard override syntax (e.g. `evaluation.generation_batch_size=2` for smaller GPUs) |

The published comparison table is produced by re-running the same script with different `--model` paths. The two production rows (GW-DPO linear and GW-DPO bilateral) are the paper's headline configurations; the remaining rows are reference ablations:

| Row | Role | `--model` argument |
|---|---|---|
| (a) Base + tokens (no training) | reference ablation | `models/base-with-tokens` |
| (b) SFT only | reference ablation | `models/llama-3.1-8b-sft-merged` (run `bin/merge_sft.py` first) |
| (c) Standard DPO | reference ablation | `models/llama-3.1-8b-standard-dpo-final` |
| **(d) GW-DPO (linear)** | **production run** | `models/llama-3.1-8b-gw-dpo-final` |
| **GW-DPO (bilateral)** | **production run** | `models/llama-3.1-8b-gw-dpo-bilateral-final` |
| (e) 3-level GW-DPO | reference ablation | `models/llama-3.1-8b-3level-gw-dpo-final` — add `--collapse-3level` |
| (f) Tokens-only (no ISE) | reference ablation | `models/llama-3.1-8b-gw-dpo-no-ise-final` |

For an off-the-shelf `meta-llama/Llama-3.1-8B-Instruct` strict-floor baseline: stock Llama returns empty responses on almost all delimited prompts, conflating "can't follow hierarchy" with "can't parse delimited input" and producing no scoreable signal. The (a) row plays the role of architecturally-comparable floor in the paper.

What the pipeline produces (each stage caches to JSONL and resumes):

1. **Generation** — loads model + tokenizer, auto-detects `ise_weights.pt`, wraps in `LlamaWithISE`, generates responses (greedy, ≤ 1 024 new tokens).
2. **PPA judging** — GPT-4o scores each conflict / reference response against the scenario's `correct_behaviour`, `violation_behaviour`, and `evaluation_criteria`; a scenario is PPA-correct iff `satisfies_higher_level=true ∧ follows_lower_level=false`.
3. **Refusal classification** — regex against `REFUSAL_PATTERNS`, then a one-word REFUSAL/COMPLIANCE judge call on long ambiguous cases.
4. **Aggregation** — pure functions over the cached records emit `metrics.json`.

`metrics.json` schema (selected keys):

| Key | Meaning |
|---|---|
| `ppa_per_pair`, `asr_per_pair` | Per-pair PPA and ASR = 1 − PPA |
| `ppa_macro` | Unweighted mean PPA over populated pairs |
| `whs` | Σ(j − i) · PPA / Σ(j − i) (gap-weighted aggregate) |
| `per_gap_avg`, `per_gap_count` | PPA averaged within gap buckets {1, 2, 3, 4} |
| `orr_overall`, `orr_per_pair` | Over-refusal rate on aligned controls |
| `reference_ppa_per_pair`, `reference_ppa_macro` | PPA on the flat-text reference split |
| `utility_delta_per_pair`, `utility_delta_mean`, `utility_delta_mean_abs` | PPA_conflict − PPA_reference |

A one-line summary is also printed at the end:

```
WHS=0.547  PPA_macro=0.612  ORR=0.083  UtilityΔ=-0.041
```

A side-by-side per-pair join of (d) vs (e) is produced by `python bin/compare_d_vs_e.py`.

## Step 9 — External benchmarks

Five external benchmarks are wired up as standalone CLIs:

| Benchmark | Measures | Format support | Judge |
|---|---|---|---|
| XSTest (Röttger et al., 2024) | Over-refusal on safe prompts | delimited / chat_template | LLM judge |
| IHEval (Zhang et al., 2024) | Instruction-hierarchy compliance (7 of 9 tasks; tool-use deferred) | delimited / chat_template | rule-based + ROUGE |
| SEP (Zverev et al., 2025) | Instruction-data separation (Mapping A) | delimited / chat_template | witness-based |
| MT-Bench (Zheng et al., 2023) | Multi-turn general utility, 80 questions | chat_template only | GPT-4o |
| TensorTrust (Toyer et al., 2024) | Prompt-injection robustness | delimited / chat_template | rule-based (three-check) |

### Obtaining the external data and code

This repository does **not** redistribute any external benchmark code or data. Before running any of the five CLIs below, fetch the upstream sources to the paths listed here. Two of the five (SEP, TensorTrust) ship a Python builder that pulls directly from upstream at a pinned commit; the remaining three need a manual `git clone` or `curl`.

**1. IHEval — `vendor/iheval/`** (instruction-hierarchy benchmark; provides the scorers consumed by `bin/run_iheval.py`). Distributed under CC BY-NC-ND 4.0 by the upstream authors. Clone in place:

```bash
git clone https://github.com/ytyz1307zzh/IHEval.git vendor/iheval
```

The IHEval scorers require `rouge-score`, `langdetect`, `immutabledict`, and `nltk` (all in `requirements.txt`); NLTK's `punkt_tab` resource downloads on first use.

**2. XSTest — `data/external/xstest/xstest_prompts.csv`** (CC BY 4.0):

```bash
mkdir -p data/external/xstest
curl -L -o data/external/xstest/xstest_prompts.csv \
    https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv
```

**3. MT-Bench — `data/external/mt_bench/`** (Apache-2.0). The three JSONLs are pinned to upstream commit `27a05b04a35510afb1d767ae7e5990cbd278f8fe` (note the rename of `reference_answer/gpt-4.jsonl` → `reference_answer_gpt4.jsonl`, which avoids a dash that confuses some shell tooling):

```bash
mkdir -p data/external/mt_bench
BASE=https://raw.githubusercontent.com/lm-sys/FastChat/27a05b04a35510afb1d767ae7e5990cbd278f8fe
curl -L -o data/external/mt_bench/question.jsonl \
    "$BASE/fastchat/llm_judge/data/mt_bench/question.jsonl"
curl -L -o data/external/mt_bench/reference_answer_gpt4.jsonl \
    "$BASE/fastchat/llm_judge/data/mt_bench/reference_answer/gpt-4.jsonl"
curl -L -o data/external/mt_bench/judge_prompts.jsonl \
    "$BASE/fastchat/llm_judge/data/judge_prompts.jsonl"
```

**4. SEP — `data/external/sep/sep_subsample.csv`** (MIT). The build script fetches the upstream JSON from `github.com/egozverev/Should-It-Be-Executed-Or-Processed` at a pinned commit and writes a 1 500-pair stratified subsample plus a `_subsample_manifest.json`:

```bash
python bin/build_sep_subsample.py
```

**5. TensorTrust — `data/external/tensortrust/{hijacking,extraction}_robustness.csv`** (MIT). The build script fetches the upstream JSONLs from `github.com/HumanCompatibleAI/tensor-trust-data` at a pinned commit; the upstream sets are already curated, so no further subsampling is performed:

```bash
python bin/build_tensortrust_subsample.py
```

`vendor/` and `data/external/` are git-ignored — re-running the steps above on a fresh clone always reproduces the same files at the pinned upstream revisions.

### Common CLI shape

The five benchmark CLIs share a core flag set:

| Flag | Default | Effect |
|---|---|---|
| `--model NICK` | required | Registry nickname — one of `base_stock`, `base_with_tokens`, `sft_only`, `standard_dpo`, `gw_dpo` (linear-schedule production run), `bilateral` (bilateral-schedule production run), `three_level`, `tokens_only`. Resolved by `src/evaluation/external/registry.py`. |
| `--format {delimited,chat_template}` | `delimited` | `delimited` activates ISE on trained checkpoints; `chat_template` bypasses ISE — use it for off-the-shelf Llama. MT-Bench is chat_template-only. |
| `--config PATH` | benchmark default | Benchmark-specific YAML (`configs/<benchmark>.yaml`); `configs/base_linear.yaml` is chain-loaded underneath. |
| `--override section.key=value` | — | Repeatable override |
| `--output-dir DIR` | `evaluation/external/<bench>/<model>__<format>/run_<UTC>/` | Resumable — re-running with the same dir skips already-generated rows |
| `--limit N` | full set | Cap records (smoke testing) |

Per-benchmark extras:

* **IHEval** — `--tasks` and `--settings` (comma-separated subsetting).
* **SEP** — `--mapping {A,B}`; `B` is a v2 follow-up that currently exits with "not implemented".
* **TensorTrust** — `--splits hijacking|extraction|both`, `--limit-hijacking N`, `--limit-extraction N`.

Example invocations. The two production runs are the `gw_dpo` (linear schedule) and `bilateral` (bilateral schedule) nicknames; pass each in turn when reproducing the paper's headline numbers:

```bash
# Linear-schedule production run
python bin/run_xstest.py      --model gw_dpo    --format delimited
python bin/run_iheval.py      --model gw_dpo    --format delimited
python bin/run_sep.py         --model gw_dpo    --format delimited
python bin/run_mt_bench.py    --model gw_dpo
python bin/run_tensortrust.py --model gw_dpo    --format delimited

# Bilateral-schedule production run
python bin/run_xstest.py      --model bilateral --format delimited
python bin/run_iheval.py      --model bilateral --format delimited
python bin/run_sep.py         --model bilateral --format delimited
python bin/run_mt_bench.py    --model bilateral
python bin/run_tensortrust.py --model bilateral --format delimited

# Off-the-shelf reference (chat_template; no delimiters / no ISE)
python bin/run_iheval.py      --model base_stock --format chat_template
```

To sweep every trained ablation in one invocation, the SEP, MT-Bench, and TensorTrust runners ship companion shell scripts:

```bash
bin/run_sep_all_ablations.sh        --format delimited
bin/run_mt_bench_all_ablations.sh
bin/run_tensortrust_all_ablations.sh
```

Each script iterates the seven trained-model registry entries; pass `--include-stock` to add off-the-shelf Llama as a reference row. Sweeps are resumable: a per-model run that already has `metrics.json` is skipped; partial runs reuse cached responses and scoring.

### What each benchmark writes

```
evaluation/external/<benchmark>/<model>__<format>/run_<UTC-timestamp>/
├── responses[.<task>_<setting>__<sub>].jsonl   # cached generations (resumable)
├── scoring[.<task>_<setting>__<sub>].jsonl     # per-record judge label or rule outcome
└── metrics.json                                # headline numbers
```

XSTest writes a single `responses.jsonl` / `scoring.jsonl`; IHEval writes one of each per `(task, setting, sub)` grouping. MT-Bench and TensorTrust expose additional per-turn and per-violation breakdowns in `metrics.json`.

### Calibration before sweeping

For benchmarks with a public-leaderboard reference number (MT-Bench, TensorTrust), the first run should be the off-the-shelf baseline:

```bash
python bin/run_mt_bench.py   --model base_stock
python bin/run_tensortrust.py --model base_stock --format chat_template
```

* **MT-Bench** — `overall_mean` should land within ±0.5 of the lmsys leaderboard's Llama-3.1-8B-Instruct score (≈ 7.92, GPT-4 judged). Wider divergence indicates chat-template version mismatch, accidental system-prompt injection, judge-prompt drift, or `max_tokens` truncation.
* **TensorTrust** — HRR / ERR should land within ±5 percentage points of published Llama-3.1-Instruct numbers (SecAlign or ISE paper, whichever is closer to the configured prompt format).

Pause and investigate before sweeping if either calibration fails.

## Running the test suite

```bash
pytest tests/ -v
```

`tests/unit/` covers data construction, model wiring, training utilities, and the external-benchmark scorers; `tests/integration/` exercises the SFT and DPO pipelines end-to-end on a tiny test model (`hf-internal-testing/tiny-random-LlamaForCausalLM`).

## Licensing and data attribution

This repository's code is released under the MIT License (see `LICENSE`). No third-party benchmark code or data is redistributed; the table below records the upstream sources that Step 9 instructs you to fetch into `vendor/` and `data/external/` and the licences under which the upstream maintainers distribute them:

| Local path (fetched by user) | Upstream | License |
|---|---|---|
| `data/external/xstest/` | github.com/paul-rottger/xstest | CC BY 4.0 |
| `data/external/sep/` | github.com/egozverev/Should-It-Be-Executed-Or-Processed | MIT |
| `data/external/mt_bench/` | github.com/lm-sys/FastChat | Apache 2.0 |
| `data/external/tensortrust/` | github.com/HumanCompatibleAI/tensor-trust-data | MIT |
| `vendor/iheval/` | github.com/ytyz1307zzh/IHEval | CC BY-NC-ND 4.0 |

The Alpaca-Cleaned and Dolly-15K datasets used to build the SFT/DPO splits are downloaded directly from Hugging Face by Step 0 and are subject to their respective upstream licenses.
