# Tiny Mamba Memory Research

Local research workspace for HackerRank's long-horizon state-tracking assignment.
It starts from John Ma's readable `mamba-minimal` implementation and preserves
the upstream license and history.

## Current setup

- Python 3.11, managed locally in `.venv` by `uv`
- Fixed baseline config in `configs/baseline.yaml`
- Key-conditioned write-routing proposal and parameter-matched tied-row ablation
- PyTorch 2.6.0 with CUDA 12.4, using FP32
- Local smoke test: 41 logical tokens pad to 48 embedding tokens

Verified locally on an NVIDIA GeForce GTX 1650. A real Mamba forward/backward
smoke test used approximately 17.4 MiB for a `(2, 32)` input batch.

## Reproduce the environment

```powershell
uv sync --all-groups
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

On this Windows machine, using a project-local uv cache avoids user-cache ACL
issues:

```powershell
$env:UV_CACHE_DIR = Join-Path (Get-Location) '.uv-cache'
uv sync --all-groups --link-mode copy
```

## Planned implementation order

1. Implement and test the leakage-free synthetic generator and exact oracle.
2. Add the five-class state-tracking head and query-only loss.
3. Establish constant, oracle, and unmodified Mamba baselines.
4. Diagnose failures by delay, overwrites, and cross-key interference.
5. Choose one hypothesis-driven modification and a controlled ablation.
6. Run fixed-data comparisons and produce the notebook and report.

Part A2 is complete. Fixed evaluation files
from the assignment's `sample_data.zip` are extracted under `data/generated/`
(git-ignored), and the supplied generator is retained as `generate_dataset.py`.

Part B implements an 8-by-64 soft key-routing table on the selective-scan input
write. The tied-row control retains the same parameter table while removing key
selectivity. Reproduce the seed-42 comparison with:

```powershell
.\.venv\Scripts\python.exe train.py --config configs\proposed.yaml --seed 42 --max-updates 1500 --output-dir results\part_b\seed42\key_routing\u1500 --device cuda
.\.venv\Scripts\python.exe train.py --config configs\ablation.yaml --seed 42 --max-updates 1500 --output-dir results\part_b\seed42\tied_routing\u1500 --device cuda
.\.venv\Scripts\python.exe part_b_compare.py --device cuda
```

Run the five-seed frozen-model diagnostic that independently suppresses the
input-write or decay term at unrelated SET events with:

```powershell
.\.venv\Scripts\python.exe interference_term_ablation.py --device cuda
```

Its paired predictions, hierarchical-bootstrap contrasts, and plots are saved
under `results/interference_term_ablation/`.

Train and evaluate the learned selective-retention proposal and its matched
coupled gate with:

```powershell
.\.venv\Scripts\python.exe train.py --config configs\event_decay.yaml --seed 42 --max-updates 1500 --output-dir results\selective_retention\seed42\event_decay\u1500 --device cuda
.\.venv\Scripts\python.exe train.py --config configs\coupled_event_gate.yaml --seed 42 --max-updates 1500 --output-dir results\selective_retention\seed42\coupled_event_gate\u1500 --device cuda
.\.venv\Scripts\python.exe selective_retention_compare.py --device cuda
```

The learned gates, paired contrasts, and plots are under
`results/selective_retention/seed42/comparison/`.

Run the old-value/new-value overwrite and stale-error diagnostic with:

```powershell
.\.venv\Scripts\python.exe stale_value_diagnostic.py --device cuda
```

It evaluates a grid of pre- and post-overwrite delays with old and new values
forced to differ. Results are saved under
`results/selective_retention/seed42/stale_value/`.

Run the supplementary convolution-width sweep and targeted history-reset
diagnostic with:

```powershell
.\.venv\Scripts\python.exe train.py --config configs\event_decay_conv1.yaml --seed 42 --max-updates 1500 --output-dir results\selective_retention\seed42\dconv_sweep\dconv1\u1500 --device cuda
.\.venv\Scripts\python.exe train.py --config configs\event_decay_conv2.yaml --seed 42 --max-updates 1500 --output-dir results\selective_retention\seed42\dconv_sweep\dconv2\u1500 --device cuda
.\.venv\Scripts\python.exe train.py --config configs\event_decay_conv8.yaml --seed 42 --max-updates 1500 --output-dir results\selective_retention\seed42\dconv_sweep\dconv8\u1500 --device cuda
.\.venv\Scripts\python.exe dconv_overwrite_sweep.py --device cuda
.\.venv\Scripts\python.exe conv_history_reset_diagnostic.py --device cuda
```

Width 4 reuses the mandatory independently gated checkpoint. Outputs are under
`results/selective_retention/seed42/dconv_sweep/`.

Run the adaptive causal-kernel experiment and its parameter-matched static
mixture control with:

```powershell
.\.venv\Scripts\python.exe train.py --config configs\adaptive_kernel.yaml --seed 42 --max-updates 1500 --output-dir results\adaptive_kernel_kernel_only\seed42\adaptive\u1500 --device cuda
.\.venv\Scripts\python.exe train.py --config configs\static_kernel_mix.yaml --seed 42 --max-updates 1500 --output-dir results\adaptive_kernel_kernel_only\seed42\static\u1500 --device cuda
.\.venv\Scripts\python.exe adaptive_kernel_compare.py --device cuda
```

Both routed variants use one shared width-8 kernel, candidate widths
`{1, 2, 4, 8}`, identical 10,729-parameter budgets, and the same soft width-4
initialization. Outputs are under
`results/adaptive_kernel_kernel_only/seed42/comparison/`.

Run the assignment-defined length, delay-stress, and overwrite-stress protocol
with the additional calibration, class, stale-value, throughput, and memory
metrics using:

```powershell
.\.venv\Scripts\python.exe assignment_stress_evaluation.py --device cuda --batch-size 4
```

This evaluates the original baseline, proposed adaptive kernel,
parameter-matched static ablation, and fixed-width references on the same fixed
evaluation files. Outputs are under
`results/adaptive_kernel_kernel_only/seed42/assignment_stress/`.

Run the frozen-model, decay-only 511-step retention diagnostic with:

```powershell
.\.venv\Scripts\python.exe decay_only_retention_sweep.py --device cuda --batch-size 8
```

It preserves the trained input-write pathway, allows the first three NOOPs to
complete the width-4 convolutional response, and changes only the recurrent
decay exponent over the remaining 507 NOOPs. Outputs are under
`results/decay_only_retention/seed42/`.

Run the H5 frozen-backbone, decay-matched readout experiment with:

```powershell
.\.venv\Scripts\python.exe h5_retention_readout.py --device cuda --batch-size 8
```

Separate generated sequences are used for readout training, validation, and
testing. Outputs are under `results/h5_retention_readout/seed42/`.

Run the matched short-horizon frozen-head control with:

```powershell
.\.venv\Scripts\python.exe h5_short_horizon_control.py --device cuda
```

This separates the effect of long-horizon supervision from merely fitting a
fresh head on a frozen backbone. Outputs are under
`results/h5_short_horizon_control/seed42/`.

Run the short-only feature-alignment and decay-consistency study with:

```powershell
.\.venv\Scripts\python.exe horizon_robust_readout.py --device cuda
```

This compares short-only affine alignment, symmetric-KL and detached-teacher
consistency, plus a clearly labeled unlabeled-long-calibration CORAL control.
Outputs are under `results/horizon_robust_readout/seed42/`.

Run the matched class-conditional horizon-geometry training experiment with:

```powershell
.\.venv\Scripts\python.exe horizon_geometry_training.py --device cuda
```

This trains baseline, class-conditioned, and global-covariance controls only on
length-32--128 sequences. Outputs and independent checkpoints are under
`results/horizon_geometry_training/seed42/`.

Run the H5b independently trained internal-location probes with:

```powershell
.\.venv\Scripts\python.exe h5b_internal_readout.py --device cuda --batch-size 8
```

This compares the flattened recurrent state, the `C_t h_t` SSM readout, the
final backbone feature, and the original task head. It also includes 32D
PCA-based parameter-matched controls. Outputs are under
`results/h5b_internal_readout/seed42/`.

## H8 decoupled retention and replacement

H8 splits the slow-state update into independently learned erase and write
gates. Both gates use the current convolution feature, the just-updated fast
state, and a learned event-token embedding. The tied control has exactly the
same parameters but enforces `erase_gate == write_gate` during computation.

Train the decisive seed-42 pair with matched 1,500-update budgets:

```powershell
.\.venv\Scripts\python.exe train.py --config configs\h8_decoupled_event_gate.yaml --output-dir results\h8\checkpoints\independent --device cuda --seed 42
.\.venv\Scripts\python.exe train.py --config configs\h8_decoupled_tied_gate.yaml --output-dir results\h8\checkpoints\tied --device cuda --seed 42
```

Then run retention, independently trained state probes, event-conditioned gate
audits, slow-state drift, and overwrite/SVER diagnostics:

```powershell
.\.venv\Scripts\python.exe h8_decoupled_retention.py --device cuda
```

The no-event and no-fast ablation configurations are also provided, but are
secondary to the parameter-matched independent-versus-tied comparison.

Run the complete baseline training/validation pipeline with:

```powershell
.\.venv\Scripts\python.exe train.py --config configs\baseline.yaml
```

For a fast installation check:

```powershell
.\.venv\Scripts\python.exe train.py --max-updates 2 --max-validation-batches 2 --output-dir results\smoke_a2
```

## A3 baseline record

The seed-42 baseline was trained for 300 optimizer updates and evaluated on the
fixed 128, 256, 512, delay-stress, and overwrite-stress splits. Machine-readable
metrics and PNG plots are under `results/a3/`. Reproduce them with:

```powershell
.\.venv\Scripts\python.exe train.py --device cuda --max-updates 300 --output-dir results\a3\checkpoints
.\.venv\Scripts\python.exe evaluate.py --device cuda --checkpoint results\a3\checkpoints\baseline.pt --output-dir results\a3
```

## Upstream project: mamba-minimal

Simple, minimal implementation of Mamba in one file of PyTorch.

Featuring:
* Equivalent numerical output as official implementation for both forward and backward pass
* Simplified, readable, annotated code

Does NOT include:
* Speed. The official implementation is heavily optimized, and these optimizations are core contributions of the Mamba paper. I kept most implementations simple for readability.
* Proper parameter initialization (though this could be added without sacrificing readability)

## Demo

See [demo.ipynb](demo.ipynb) for examples of prompt completions.

```python
from model import Mamba
from transformers import AutoTokenizer

model = Mamba.from_pretrained('state-spaces/mamba-370m')
tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')

generate(model, tokenizer, 'Mamba is the')
```
> Mamba is the world's longest venomous snake with an estimated length of over 150 m. With such a large size and a venomous bite, Mamba kills by stabbing the victim (which is more painful and less effective than a single stab of the bite)

150 meters... 🫢 scary!

## References

The Mamba architecture was introduced in [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) by [Albert Gu](https://twitter.com/_albertgu?lang=en) and [Tri Dao](https://twitter.com/tri_dao?ref_src=twsrc%5Egoogle%7Ctwcamp%5Eserp%7Ctwgr%5Eauthor).

The official implementation is here: https://github.com/state-spaces/mamba/tree/main
