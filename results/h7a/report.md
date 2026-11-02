# H7a seed-42 pilot: fast/slow SSM mode decomposition

## Research question

Does enforcing a persistent four-fast/four-slow decomposition preserve useful long-horizon information beyond the same separated initialization without persistent constraints? The decisive comparison is M1a versus M1b.

## Controlled architecture

All models retain one Mamba block, 64 inner channels, eight modes per channel (512 recurrent scalars), the fixed width-four causal convolution, input-dependent delta/B/C, and the original jointly trained `Linear(32,5)` head. M1a hard-constrains fast rates above 1.0 and slow rates below 0.004. M1b starts from exactly the same fast/slow decay matrix but removes the constraints. Each model has 10,149 parameters.

The slow cap was fixed from the baseline short-validation delta audit (mean NOOP delta 0.345) before M1 long-horizon evaluation. No sequence longer than 128 entered backbone training or checkpoint selection.

## Three-result diagnosis at delay 511

| Model | Group-2 survival | Group-2 long-trained probe | Original short-trained head |
|---|---:|---:|---:|
| M0 | 0% | 30.5% | 25.0% |
| M1a constrained | 39.2% | 53.5% | 39.5% |
| M1b unconstrained | 30.1% | 52.0% | 30.1% |

M1a-M1b original-head effect at delay 511: +9.4 pp (paired 95% bootstrap CI [4.3, 14.8], n=256). Persistent constraint enforcement is **supported in this pilot**. Long-trained probe accuracy is a diagnostic of accessible information, not strict length generalization.

"Group 2" means modes 4--7. It is the deliberately slow group in M1a/M1b and the corresponding mode slice in M0; it is not inherently a slow group in the original baseline.

## Interpretation boundaries

- High numerical survival alone does not demonstrate semantic memory.
- A long-trained probe can expose retained information but receives supervision unavailable to the original head.
- M1a versus M1b isolates persistent constraint enforcement beyond their identical separated initialization.
- This is one training seed. Even a positive paired example-level interval is not a multi-seed architecture conclusion.

## Artifacts

Raw metrics, paired contrasts, event-specific timescale audits, mode-masking results, cached frozen features, and publication figures are stored beside this report. The next decision is whether the evidence favors a dedicated slow-memory readout (information survives but the original head fails) or a selective replacement mechanism (retention improves but overwrite behavior degrades).
