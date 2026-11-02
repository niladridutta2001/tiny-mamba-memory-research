# H8 seed-42 pilot: decoupled retention and selective replacement

The four slow modes use separate channel-wise erase and write gates conditioned on the current convolution feature, just-updated fast state, and event embedding. The tied control retains the same parameters but enforces `r_t = w_t` during computation.

| Model | Validation | Delay-511 head | Slow short probe | Slow long probe | Immediate overwrite | Post-32 overwrite | Post-32 SVER |
|---|---:|---:|---:|---:|---:|---:|---:|
| M1a constrained | 82.23% | 39.45% | 31.64% | 57.81% | 97.40% | 73.44% | 26.04% |
| H8 independent r/w | 95.18% | 31.25% | 29.69% | 39.84% | 100.00% | 91.15% | 3.12% |
| Tied-gate control | 99.63% | 40.23% | 28.12% | 52.73% | 100.00% | 100.00% | 0.00% |

Independent H8 minus M1a at delay 511: -8.20 pp (paired 95% bootstrap CI [-16.41, -0.39]). Independent H8 minus the parameter-matched tied control: -8.98 pp (CI [-16.80, -1.56], n=256).

The seed-42 pilot falsifies H8 as proposed: independent erase/write gates improve overwrite behavior but reduce long-gap retention and decodability. The tied control is the strongest tested variant. Long-trained probes remain diagnostic, and a multi-seed experiment is required for an architecture-level conclusion.
