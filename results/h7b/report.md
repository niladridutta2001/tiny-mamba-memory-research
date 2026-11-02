# H7b seed-42 pilot: identity-default slow-state updates

## Design

H7b keeps M1a's four constrained fast and four constrained slow modes, but replaces the slow recurrence with an interpolation between exact identity and the ordinary candidate update. The adaptive gate depends channel-wise on the current convolution-processed input; the static control has the same 128 additional parameters but receives an input-independent constant. Both start at sigmoid(-4)=0.0180.

## Delay-511 diagnosis

| Model | Original head | Slow short probe | Slow long probe | Cumulative NOOP drift |
|---|---:|---:|---:|---:|
| M1a constrained | 39.5% | 31.6% | 57.8% | 2.793 |
| H7b adaptive | 62.1% | 48.8% | 55.1% | 2.452 |
| Static control | 37.1% | 48.8% | 57.0% | 3.046 |

Adaptive gate means: NOOP 2.02%, SET 2.09%, QUERY 2.02%. The diagnostic mechanism is supported only if the adaptive gate suppresses irrelevant NOOP updates relative to useful SET updates and improves the outcome over both M1a and the static control.

Original-head effect versus M1a: +22.7 pp (paired 95% bootstrap CI [14.5, 30.5], n=256). Effect versus the parameter-matched static control: +25.0 pp (CI [18.0, 32.0]).

## Mandatory overwrite check: 100 NOOPs before overwrite

| Model | Immediate overwrite | Immediate stale error | Overwrite after 32 NOOPs | Stale error after 32 NOOPs |
|---|---:|---:|---:|---:|
| M1a constrained | 97.4% | 2.6% | 73.4% | 26.0% |
| H7b adaptive | 68.2% | 25.5% | 38.5% | 45.3% |
| Static control | 74.0% | 16.7% | 40.6% | 42.2% |

## Conclusion

The adaptive identity gate is a successful extreme-horizon diagnostic but not a successful replacement model in its current form. It reduces delay-511 cumulative slow-state drift from 2.793 to 2.452 and raises original-head accuracy from 39.5% to 62.1%. However, the learned event separation is small (SET minus NOOP = 0.07 percentage points), short-distribution validation falls, and obsolete values persist much more often. The current hypothesis is therefore only partially supported: identity-default dynamics help extreme retention/readout stability, but a single coupled gate suppresses useful replacement writes too aggressively.

## Interpretation boundaries

- Long-trained probes are diagnostic and receive long-horizon supervision unavailable to the original task head.
- The static control isolates input dependence from additional gate capacity, but its redundant channel-wise weights collapse to an input-independent parameterization during computation.
- Overwrite accuracy and stale-value errors are mandatory: a retention gain is not useful if obsolete assignments cannot be replaced.
- This is one training seed; paired example-level intervals are not a multi-seed architecture conclusion.
