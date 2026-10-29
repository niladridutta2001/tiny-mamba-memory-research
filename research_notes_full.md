# Long-Horizon State Tracking with a Compact Selective SSM

Last updated: 27 October 2026.

This report separates observed measurements, interpretation, and speculation.

## Executive summary

The 10,149-parameter Mamba baseline was trained on sequence lengths 32--128.
After extending the weak 300-update run to 1,500 updates, it reached 75.05% on
ID-128 and 69.64% on OOD-512, but controlled retrieval fell from 84.38% at
delay 10 to chance at delay 500 across three training seeds. Five-seed tests
also established substantial cross-key interference: at delay 32, replacing
all intervening NOOPs with unrelated SETs reduced accuracy by 50.49 points.

Mechanistically, median cumulative recurrent retention falls to approximately
\(10^{-10}\) by delay 128 and \(10^{-40}\) by delay 500. A convolution-aware
oracle identity transition after the first three post-SET NOOPs restores about
97% accuracy through delay 128, strongly implicating continued passive-state
updates. However, frozen decay reduction alone does not restore task accuracy:
at a 511-step dependency, reduced-decay representations can be almost perfectly
decoded by a newly trained head while the original head stays near chance.

The architecture studies are mainly negative or qualified. Key-conditioned
write routing does not meaningfully beat its tied, parameter-matched control.
Independent event-conditioned decay modestly improves several benchmarks and
interference conditions but still fails at 500 steps and exhibits a conditional
overwrite cost. Adaptive convolutional width does not beat its static-mixture
control on the decisive stress tests. Finally, short-only KL consistency,
affine alignment, and class-conditional geometry regularization all fail to
produce reliable 511-step transfer. The strongest current conclusion is that
extreme-horizon failure combines recurrent attenuation with horizon-dependent
feature/readout shift; stabilizing only short-horizon output geometry is not
sufficient.

## Architecture

The Part B proposal adds an 8-by-64 routing-logit table to the single Mamba
layer. At a `SET(key, value)` token, the sigmoid of the row for that key
multiplies the discretized input write \(\Delta_t B_t u_t\) across the 64 inner
channels and broadcasts across all eight SSM modes. NOOP and QUERY writes are
unchanged. The table is initialized to a gate value of 0.95 so the initial
function remains close to the baseline. This adds 512 parameters, increasing
the model from 10,149 to 10,661 parameters (+5.04%).

The parameter-matched ablation retains the same 8-by-64 tensor and computation,
but averages its rows before the sigmoid and expands the resulting shared row
for every key. It therefore has the same parameter count and routing overhead
while eliminating key selectivity.

Subsequent experiments remain separate from that Part B routing proposal. The
independently gated retention model learns a 3-by-64 event-conditioned table
that scales only the recurrent decay exponent while leaving the input-write
term unchanged. Its coupled ablation applies the same gate to decay and write.
The gate-free adaptive-kernel diagnostic adds a shared width-8 depthwise kernel
and a soft router over effective widths \(\{1,2,4,8\}\), compared with a
parameter-matched static mixture. All kernel variants retain the original
Mamba recurrence and contain no event-conditioned retention gate. The final
horizon-geometry study returns to
the independently gated width-4 architecture and changes only its training
loss; it is not presented as a modification of the completed routing model.

## Falsifiable hypotheses

### H1 - Long-horizon retention degradation

**Hypothesis.** For a fixed key-value assignment, increasing the number of
intervening NOOP timesteps will reduce the baseline Mamba model's QUERY accuracy
because its recurrent dynamics fail to preserve a sufficiently decodable
representation of that assignment.

**Controlled test.** Generate balanced sequences of `SET(key, value)`, followed
by \(d \in \{10, 32, 64, 128, 256, 500\}\) NOOP events and a QUERY for the same
key. Balance all eight keys and four values and repeat the experiment across
random seeds. Use the same trained checkpoint at every delay and report accuracy
with seed-level confidence intervals. Because only the number of NOOP events
changes, the test excludes overwrite and cross-key-update interference.

**Prediction.** Accuracy will decline as delay increases, with the clearest loss
beyond the maximum training horizon.

**Falsification criterion.** H1 is not supported if long-delay accuracy is
statistically indistinguishable from short-delay accuracy, or increases, with
confidence intervals narrow enough to exclude a practically important decline.

**Mechanistic follow-up.** Examine two quantities:

1. **Effective retention.** Track the per-timestep retention term
   \(R_t = \exp(\Delta_t A)\) for every inner channel and state dimension, and
   measure cumulative retention over a NOOP interval as
   \(R_{i:j} = \exp(\sum_{t=i}^{j} \Delta_t A)\). Computing it in log space
   avoids numerical underflow from multiplying many small factors.
2. **Information decodability.** Train a linear diagnostic probe on held-out
   sequences to predict a key's stored value from intermediate post-backbone
   hidden states, and evaluate probe accuracy as a function of elapsed delay.

If value information remains decodable while the task head fails, the evidence
favors readout failure. If probe decodability declines with cumulative
retention, it favors representational retention failure.

### H2 - Cross-key updates destructively interfere with retrieval

**Hypothesis.** At a fixed SET-to-QUERY delay, replacing passive NOOP events with
SET events for other keys will reduce retrieval accuracy for the tracked key.
The compact recurrent state and shared selective-update projections do not
perfectly isolate key-specific information, so updates for unrelated keys
partially overwrite or entangle the tracked value.

**Controlled test.** Construct fixed-length sequences beginning with a balanced
`SET(tracked_key, value)` and ending with `QUERY(tracked_key)`. At the same event
positions, use cross-key SET densities of
\(p \in \{0, 0.1, 0.25, 0.5, 0.75, 1.0\}\); fill remaining positions with
NOOPs. Never update the tracked key. Balance tracked keys, values, distractor
keys, and seeds. Compare against a QUERY-distractor control to distinguish
interference caused by state-changing SET events from merely processing
non-NOOP inputs.

**Prediction.** Accuracy and hidden-state value decodability will decrease as
cross-key SET density increases, even though delay and the correct answer remain
fixed. SET distractors should be more damaging than QUERY distractors.

**Falsification criterion.** H2 is not supported if retrieval accuracy and probe
decodability do not decline with cross-key SET density, or if matched QUERY
distractors produce the same degradation, with sufficiently precise estimates.

**Mechanistic follow-up.** Compare the input-dependent \(\Delta\), \(B\), and
the resulting state change on NOOP, unrelated SET, and QUERY events. Measure
whether distractor SETs selectively alter state components that the diagnostic
probe uses for the tracked value.

### Seed-42 diagnostic pilot

The first balanced pilot is deliberately recorded as **inconclusive**, not as
confirmation of either hypothesis. On isolated `SET - NOOPs - QUERY` sequences,
the 300-update checkpoint scored 0% at delays 0 and 1, 25% at delay 2, and 0%
from delay 4 through 500. The non-monotonic delay-2 result and immediate floor
show that this checkpoint has not learned reliable isolated recall under this
controlled sequence format. Consequently, the proposed delay range cannot yet
identify a retention slope.

The cross-key pilot has the same problem: its zero-distractor control is at 0%,
so changes with distractor density cannot be interpreted as destructive
interference. Some distractor conditions slightly improve accuracy, most likely
because they move the input pattern closer to the random-event training
distribution; this is a hypothesis for follow-up, not an established cause.

Before using these controls to select a mechanism, the baseline should be
trained until short-delay isolated recall is reliably above floor, or the
controls should be redesigned to preserve the training distribution while
varying only delay/interference. This prevents a distribution-shift/readout
failure from being mislabeled as recurrent forgetting.

After extending the seed-42 baseline to 1500 updates and adding a fixed 32-NOOP
burn-in, the prerequisite control passed: accuracy was 90.63% at delay 2,
87.50% at delay 4, and 78.13% at delay 10. Accuracy then fell to 68.75% at delay
32, 43.75% at 64, 28.13% at 128, 21.88% at 256, and 25.00% at 500.

At a four-step delay, cross-key SET density did not cause a monotonic accuracy
decline (87.50% with no distractors versus approximately 78-88% across tested
densities). Thus this short-delay pilot did not support H2. It motivated the
longer fixed-delay and matched QUERY-distractor experiment reported immediately
below.

### H2 definitive multi-seed conclusion

H2 was re-tested using the five independently trained 1,500-update checkpoints
(seeds 42--46). Each condition contained 1,024 examples per model, balanced
over all eight tracked keys and four stored values. Every example had the same
randomized 32-event context across its paired conditions; that context never
updated the tracked key. At fixed SET-to-QUERY delays of 10, 32, 64, and 128,
nested subsets of the intervening positions were replaced at densities
\(p \in \{0, .1, .25, .5, .75, 1\}\). Distractor SETs always targeted one of
the other seven keys. A matched control replaced the same positions with QUERY
events for those keys, separating cross-key writes from generic processing of
active, non-NOOP tokens. Accuracy effects use 10,000 hierarchical bootstrap
draws over model seeds and paired examples; a two-percentage-point loss was
predeclared as practically relevant.

With only NOOPs between SET and QUERY, accuracy was 93.16%, 82.23%, 54.16%, and
29.47% at delays 10, 32, 64, and 128. At 100% unrelated-SET density, accuracy
fell to 78.67%, 31.74%, 21.80%, and 21.21%. The paired SET-versus-NOOP effects
were therefore -14.49 points (95% CI [-16.86, -12.19]), -50.49
[-53.91, -46.93], -32.36 [-39.67, -24.86], and -8.26 [-12.93, -3.69]. The
density response was small at low density but became monotonic and steep from
25--50% upward. For example, at delay 32 the losses were -5.59, -23.14,
-40.29, and -50.49 points at densities 25%, 50%, 75%, and 100%.

The active-token control shows that not all of this decline is specific to
writes. At 100% QUERY-distractor density, accuracy was 92.52%, 60.72%, 27.77%,
and 20.23%. Nevertheless, unrelated SETs were more damaging than matched
QUERYs at delays 10, 32, and 64: SET-minus-QUERY effects were -13.85 points
(CI [-16.99, -10.27]), -28.98 [-37.97, -19.61], and -5.98
[-10.20, -2.01]. All three intervals lie beyond the predeclared two-point
margin, although the delay-64 bound is marginal. At delay 128, SETs were not
more damaging than QUERYs (+0.98 points, CI [-2.23, 4.24]); the NOOP baseline
was already close to 25% chance and both active-token conditions were below
chance, so this condition has little remaining dynamic range for identifying a
SET-specific effect.

As a descriptive error check, when every intervening event was an unrelated
SET and its latest value differed from the correct target, predictions copied
that distractor value on 5.64%, 18.99%, 20.08%, and 22.15% of eligible
model-example pairs at delays 10, 32, 64, and 128. This is consistent with
cross-key value leakage, but it is not by itself a causal statistic because
class-frequency bias can also affect this rate.

**Conclusion.** H2 is supported over the usable operating range. Cross-key
SETs cause a strong density-dependent accuracy loss, and at delays 10--64 they
are materially more damaging than matched unrelated QUERY events. The QUERY
control also degrades accuracy at delays 32 and 64, so the failure is a mixture
of generic active-token interference and a distinct cost from writing unrelated
key-value associations. The delay-128 comparison is inconclusive about the
SET-specific mechanism because the baseline is already near floor; it does not
falsify the clearer effects at shorter delays.

### Frozen-model transition-term intervention

To distinguish incoming-write interference from shared attenuation, the five
trained baseline checkpoints were evaluated on the same paired H2 unrelated-SET
sequences under two diagnostic interventions. At unrelated SET positions only,
the write intervention set the recurrent input term \(\Delta_t B_tu_t\) to zero
while retaining the original \(\exp(\Delta_tA)\), token processing, output
readout, and direct \(D u_t\) path. The decay intervention instead set
\(\exp(\Delta_tA)=1\) while retaining the original input write and output
dynamics. Each delay-density condition had 1,024 examples per checkpoint, and
paired effects used 10,000 hierarchical bootstrap draws. The zero-density
conditions were exactly unchanged under both interventions, providing an
implementation check.

Suppressing unrelated input writes did not rescue retrieval. At 100% SET
density it changed accuracy by -54.65 points at delay 10 (95% CI [-59.82,
-49.32]), -15.49 at delay 32 [-19.28, -11.99], -12.05 at delay 64 [-15.16,
-8.69], and -14.53 at delay 128 [-18.63, -8.46]. The harm was monotonic with
density at delay 10 and was negative throughout nearly all other nonzero
conditions. Thus the frozen model appears to use the incoming writes from
unrelated SETs as part of its expected computation; simply eliminating those
writes is not a useful corrective intervention.

Suppressing attenuation produced a different pattern. At 100% density it
changed accuracy by +3.52 points at delay 10 (CI [-1.02, 9.06]), +17.38 at
delay 32 [8.77, 26.97], +6.04 at delay 64 [-2.89, 14.92], and -4.55 at delay
128 [-13.13, 4.06]. The benefit was clearer at intermediate densities: at 50%
density the effects were -0.49, +12.75, +15.37, and +5.88 points for delays 10,
32, 64, and 128, respectively; the latter three confidence intervals excluded
zero. This pattern implicates shared decay as an important contributor to
cross-key interference in the model's usable operating range, although the
nonmonotonic 100%-density results and floor effects prevent a universal claim.

These are causal perturbations of a frozen trained system, not candidate
replacement architectures. Zeroing either term moves hidden states away from
the training distribution, so the large write-suppression penalty does not
prove that incoming writes are intrinsically harmless. It does show that an
oracle which merely blocks those writes fails the proposed discriminating test,
whereas preserving prior state during unrelated SETs can recover accuracy in
several controlled conditions. This makes selective protection from attenuation
a stronger next design target than write gating alone.

### Revised H3: learned selective protection against decay

The frozen-model result motivated a learned event-conditioned multiplier
\(r_{e,d}=\exp(\tanh(\gamma_{e,d}))\), with a 3-by-64 table for NOOP, SET, and
QUERY events. The proposed model applies this multiplier only to the decay
exponent,

\[
h_t=\exp(r_t\Delta_tA)\odot h_{t-1}+\Delta_tB_tu_t,
\]

while the parameter-matched control applies the same gate to both decay and
the input write. Both tables were initialized at zero, giving \(r=1\) and an
exact baseline-equivalent initial function. Baseline, independent-decay, and
coupled models used seed 42, identical data order, AdamW settings, and 1,500
updates. The learned variants each add 192 parameters (10,341 total versus
10,149 for baseline). Their validation accuracies were 79.20%, 80.29%, and
78.51%, respectively.

On isolated balanced retrieval, independent decay improved over baseline by
+5.66 points at delay 10 (paired-example 95% CI [3.91, 7.42]) and +6.05 at
delay 32 [3.81, 8.30]. It was statistically indistinguishable at delays 64 and
128, improved by +9.47 at delay 256 [6.64, 12.30], and all models were at 25%
chance by delay 500. Relative to the coupled control, independent decay was
better at delays 10, 32, 64, and 256, with intervals excluding zero, but not at
128 or 500.

On the cross-key interference test, independent decay exceeded baseline across
all densities at delays 10 and 32. At 100% unrelated-SET density the gains were
+3.91 points at delay 10 (CI [0.98, 6.84]) and +4.49 at delay 32 [1.56, 7.42].
At delay 64 it helped at 25--75% density but not at the endpoints. At delay 128
it helped from 25% density upward, including +5.08 points at 100% [2.64, 7.62].
Against the coupled control at 100% density, the proposed model gained +4.30
points at delay 10 [1.86, 6.64], +2.44 at delay 32 [0.00, 4.88], and had no
clear advantage at 64 or 128.

The fixed assignment splits consistently favored independent decay over
baseline: 76.42% versus 75.05% ID, 72.71% versus 71.23% at length 256, 71.15%
versus 69.64% at length 512, 71.32% versus 69.59% on delay stress, and 70.89%
versus 69.67% on overwrite stress. The coupled control was below baseline on
all five splits.

There is a modest overwrite cost. Independent decay was 0.59--3.61 points
below baseline across the six controlled overwrite conditions; only the
one-overwrite, final-delay-64 loss had a paired interval excluding zero (-3.61
points, CI [-6.74, -0.39]). This is compatible with a retention--overwrite
trade-off, but does not show a broad severe penalty.

The learned independent multipliers were below one on most channels, meaning
less aggressive decay: means were 0.794 for NOOP, 0.848 for SET, and 0.862 for
QUERY, with 90.6%, 81.3%, and 82.8% of channels below one. The coupled model
learned means of 0.869, 0.963, and 0.982. Thus the proposed model learned
stronger retention protection while leaving writes intact, and its advantage
over the same-capacity coupled control supports the value of decoupling the two
terms.

**Conclusion.** Revised H3 is partially supported for seed 42. Independent
retention control improves the fixed benchmarks and several delay/interference
regimes, and it consistently outperforms the coupled gate in the clearest
conditions. The effect is not universal across delays, reaches chance at 500,
and introduces a small overwrite cost. Multi-seed training is required before
calling this a definitive architectural improvement.

### Stale-value diagnostic after long retention

The overwrite mechanism was examined without changing the architecture. Every
example used different old and new values, so predicting the old value was
unambiguously wrong. Five pre-overwrite NOOP delays (10, 32, 64, 100, and 128)
were crossed with six post-overwrite delays (0, 1, 4, 16, 32, and 64). Each
condition contained 1,024 balanced examples for the baseline, independent-decay,
and coupled models. Stale-value error rate (SVER) was the fraction of all
overwritten queries whose prediction equaled the obsolete value.

The proposed 100-NOOP example does not show a simple retention-versus-overwrite
failure because the old value is already weakly retained. At zero post-overwrite
delay, independent decay achieved 42.19% on the pure-retention control, 81.05%
overwrite accuracy, and 8.89% SVER. After four post-overwrite NOOPs, overwrite
accuracy rose to 91.80% and SVER fell to 4.69%. Thus a new assignment becomes
more reliably decodable after a few subsequent state transitions rather than
being blocked immediately by the old association.

The direct trade-off appears when the old value is still strongly retained. At
pre-delay 10, independent decay had 93.36% pure-retention accuracy but only
35.64% immediate overwrite accuracy, with 60.25% SVER. Relative to baseline,
this was +4.98 points in retention (95% CI [3.32, 6.74]), -6.15 points in
overwrite accuracy [-9.18, -3.03], and +4.30 points in SVER [1.07, 7.42]. At
pre-delay 32, it similarly gained 4.88 retention points [2.54, 7.23] while
losing 4.30 immediate-overwrite points [-7.42, -1.17]. These paired effects
provide direct evidence that stronger persistence can delay replacement of a
still-decodable old association.

Post-overwrite evolution is nonmonotonic. Across pre-delays, new-value accuracy
generally peaked after four NOOPs, then declined. For the 100-step primary
condition, independent-decay overwrite accuracy fell from 91.80% at post-delay
4 to 68.07% at 32 and 51.07% at 64, while SVER rose from 4.69% to 12.40% and
19.92%. At post-delay 32, independent decay was 11.04 accuracy points below
baseline (CI [-13.67, -8.40]) and 4.20 SVER points above it [2.05, 6.45]. The
failure therefore includes delayed loss of the replacement and, in this
condition, statistically detectable re-emergence of the obsolete value.

**Conclusion.** The diagnostic supports a conditional retention--overwrite
trade-off. When the old assignment remains highly decodable, the independent
gate preserves it more strongly and replacement is initially harder. After a
long enough pre-delay, the old trace is already weak and immediate overwrite is
easy, but the new assignment is most decodable only after a short settling
period and can later degrade. SVER explains part, but not all, of the later
errors; other wrong values also increase. This argues for analyzing memory
replacement as a time-dependent process rather than a single overwrite score.

### Supplementary convolution-width diagnostic

To test whether the post-overwrite peak comes from Mamba's local convolution,
the independently gated model was trained with \(d_{\mathrm{conv}}\in
\{1,2,4,8\}\). Width 4 remains the mandatory assignment baseline; the other
widths are supplementary diagnostics. All runs used seed 42, the same data
order, optimizer, and 1,500-update budget. Changing width also changed parameter
count (10,149, 10,213, 10,341, and 10,597 parameters) and optimization. Their
validation accuracies were 94.02%, 77.71%, 80.29%, and 80.65%, respectively,
so raw accuracy cannot be attributed solely to temporal mixing.

With 100 NOOPs before the overwrite, the post-overwrite accuracy peak moved
systematically with width. Widths 1 and 2 peaked immediately at post-delay 0.
Width 4 peaked after two NOOPs, rising from 81.05% to 94.92%, a paired gain of
13.87 points (95% CI [11.62, 16.31]). Width 8 peaked after four NOOPs, rising
from 89.26% to 97.36%, a gain of 8.11 points [6.35, 9.96]. The ordered peak
locations \(0,0,2,4\) support the hypothesis that the short settling interval
is related to convolutional receptive-field width. The result remains
supplementary because width 1 learned a qualitatively stronger overall model
and width 2 learned a weaker one.

A frozen width-4 intervention then reset only causal convolutional history at
the overwrite boundary while retaining the recurrent SSM state and all learned
parameters. This improved immediate overwrite accuracy by 9.47 points (CI
[6.45, 12.50]) and reduced immediate SVER by 7.62 points [-9.47, -5.86]. At
post-delay 32 it improved accuracy by 25.88 points [23.14, 28.71] and reduced
SVER by 11.33 points [-13.28, -9.47]. The reset model reached 100% accuracy and
0% SVER at post-delay 2.

The reset did not move the width-4 peak away from delay 2, so local history is
not the sole reason that the replacement is most decodable after two steps.
However, its large paired improvements and near-elimination of stale errors
show that pre-overwrite convolutional context materially contaminates the
processing of the replacement and that the resulting early perturbation can
persist in recurrent state long after the local convolution window has moved
on. As an out-of-distribution perturbation of a frozen model, this is causal
diagnostic evidence rather than a benchmark-ready modification.

### Multi-seed retention conclusion

The final H1 experiment used three independently trained models (seeds 42, 43,
and 44), each trained for the fixed 1500-update budget. Their validation
accuracies were 79.20%, 81.29%, and 81.29%. Mean balanced recall accuracy was
93.75% at delay 2, 87.50% at delay 4, 84.38% at delay 10, 64.58% at delay 32,
45.83% at delay 64, 28.13% at delay 128, 23.96% at delay 256, and exactly 25.00%
at delay 500.

The paired seed-level drop from delay 10 to delay 128 was 56.25 percentage
points (95% t-interval: 40.72 to 71.78 points). The drop from delay 10 to delay
256 was 60.42 points (95% t-interval: 51.45 to 69.38), and the drop to delay 500
was 59.38 points (95% t-interval: 45.93 to 72.82). All intervals exclude zero.
Because the balanced test has four possible stored values, 25% is chance.

These results support H1 across the three tested training seeds: retrieval
degrades strongly with passive delay and reaches chance at the longest horizon.
The conclusion concerns behavioral retrieval, not yet the mechanism. Retention
traces and hidden-state probes are still required to distinguish information
loss from readout failure. With only three model seeds, confidence intervals are
wide and should be reported rather than implying high-precision effect sizes.

### H3 - Overwrite-retention trade-off

**Hypothesis.** Mamba configurations with slower effective state decay achieve
better long-delay retention but worse overwrite accuracy because obsolete
key-value associations persist in the recurrent representation.

**Controlled intervention.** Scale the effective step size using
\(\Delta'_t = \alpha\Delta_t\), initially with
\(\alpha \in \{0.25, 0.5, 1.0, 2.0\}\). Values below one slow passive decay;
values above one accelerate decay and updating. Compare all settings using the
same trained seed-42 checkpoint and fixed evaluation sequences before training
any modified model. This inference-time sweep is a mechanistic intervention,
not the final proposed-model comparison.

Because \(\Delta_t\) appears in both
\(\exp(\Delta_t A)\) and the discretized input term
\(\Delta_t B_t u_t\), global scaling changes both retention and write strength.
To localize the effect, add a second controlled sweep that scales \(\Delta_t\)
only at NOOP events while leaving SET and QUERY events unchanged. This isolates
passive retention. A complementary SET-only sweep measures overwrite/write
sensitivity directly.

**Evaluation.** Measure balanced long-delay accuracy at the established delay
grid and probe-only overwrite accuracy for 1, 2, 4, and 8 overwrites. Match the
delay from the final SET to QUERY across conditions. In addition to accuracy,
report the fraction of errors equal to the immediately previous value. This
`obsolete-value error rate` distinguishes persistence of stale memory from
generic misclassification.

**Prediction.** Smaller effective step sizes will improve delayed retrieval but
increase obsolete-value errors after overwrites. Larger step sizes will improve
replacement of old values while worsening long-delay retention.

**Falsification criterion.** H3 is not supported if stronger retention
consistently improves both delayed retrieval and overwrite correctness without
an observable increase in obsolete-value errors, with confidence intervals
narrow enough to exclude a practically relevant penalty.

**Research significance.** This intervention directly connects the selective
step size to the assignment's central memory trade-off. It also distinguishes a
genuine retention-overwrite compromise from a modification that simply improves
optimization or adds capacity.

### H3 inference-time delta sweep result (seed 42)

The trained seed-42 checkpoint was evaluated without retraining under global,
NOOP-only, and SET-only scaling with
\(\alpha \in \{0.25, 0.5, 1.0, 2.0\}\). The unmodified \(\alpha=1\) model had
28.13% balanced retention accuracy at delay 128, 87.40% overwrite-probe
accuracy, and a 5.00% obsolete-value error rate.

Global scaling produced the appearance of a trade-off: reducing \(\alpha\) to
0.25 left delay-128 accuracy unchanged at 28.13% but reduced overwrite accuracy
to 80.40% and increased obsolete-value errors to 7.20%. Increasing \(\alpha\)
to 2.0 reduced delay-128 accuracy to 25.00% while improving overwrite accuracy
to 91.40% and reducing obsolete errors to 3.40%. However, global scaling also
changes the strength of SET writes and QUERY processing, so it does not isolate
passive retention.

The cleaner NOOP-only intervention did not show the predicted overwrite cost.
At \(\alpha=0.5\), delay-64 retention improved from 43.75% to 59.38%, while
overwrite accuracy improved slightly from 87.40% to 89.00% and obsolete errors
fell from 5.00% to 4.00%. The benefit did not persist at delay 128, where
accuracy was 25.00%. NOOP-only \(\alpha=0.25\) similarly left delay-128 accuracy
unchanged while improving overwrite accuracy to 89.00%.

SET-only scaling localized the write-strength effect. Weakening SET updates to
\(\alpha=0.25\) reduced overwrite accuracy to 79.00% and increased obsolete
errors to 7.20%, without improving delay-128 retention. Strengthening SET updates
to \(\alpha=2.0\) raised delay-128 accuracy to 34.38% while leaving overwrite
accuracy near baseline at 87.00%.

**Conclusion.** H3 is not supported as a general intrinsic trade-off by this
inference-time sweep. The overwrite penalty under small global \(\alpha\) is
better explained by weakened encoding of new SET values than by excessive
persistence during NOOP intervals. Moderate NOOP-only scaling can improve
mid-horizon retention without an overwrite penalty, directly contradicting the
predicted compromise at that horizon. Nevertheless, none of the fixed scales
solves retention beyond 128 steps. These are single-checkpoint interventions;
a trained adaptive mechanism and controlled ablation are required for Part B.

### H3 definitive multi-seed follow-up

The pilot conclusion above was re-tested with five independently trained
1,500-update checkpoints (seeds 42--46). Seeds 45 and 46 achieved 80.32% and
82.91% validation accuracy, respectively, and were accepted without selecting
on H3 performance. Each model was evaluated on the same 1,024 examples per
condition. Examples were balanced over all eight keys and four assigned values,
but used randomized 32-event contexts that excluded the tracked key. The
overwrite tests contained 1, 4, or 8 actual replacements, with four NOOPs
between replacements and a final delay of 32 or 64 NOOPs before the query.
Every alpha comparison was paired at the example level. Reported 95% intervals
come from 10,000 hierarchical bootstrap draws that resample both model seeds
and examples. Before analysis, a two-percentage-point overwrite-accuracy loss
was defined as the smallest practically relevant penalty.

The clean NOOP-only intervention produced a horizon-dependent retention effect.
Relative to \(\alpha=1\), \(\alpha=0.5\) reduced delay-32 accuracy from 81.15%
to 72.60% (effect -8.55 points, 95% CI [-10.21, -6.88]), was indistinguishable
at delay 64 (54.47% to 53.54%; -0.94 points, CI [-3.93, 2.38]), and improved
delay-128 accuracy from 29.32% to 33.98% (+4.67 points, CI [0.21, 9.49]). Thus
slower passive dynamics provide a small but statistically resolved benefit only
at the longest tested horizon; they are not uniformly beneficial.

The same \(\alpha=0.5\) intervention produced an overwrite penalty when the
latest value was queried after 32 NOOPs. Accuracy changed by -7.36 points for
one overwrite (CI [-9.22, -5.43]), -2.56 for four (CI [-4.14, -0.98]), and
-3.44 for eight (CI [-5.41, -1.54]). For one overwrite, obsolete-value errors
also increased by 3.18 points (CI [0.72, 5.55]), directly demonstrating greater
persistence of the replaced value. The obsolete-error increases for four and
eight overwrites were not statistically resolved. Only the one-overwrite
accuracy interval lay entirely beyond the predeclared two-point practical
penalty margin, although all three accuracy effects were below zero.

After a final delay of 64, the overwrite effects were not resolved: -0.49
points for one overwrite (CI [-3.57, 2.60]), +1.64 for four (CI [-0.57, 4.10]),
and +0.29 for eight (CI [-2.13, 2.79]). Their obsolete-error intervals also
included zero. This likely reflects convergence toward a low-information regime
in which neither the latest nor obsolete value remains reliably readable.

**Revised conclusion.** H3 receives qualified, not universal, support. Across
five models, slowing only passive NOOP dynamics with \(\alpha=0.5\) caused both
a resolved delay-128 retention gain and a resolved overwrite cost after a
32-NOOP interval; the one-overwrite condition specifically shifted errors
toward the obsolete value. However, slower dynamics harmed short-delay
retention, had no clear effect at delay 64, and the overwrite penalty was not
detectable after 64 NOOPs. The evidence therefore supports a finite-horizon
overwrite--retention trade-off, not a monotonic law that applies at every delay
or overwrite count. This strengthens the motivation for an event-adaptive
controller rather than choosing one fixed global decay scale.

### H4 - NOOP is not a learned identity transition

**Hypothesis.** The baseline fails to approximate an identity transition during
semantically passive NOOP events. Repeated unnecessary changes accumulate and
contribute causally to long-delay retrieval failure. Nonzero displacement alone
is not treated as an error because a changing representation may remain fully
decodable.

**Controlled test.** For the five 1,500-update checkpoints, evaluate 1,024
balanced randomized-context examples at NOOP delays 10, 32, 64, and 128.
Reconstruct the exact selective-scan state at every timestep and also record the
post-backbone hidden vector. At each intervening NOOP measure
\(\|h_t-h_{t-1}\|_2/(\|h_{t-1}\|_2+\epsilon)\), cumulative path length, and net
displacement. Test the association between cumulative displacement and errors
within each fixed delay using a standardized slope and a hierarchical bootstrap
over models and examples. This within-delay analysis avoids declaring support
merely because both elapsed time and errors increase.

Add two causal interventions without retraining. First, force
\(\Delta_t=0\) at every post-SET NOOP, which makes the selective-scan recurrence
an exact identity at those positions. Second, account for the model's
four-token depthwise convolution: allow the first \(d_{conv}-1=3\) NOOP
positions to complete the convolutional response to the SET token, then force
identity for all remaining NOOPs. Prefix positions and the final QUERY are
unchanged in both interventions.

**Observed displacement.** Baseline NOOP transitions were substantially
non-identity. Median cumulative relative displacement of the selective-scan
state rose from 2.52 at delay 10 to 2.65, 2.80, and 2.89 at delays 32, 64, and
128. Corresponding post-backbone values were 2.19, 2.49, 2.69, and 2.84. The
curves saturate rather than growing linearly, indicating that most movement
occurs early in the interval.

Natural example-to-example drift did not satisfy the simple correlational
prediction. Within a fixed delay, one standard deviation more recurrent-state
path length was associated with *lower* error at delay 10 (-2.81 percentage
points, 95% CI [-6.08, -0.04]) and delay 32 (-4.59 points,
[-7.81, -1.20]); effects at 64 and 128 included zero. Post-backbone cumulative
drift effects included zero at every delay. Larger naturally occurring drift is
therefore not a reliable marker of failure and may partly reflect a stronger
initial write or useful early computation.

Forcing identity immediately after SET collapsed accuracy to approximately
22% at every delay. This does not show that passive drift is necessary: because
the depthwise convolution has width four, the SET representation continues to
enter the recurrence during the next three positions. The immediate freeze
prevents completion of that write and is therefore an intentionally recorded
confounded intervention.

The convolution-aware intervention gives the decisive result. Allowing those
three completion steps and then freezing all later NOOP updates produced 96.84%,
96.70%, 97.05%, and 96.60% accuracy at delays 10, 32, 64, and 128, compared
with baseline accuracies of 93.36%, 81.15%, 54.47%, and 29.32%. Paired gains
were +3.48 points (CI [2.27, 4.86]), +15.55 [13.44, 17.77], +42.58
[34.38, 50.88], and +67.29 [62.71, 71.88]. Thus an exact recurrent identity
after convolutional washout nearly eliminates the behavioral retention loss
through delay 128.

**Conclusion.** H4 is strongly supported in a refined causal form but its
simple observational prediction is not. Continued recurrent updates during
late, semantically passive NOOPs are harmful, while the first three apparent
NOOPs are computationally active because they complete the convolutional SET
write. Ordinary drift magnitude within a fixed condition does not positively
predict error. The appropriate proposed mechanism is therefore a delayed or
event-aware no-update gate that preserves convolutional write completion and
then approaches identity, not a rule that freezes every token labelled NOOP.
The intervention is an oracle diagnostic rather than a trained final model.

### Mechanistic retention and decodability analysis

For each trained seed, the exact single-layer selective-scan retention term was
reconstructed from the model's convolutional input, `x_proj`, `dt_proj`, and
state matrix. The cumulative product was evaluated in log space over only the
intervening NOOP positions. Median cumulative log10 retention declined from
-0.91 at delay 10 to -2.65 at 32, -5.17 at 64, -10.22 at 128, -20.31 at 256,
and -39.56 at 500. Thus the median recurrent component preserves only about
10^-10 of its prior-state contribution by delay 128 and about 10^-40 by delay
500. Retention is heterogeneous: at delay 500, the per-seed 90th-percentile
log10 retention ranged from approximately -4.07 to -6.15, while the fastest
decaying components were far smaller.

A separate four-class linear probe was trained at each delay on frozen
post-backbone hidden states. The probe received the 32-dimensional hidden state
and an eight-way one-hot indication of the tracked key. Training and evaluation
used disjoint randomized 32-event prefixes that never updated the tracked key;
the model weights remained frozen. Mean held-out probe accuracy was 49.09% at
delay 10, 48.18% at 32, 47.01% at 64, 45.57% at 128, 41.67% at 256, and 33.98%
at 500. At delay 500 its 95% seed-level t-interval was 28.94-39.03%, above the
25% four-value chance level.

**Interpretation.** The sharply decreasing effective-retention factors provide
mechanistic evidence that passive recurrent decay is real. However, the
delay-specific linear probe remains above chance after the fixed task head has
fallen to chance. Therefore the failure is not complete information erasure:
some value information survives in a delay-dependent, linearly decodable form
that the fixed task head does not reliably use. The evidence supports a mixed
failure involving both representational attenuation/drift and readout mismatch.

The task-head and probe curves are not directly interchangeable: the task-head
pilot uses a fixed NOOP burn-in, whereas probe generalization uses randomized
held-out prefixes, and a separate probe is fitted at every delay. A single probe
trained only on short delays and tested without refitting at long delays was
therefore pursued through the later H5 short-only controls.

## Method

### Part B training protocol

Baseline, key-conditioned routing, and tied routing use the same seed-42 data,
data order, AdamW optimizer, learning rate 0.001, batch size 16, FP32 precision,
and 1,500-update budget. Proposed and tied models start with exactly identical
predictions; only their ability to specialize routing rows differs during
learning. No checkpoint was selected on an H2 metric. Validation accuracy after
1,500 updates was 79.20% for baseline, 79.84% for key-conditioned routing, and
79.62% for tied routing, placing all three in the same learned regime.

The predeclared primary test is the paired H2 unrelated-SET evaluation with
1,024 examples per condition. The decisive contrast is key-conditioned versus
tied routing, plus a difference-in-differences comparison of each model's
SET-induced loss relative to its own zero-distractor accuracy. Example-level
95% bootstrap intervals use 10,000 paired resamples. Because only one training
seed has been run for the proposed models, these intervals quantify evaluation
sampling uncertainty, not training-seed uncertainty.

## Results

### Part B key-conditioned routing result

The proposal did not produce a practically meaningful advantage over the
parameter-matched tied control. At 100% unrelated-SET density, proposed-minus-
tied accuracy was +0.49 points at delay 10 (95% paired bootstrap CI
[0.10, 0.98]), +0.10 at delay 32 [-0.39, 0.59], -0.39 at delay 64
[-0.88, 0.00], and +0.29 at delay 128 [-0.29, 0.88]. All effects are below one
percentage point and all are below the predeclared two-point threshold for a
practically relevant improvement.

The difference-in-differences result likewise showed no resolved reduction in
interference: +0.59 points at delay 10 (CI [0.00, 1.27]), 0.00 at delay 32
[-0.68, 0.68], +0.10 at delay 64 [-0.49, 0.68], and +0.59 at delay 128
[-0.10, 1.37]. Thus the small raw delay-10 advantage is not evidence of a
meaningful key-specific protection effect.

On fixed assignment splits, key-conditioned and tied routing were also nearly
identical. Their respective accuracies were 75.57% versus 75.50% on ID-128,
71.54% versus 71.42% on OOD-256, 69.86% versus 69.75% on OOD-512, 70.06%
versus 69.96% on delay stress, and 69.55% versus 69.50% on overwrite stress.
Both routed models sometimes slightly exceed the smaller baseline, but the tied
control reproduces those gains, so they cannot be attributed to key selectivity.

The learned proposed table did develop non-identical rows, but specialization
was weak: mean gate 0.946, mean channel-wise standard deviation across keys
0.0152, mean absolute deviation from each channel's key mean 0.0114, and mean
pairwise row cosine similarity 0.99958. Because cosine similarity alone is
insensitive to magnitude, ordered row pairs were also evaluated using
\(\|r_i-r_j\|_2/(\|r_i\|_2+\epsilon)\). The mean normalized distance was
0.0286, the median 0.0282, and the range 0.0184--0.0401. Mean per-channel
variance across keys was 0.000333 (median 0.000141; maximum 0.002067). Gates
ranged from 0.819 to 0.981. Thus rows are not identical in either direction or
magnitude, but their typical separation is only about 2.9% of a row norm and
the across-key variance is small. The network made modest key-dependent
adjustments while retaining broadly overlapping channel-use patterns.

Training took 953.6 seconds for key-conditioned routing and 1,162.3 seconds for
tied routing on the GTX 1650, versus approximately 449.6 seconds for the
extended seed-42 baseline. Both routed models peaked at 57,888,256 allocated GPU
bytes, compared with 53,143,040 for the baseline. The readable per-position
routing implementation therefore carries substantial wall-clock overhead even
though its parameter and memory increases are small.

### A3 initial baseline (seed 42, 300 updates)

The fixed Mamba-1 configuration was trained from random initialization for 300
AdamW updates at batch size 16 and learning rate 0.001 in FP32. Training took
77.18 seconds on an NVIDIA GeForce GTX 1650. The model has 10,149 trainable
parameters and peak allocated GPU memory was 53,143,040 bytes (50.68 MiB).

| Evaluation split | Length | Neural | Always UNSET | Train majority | Oracle |
|---|---:|---:|---:|---:|---:|
| ID | 128 | 25.67% | 23.28% | 23.28% | 100% |
| OOD | 256 | 14.81% | 11.28% | 11.28% | 100% |
| OOD | 512 | 10.45% | 5.97% | 5.97% | 100% |
| Delay stress | 512 | 8.64% | 4.21% | 4.21% | 100% |
| Overwrite stress | 512 | 8.86% | 4.28% | 4.28% | 100% |

The training-set majority class is UNSET, so the two non-neural baselines are
identical for this seed and dataset.

### Extended baseline selected for subsequent experiments

The 300-update model learned only weakly and failed the immediate-recall
prerequisite in the controlled H1 pilot. Following the assignment's instruction
to extend training when necessary, the same seed-42 model and AdamW state were
resumed to predeclared checkpoints at 600, 1000, and 1500 total updates.
Checkpoint selection used validation accuracy only: 54.78%, 67.70%, and 79.20%,
respectively. The 1500-update checkpoint was therefore selected. Its cumulative
training time was approximately 449.6 seconds on the GTX 1650.

| Evaluation split | Length | Neural | Always UNSET | Oracle |
|---|---:|---:|---:|---:|
| ID | 128 | 75.05% | 23.28% | 100% |
| OOD | 256 | 71.23% | 11.28% | 100% |
| OOD | 512 | 69.64% | 5.97% | 100% |
| Delay stress (all queries) | 512 | 69.59% | 4.21% | 100% |
| Overwrite stress (all queries) | 512 | 69.67% | 4.28% | 100% |

The extended model demonstrates substantial state-tracking learning while
retaining a measurable length-generalization gap. Stress values above aggregate
all queries in each split; probe-only stress metrics must be reported separately
in the controlled experiment and must not be inferred from this table.

## Ablation

The tied-row experiment falsifies the mechanism-specific claim for this run.
Giving each key its own learned routing row does not outperform using one shared
effective routing row with the same nominal parameter table and computation.
The proposal satisfies the implementation and capacity controls, but it fails
its stated causal success criterion. A multi-seed proposed-model replication
would narrow training uncertainty, yet effects below one point in this seed do
not justify claiming that key-conditioned writes solve the H2 bottleneck.

The result also clarifies why write routing alone may be insufficient. It gates
only \(\Delta_t B_tu_t\); unrelated SETs still decay all existing recurrent
coordinates through \(\exp(\Delta_t A)\), and the input/output projections mix
channels. H4's much larger convolution-aware identity effect suggests that
controlling passive transition dynamics is a stronger next mechanism than
partitioning only the input write.

## Failure analysis

**Observed measurement.** Query accuracy falls from 25.67% at length 128 to
14.81% at 256 and 10.45% at 512. Delay and overwrite stress accuracy is below
9%. The model remains above the constant baseline, but the absolute gap to the
exact oracle is large.

**Interpretation at the A3 stage.** This is direct evidence of a
length-generalization failure:
the state tracker learned some useful behavior in the training range but does
not preserve and retrieve it reliably as the sequence grows. These A3 aggregate
measurements alone did not distinguish retention failure from interference or
readout failure; the controlled experiments later in this report address that
distinction.

**Speculation to test in Part C.** Retention and cross-key interference are
plausible bottlenecks because both OOD accuracy and the long-gap stress result
degrade sharply. Accuracy by SET-to-QUERY delay and controlled ablations are
needed before treating that explanation as established.

## Further experiments and limitations

### Adaptive causal-kernel diagnostic (seed 42)

The adaptive model uses one shared depthwise kernel of maximum width eight and
an input-dependent soft mixture over effective widths \(\{1,2,4,8\}\). A
parameter-matched static control uses the same kernel, router parameters, and
10,729-parameter budget, but learns one mixture shared by every token. Both
routers were initialized with the same soft preference for width four. None of
the models in this comparison has an event-conditioned decay gate: original
Mamba is the fixed-width-four control, and the fixed-width-eight model changes
only the causal convolution. An earlier gated-backbone run was discarded after
identifying the retention-gate confound and is not used below.

| Model | Validation | ID-128 | OOD-256 | OOD-512 | Overwrite stress |
|---|---:|---:|---:|---:|---:|
| Original / fixed width 4 | 79.20% | 75.05% | 71.23% | 69.64% | 69.67% |
| Fixed width 8 | 80.05% | 76.59% | 73.48% | 71.94% | 71.86% |
| Static mixture | 81.62% | 77.62% | 74.46% | 72.94% | 72.56% |
| Adaptive mixture | **82.26%** | **78.95%** | **76.06%** | **73.91%** | **73.98%** |

The adaptive model has the best aggregate accuracy, leading static by 0.64
validation points, 0.97 OOD-512 points, and 1.42 overwrite-stress points. Those
stream-level gains do not establish that input-dependent routing improves the
designed memory operations, which require paired final-query probes.

The targeted pre-delay-100 diagnostic is more decisive against the proposed
mechanism. Across post-overwrite delays 0, 1, 2, 4, 8, 16, and 32, adaptive
overwrite accuracy is below the static control by 1.86--10.16 percentage
points. Paired 95% bootstrap intervals exclude zero at post-delays 0, 1, 2, 4,
16, and 32; only delay 8 is inconclusive. Pure-retention accuracy is also much
lower for adaptive: 23.34% versus 42.19% at post-delay 0 and 14.45% versus
32.23% at post-delay 32.

The learned routing is genuinely event dependent: expected widths are 7.99 for
NOOP, 6.16 for SET, and 7.55 for QUERY, whereas the static control uses 4.21 for
all events. The router therefore learned mostly wide processing, especially on
NOOPs, but that behavior does not improve the controlled retention/overwrite
task.
This seed does not support the claim that input-dependent receptive-field
selection solves overwrite contamination. A crucial limitation is that the
router observes the current token-side representation rather than recurrent
memory, so it cannot directly distinguish a first assignment from an overwrite.
Multi-seed replication would be required before interpreting the small
validation difference as a reliable general gain.

### Assignment-defined stress tests and additional metrics

The final baseline/proposed/ablation evaluation used the exact fixed files
generated for the assignment: 500 length-512 delay-stress sequences and 500
length-512 overwrite-stress sequences. The delay stress sets the focus key at
the first timestep, excludes subsequent writes to that key, and queries it at
step 511. The overwrite stress applies exactly 1, 2, 4, or 8 focus-key
overwrites (125 examples each) before the final query. All-query accuracy and
the designed final-probe accuracy are reported separately.

| Model (assignment role) | Delay all queries | Gap-511 probe | Overwrite all queries | Final overwrite probe |
|---|---:|---:|---:|---:|
| Original Mamba (baseline) | 69.59% | 22.8% | 69.67% | 87.4% |
| Fixed width 8 (reference) | 71.54% | 22.6% | 71.86% | 92.0% |
| Static mixture (ablation) | 72.61% | **24.4%** | 72.56% | **93.4%** |
| Adaptive mixture (proposed) | **73.58%** | 23.2% | **73.98%** | 92.4% |

On the 500 paired probes, adaptive minus static is -1.2 points for the extreme
gap (95% bootstrap CI -5.6 to +3.0) and -1.0 point for overwrite (-3.2 to
+1.2). Neither interval excludes zero. Adaptive minus original on overwrite is
+5.0 points (+1.2 to +8.8), but this comparison includes 580 additional
parameters and a different local-mixing mechanism. The matched static ablation
is therefore the primary causal control, and it provides no evidence for an
adaptive-routing advantage.

All models fail the 511-step retention probe: the best observed accuracy is
only 24.4%, near the four-value chance level of 25% for these initialized
probes. Static also slightly exceeds adaptive on final overwrite (93.4% versus
92.4%), consistent with the targeted post-overwrite sweep.

Additional metrics reinforce the distinction between the targeted probe and
general stream tracking. Adaptive has the best aggregate stress accuracy and
lower NLL than static, but those advantages do not transfer to the designed
final queries. The proposed routing therefore improves average stream tracking
while failing the experiment intended to establish adaptive receptive-field
selection as the responsible memory mechanism.

All models used 1,500 updates on an NVIDIA GeForce GTX 1650. Parameter counts
are 10,149 (original), 10,405 (fixed width 8), and 10,729
(static/adaptive). Recorded training time is 449.6 s for original, 524.2 s for
fixed width 8, 656.6 s for static, and 1,096.4 s for adaptive; recorded peak GPU
allocation is 53.2, 53.2, 56.9, and 57.5 MB.
The current masked-convolution implementation is intentionally readable rather
than optimized, so runtime is an important negative engineering trade-off.
The corresponding time-memory plot is stored at
`results/adaptive_kernel_kernel_only/seed42/assignment_stress/kernel_resource_tradeoff.png`.
It makes the asymmetry explicit: adaptive routing takes 2.44 times the original
training time and 1.67 times the static-mixture time, while its peak allocation
is only 4.1 MiB above original and 0.5 MiB above static. Compute, rather than
memory, is the dominant measured engineering cost.

### Frozen decay-only intervention over 511 steps

The single prioritized follow-up froze the seed-42 independently gated,
width-4 model and evaluated balanced key-value assignments across a 511-step
SET-to-QUERY gap. The first three NOOPs were left unchanged so the SET's local
convolutional response could finish. For the remaining 507 NOOPs, only the
decay exponent was multiplied by \(\alpha\in\{0,0.1,0.25,0.5,0.75,1,1.25\}\):

\[
h_t=\exp(\alpha\Delta_tA)\odot h_{t-1}+\Delta_tB_tu_t.
\]

The input-write term, convolution, weights, and task readout remained exactly
as trained. This is an out-of-distribution causal diagnostic, not a proposed
replacement model. Evaluation used 256 held-out examples balanced over all 32
key-value combinations; the diagnostic linear probe was trained on 384
separate balanced-context examples and received both the final pre-query hidden
representation and queried-key identity.

| Decay multiplier | Task head | Linear probe | Median cumulative log10 retention |
|---:|---:|---:|---:|
| 0.00 | 24.6% | **45.3%** | -0.41 |
| 0.10 | **31.6%** | 44.9% | -3.36 |
| 0.25 | 26.6% | 41.4% | -7.70 |
| 0.50 | 30.1% | 39.1% | -14.93 |
| 0.75 | 25.0% | 35.9% | -22.15 |
| 1.00 (trained) | 25.0% | 34.0% | -29.38 |
| 1.25 | 25.0% | 34.0% | -36.62 |

Reducing attenuation significantly increases hidden-state decodability: at
\(\alpha=0\), probe accuracy improves by 11.33 points over the trained dynamics
(key-value-cluster bootstrap 95% CI +2.34 to +21.09), and at \(\alpha=0.1\) it
improves by 10.94 points (+3.13 to +19.14). This supports accumulated recurrent
attenuation as a causal source of information loss. However, the frozen task
head remains near chance. Its largest apparent gain is 6.64 points at
\(\alpha=0.1\), with a wide interval spanning -12.5 to +25.78 points, and its
correct-class probability stays near 0.25 with a negative target-logit margin
at every multiplier.

The combined result identifies two bottlenecks rather than one. Natural decay
destroys part of the recoverable value signal, while simply preserving more
state does not make that signal usable by the trained readout. With no decay,
unchanged NOOP input-write contributions accumulate in an unfamiliar state
distribution, explaining why probe decodability can improve while the frozen
task head and NLL do not. The next architectural experiment should therefore
train retention and readout jointly, rather than expecting an inference-time
identity transition alone to solve extreme retrieval.

### H5 - retention-readout mismatch

H5 predicts that reduced decay preserves a decodable value representation but
places it outside the distribution expected by the original query-conditioned
head. The falsification test froze every backbone parameter and trained a new
`Linear(32, 5)` readout separately at \(\alpha\in\{0,0.1,1\}\). Readout
training, validation, and testing used disjoint generated contexts (384, 128,
and 256 sequences respectively); no fixed test label was used for training or
selection. All heads used the same predeclared optimizer and 500-update budget.

| Decay | Original head | Matched retrained head | Gain | Retrained test NLL |
|---:|---:|---:|---:|---:|
| 0.0 | 25.8% | **99.6%** | +73.8 pp | 0.066 |
| 0.1 | 29.3% | **98.0%** | +68.8 pp | 0.111 |
| 1.0 | 25.0% | 39.5% | +14.5 pp | 1.249 |

The matched gain is decisive under reduced decay. At \(\alpha=0\), the paired
key-value-cluster bootstrap interval is +58.59 to +87.50 points; at
\(\alpha=0.1\), it is +52.34 to +83.98 points. In contrast, the natural-decay
head's +14.45-point gain has an interval of -5.86 to +33.59 points. Validation
accuracy independently reaches 99.2% for both reduced-decay heads, excluding a
test-only fitting explanation.

Cross-decay evaluation provides the strongest evidence that this is alignment,
not merely a generally better head. The head trained at \(\alpha=0\) scores
99.6% at \(\alpha=0\) but only 25.4% at \(\alpha=0.1\) and 25.8% at
\(\alpha=1\). The \(\alpha=0.1\) head scores 98.0% on its matched dynamics but
34.0% and 25.0% at \(\alpha=0\) and \(\alpha=1\). Thus each modified decay
setting produces a highly informative but differently oriented hidden
representation.

| Head training condition | Test \(\alpha=1\) | Test \(\alpha=0.1\) | Test \(\alpha=0\) |
|---|---:|---:|---:|
| Original, lengths 32--128 | 25.0% | 29.3% | 25.8% |
| 511-step, \(\alpha=1\) | **39.5%** | 31.3% | 25.0% |
| 511-step, \(\alpha=0.1\) | 25.0% | **98.0%** | 34.0% |
| 511-step, \(\alpha=0\) | 25.8% | 25.4% | **99.6%** |

H5 is strongly supported for this controlled diagnostic. Lower attenuation
does preserve enough information for almost perfect linear retrieval, yet the
original classifier, optimized jointly with the backbone on sequences of
length 32--128, does not generalize effectively to the final representations
encountered after 511-step dependencies. A classifier trained directly on
those long-horizon representations performs substantially better. P3 and the
retrained heads receive long-horizon supervision that the original head never
received, so they are diagnostic decodability results rather than fair
replacement-model improvements under the assignment's original training
budget. This does not establish that inference-time decay modification is a
valid final model: the intervention and retrained head are specialized to long
pure-NOOP gaps. The appropriate architectural follow-up is joint training
across ordinary and long-gap conditions so retention dynamics and retrieval
remain mutually calibrated without sacrificing overwrite or in-distribution
behavior.

#### Short-horizon frozen-head control

To separate long-horizon supervision from the fact that the original head was
jointly optimized with the backbone, an additional `Linear(32,5)` head was
trained on frozen final features from the original length-32--128 assignment
training split. The control used 384 balanced known-value queries, matching the
long-horizon head's sample count, initialization seed, 500-update optimizer
budget, and five-class architecture. Its disjoint short-horizon validation
accuracy was 71.1%; the original jointly trained head reached 75.8% on the same
balanced validation subset.

| Evaluation decay | Original joint head | Separate short-horizon head | Matched 511-step head |
|---:|---:|---:|---:|
| 0.0 | 25.8% | 27.7% | **99.6%** |
| 0.1 | 29.3% | 26.2% | **98.0%** |
| 1.0 | 25.0% | 25.0% | **39.1%** |

The separately optimized short-horizon head does not improve 511-step
generalization over the original head. Their paired differences are +1.95
points at \(\alpha=0\) (95% cluster-bootstrap CI -13.67 to +17.97), -3.13 at
\(\alpha=0.1\) (-18.36 to +12.11), and 0.00 at \(\alpha=1\) (-9.38 to +9.38).
By contrast, matched long-horizon supervision improves over the short-horizon
control by 71.88 points at both reduced-decay settings, with intervals excluding
zero. The large H5 gain is therefore attributable to adapting the classifier
to the long-horizon feature distribution, not merely to training a fresh head
separately from the backbone. At natural decay the +14.06-point advantage has a
wide interval spanning zero, so that smaller effect remains inconclusive.

#### Short-only horizon-robust readout

The next experiment asked whether one classifier can be learned from supervised
length-32--128 examples and transferred to 511-step dependencies without using
any 511-step labels. Diagnostics confirm a large feature-distribution shift:
the class-conditional mean shift is 0.66--0.85 short-feature standard deviations
per coordinate, the relative within-class covariance gap is 1.19--1.96, and the
original classifier's mean correct-class margin is negative at every tested
decay setting.

Simple transformations learned entirely from paired short-sequence decay
perturbations did not transfer. Neither short-statistics normalization nor a
32D affine least-squares map improved long accuracy beyond 28.1%. Symmetric-KL
and detached-teacher consistency objectives were tested at
\(\lambda\in\{0.1,1,5\}\), using identical initialization and 500-update
budgets. All variants remained between 25.0% and 28.1% on the 511-step tests,
despite retaining 69.5--71.9% natural short-validation accuracy. Short-horizon
decay perturbations therefore do not reproduce the horizon-induced feature
shift that matters at 511 steps.

An explicitly unlabeled long-horizon calibration control gives a different
result. CORAL estimated global mean and covariance from a separate balanced
511-step calibration set, never from the labeled test set, and aligned test
features to the short training distribution. Accuracy rose from 27.7% to 71.1%
at \(\alpha=0\), from 26.2% to 72.7% at \(\alpha=0.1\), and from 25.0% to 36.3%
at \(\alpha=1\). Mean/std alignment alone was insufficient. This establishes
that much of the reduced-decay mismatch is second-order and globally
correctable, but it is not a strict short-only solution because CORAL observes
unlabeled examples at the target horizon. The strict short-only consistency
approach is falsified in this tested form; learning horizon-invariant features
without any long-horizon exposure remains unresolved.

#### Class-conditional horizon geometry training

A separate three-model experiment tested whether geometry stabilization during
ordinary training can solve this problem without any supervised or unsupervised
511-step exposure. All models used the independently gated width-4 architecture,
the same seed-42 initialization, identical length-32--128 data order, AdamW
settings, and 1,500-update budget. Query gaps inside each training batch were
divided into \(\leq8\), 9--32, and \(>32\) bins. The proposed loss aligned
class-conditional means and covariances across bins; the ablation aligned global
covariance without labels. The geometry coefficient was fixed at one.

| Objective | ID validation | Untouched 511-step | Short class-geometry loss | Short-to-long covariance gap | Mean long margin |
|---|---:|---:|---:|---:|---:|
| Cross-entropy | **80.3%** | 25.0% | 0.0790 | 1.191 | -0.90 |
| + class-conditional geometry | 74.2% | 27.0% | **0.0094** | **0.892** | **-0.53** |
| + global covariance | **81.7%** | 25.0% | 0.0460 | 1.282 | -1.08 |

The class-conditioned objective reduced its held-out short-horizon geometry
loss by 88% and reduced the short-to-long covariance mismatch by 25%. It also
made the long-horizon margin less negative. Nevertheless, 511-step accuracy
increased by only 1.95 points, with a paired key-value-cluster bootstrap 95%
interval from -13.28 to +17.58 points, while in-distribution validation accuracy
fell by 6.1 points. The global ablation produced exactly zero accuracy change
(95% interval -15.63 to +15.63) and worsened long covariance mismatch.

This satisfies the stated falsification criterion: the proposed regularizer
strongly stabilizes geometry over the horizons actually observed during
training, but retrieval remains statistically indistinguishable from chance at
511 steps. Short-range output geometry stabilization is therefore not
sufficient in this form. The evidence points back to controlling the recurrent
transition dynamics over the full retention interval, rather than expecting a
constraint observed only within 128 steps to extrapolate across 511 recurrent
updates.

### H5b - internal SSM readout bottleneck

H5b asks whether long-horizon value information remains in the 512-dimensional
recurrent state but becomes less accessible through the input-dependent
\(C_t\) contraction and later output transformations. The frozen forward pass
was instrumented and numerically checked against the model's normal final
hidden representation. Three five-class probes were then trained independently
at each decay setting using identical disjoint train, validation, and test
sequences:

- P1: flattened \(64\times8\) recurrent state plus queried-key one-hot (520D),
- P2: \(C_th_t\) before the direct \(D u_t\) path (64D),
- P3: final normalized backbone feature \(z_t\) (32D),
- P4: the original frozen task head on \(z_t\).

No probe shares parameters, losses, or gradients with another, and the backbone
is frozen. Because the probe dimensions differ, an additional control projects
P1 to 24 PCA components plus the 8D key and P2 to 32 PCA components, giving
each controlled readout exactly 32 inputs and 165 parameters, matching P3.

| Decay | P1 state + key | P2 \(C_th_t\) | P3 final feature | P4 original head |
|---:|---:|---:|---:|---:|
| 0.0 | 86.3% | 67.6% | **99.6%** | 25.8% |
| 0.05 | 66.0% | 51.2% | **98.4%** | 32.4% |
| 0.1 | 59.8% | 48.4% | **97.7%** | 29.3% |
| 0.2 | 55.1% | 43.8% | **94.9%** | 26.2% |
| 0.3 | 52.0% | 41.4% | **85.9%** | 28.1% |
| 0.4 | 47.3% | 44.1% | **74.2%** | 25.0% |
| 0.5 | 43.8% | 43.0% | **62.1%** | 30.1% |
| 0.65 | 42.6% | 41.4% | **53.9%** | 25.0% |
| 0.8 | 43.4% | 40.2% | **46.9%** | 25.0% |
| 1.0 | 40.6% | 36.7% | 39.5% | 25.0% |
| 1.25 | 37.9% | 35.9% | 33.2% | 25.0% |

The dense multiplier sweep shows a smooth transition rather than a result
specific to three hand-picked settings. P3 remains above 85% through
\(\alpha=0.3\), declines to 62.1% at \(\alpha=0.5\), and approaches the other
internal probes by \(\alpha=0.8\)--1.0. Across the entire sweep, P4 remains
between 25.0% and 32.4%. Reduced attenuation therefore changes final-feature
linear separability continuously, while the original short-horizon classifier
does not track that changing representation.

At reduced decay, P1 exceeds P2 by 18.75 points for \(\alpha=0\) (95% paired
cluster CI +10.54 to +26.95) and 11.33 points for \(\alpha=0.1\) (+3.91 to
+19.15). Thus the isolated \(C_th_t\) contraction contains less linearly
decodable value information than the full state. However, this does not create
the final bottleneck predicted by H5b: P3 reaches 99.6% and 97.7%, greatly
exceeding P2. The downstream direct path, query-conditioned gating, projection,
residual, and normalization make the value almost perfectly linearly separable.

The parameter-matched controls lead to the same qualitative conclusion. At
\(\alpha=0\), state-PCA24-plus-key reaches 67.6%, SSM-output-PCA32 reaches
51.2%, and final-feature-32D reaches 99.6%. At \(\alpha=0.1\), these are 52.0%,
47.3%, and 97.7%. The high-dimensional state probe therefore benefits from
capacity, but dimensionality cannot explain the final feature's advantage.

Under the trained decay \(\alpha=1\), P1, P2, and P3 cluster at 36.7--40.6%;
the P1-P2 difference is only 3.91 points with an interval spanning zero. This
does not support \(C_t\) as the dominant natural long-horizon bottleneck. H5b is
therefore weakened: \(C_t\) causes a measurable drop under reduced-decay
interventions, but the complete backbone preserves and reorganizes the signal.
The largest verified failure remains the failure of the original task
classifier to generalize to these long-horizon final-feature distributions.
These are long-horizon-supervised linear-decoding results, not fair
replacement-model comparisons or direct estimates of mutual information;
nonlinear transformations can improve linear separability without creating new
value information.

## Overall conclusions

1. **Long-horizon degradation is real and reproducible.** H1 is supported over
   three training seeds. Retrieval declines sharply beyond the training range
   and reaches four-value chance at a 500-step delay.
2. **Unrelated activity creates genuine interference.** H2 is supported over
   five seeds at delays with usable baseline dynamic range. Both generic active
   tokens and the additional act of writing an unrelated value contribute.
3. **Retention and overwrite form a conditional, finite-horizon trade-off.**
   H3 is not a universal monotonic law, but slower passive dynamics can improve
   delay-128 retention while making a recently overwritten value harder to
   replace at shorter post-overwrite delays.
4. **Late NOOP transitions should be closer to identity.** H4 receives strong
   causal support only after respecting the width-4 convolutional washout. An
   immediate freeze is confounded because it truncates the SET write.
5. **Write routing is not the main solution.** Key-conditioned routing fails to
   produce a practically meaningful advantage over a tied parameter-matched
   table, and suppressing unrelated writes in a frozen model is actively
   harmful.
6. **Independent decay control is useful but incomplete.** It improves several
   ordinary, delay, and interference metrics and is better than a coupled gate,
   yet it still reaches chance at extreme delays and can increase stale-value
   errors when the old association remains strong.
7. **Local-kernel adaptation does not solve extreme memory.** The adaptive
   causal kernel learns event-dependent widths but does not beat the static
   mixture on the decisive matched stress comparisons.
8. **Preservation and retrieval must be co-designed.** Reduced decay preserves
   highly decodable information, but the original classifier is misaligned to
   the resulting long-horizon features. Long-supervised readouts recover the
   signal, whereas short-only consistency and affine methods do not.
9. **Short-range geometry regularization is insufficient.** The final proposed
   regularizer reduces held-out short-horizon geometry loss by 88% and improves
   covariance alignment, but leaves 511-step accuracy statistically
   indistinguishable from chance. The next mechanism should control recurrent
   dynamics across the actual retention interval rather than only constrain
   observed short-horizon outputs.

## Scope and limitations

The H1 analysis uses three model seeds; H2, H3, H4, and the frozen transition
interventions use five. The architectural routing, independent-decay,
convolution-width, adaptive-kernel, H5, and horizon-geometry training studies
currently use seed 42 unless explicitly stated otherwise. Paired-example or
cluster-bootstrap intervals quantify evaluation uncertainty but do not replace
training-seed replication. Inference-time decay, identity, convolution reset,
write suppression, and CORAL experiments are diagnostics of frozen systems,
not benchmark-ready replacement models. Probe accuracy demonstrates linear
decodability under its stated supervision and capacity; it is not an estimate
of mutual information or a fair improvement under the original training
budget.

## Reproducibility and artifact index

The repository includes the assignment dataset generator, fixed generated
splits, experiment scripts, CSV/JSON summaries, checkpoints, and plots. The
current automated suite contains 33 passing tests; the only local warning is
that pytest cannot write its optional cache directory under the current Windows
permissions.

Key plots:

- [Extended baseline and H1 retention](results/extended/multiseed/retention_multiseed.png)
- [H2 accuracy by distractor density](results/h2_multiseed/h2_accuracy.png)
- [H2 paired effects](results/h2_multiseed/h2_effects.png)
- [Frozen write/decay intervention](results/interference_term_ablation/term_ablation_effects.png)
- [H3 multi-seed trade-off](results/h3_multiseed/h3_multiseed.png)
- [H4 NOOP identity diagnostic](results/h4_noop_identity/h4_noop_identity.png)
- [Mechanistic retention and decodability](results/mechanistic/mechanistic_analysis.png)
- [Part B routing comparison](results/part_b/seed42/comparison/part_b_h2_effects.png)
- [Learned independent-retention summary](results/selective_retention/seed42/comparison/selective_retention_summary.png)
- [Stale-value diagnostic](results/selective_retention/seed42/stale_value/stale_value_primary.png)
- [Convolution-width sweep](results/selective_retention/seed42/dconv_sweep/comparison/dconv_overwrite_sweep.png)
- [Convolution-history reset](results/selective_retention/seed42/dconv_sweep/conv_reset/conv_history_reset.png)
- [Gate-free adaptive versus static kernel](results/adaptive_kernel_kernel_only/seed42/comparison/adaptive_kernel_comparison.png)
- [Gate-free assignment stress metrics](results/adaptive_kernel_kernel_only/seed42/assignment_stress/assignment_stress_metrics.png)
- [Gate-free kernel training cost](results/adaptive_kernel_kernel_only/seed42/assignment_stress/kernel_resource_tradeoff.png)
- [Frozen decay-only sweep](results/decay_only_retention/seed42/decay_only_retention_sweep.png)
- [H5 matched and cross-decay readouts](results/h5_retention_readout/seed42/h5_retention_readout.png)
- [Short-horizon frozen-head control](results/h5_short_horizon_control/seed42/h5_short_horizon_control.png)
- [H5b internal-location probes](results/h5b_internal_readout/seed42/h5b_internal_readout.png)
- [Feature-alignment diagnostics](results/horizon_robust_readout/seed42/feature_alignment.png)
- [Short-only consistency objectives](results/horizon_robust_readout/seed42/horizon_robust_readout.png)
- [Class-conditional geometry training](results/horizon_geometry_training/seed42/horizon_geometry_training.png)

Primary scripts for the final diagnostic chain are
`decay_only_retention_sweep.py`, `h5_retention_readout.py`,
`h5_short_horizon_control.py`, `h5b_internal_readout.py`,
`horizon_robust_readout.py`, and `horizon_geometry_training.py`. Their detailed
numeric outputs are stored in the correspondingly named directories under
`results/`.
