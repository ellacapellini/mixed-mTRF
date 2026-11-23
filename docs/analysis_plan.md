# Analysis plan v1 -- freeze (git tag) BEFORE any real-data fit

Values below were chosen by the assistant, with the reasoning shown, because the
cost constraint was lifted. They rest on four things I could NOT verify and you must
confirm: (a) your number of patients N, (b) recording length per patient,
(c) your actual GPT-2 surprisal mean/SD, (d) effect sizes reported in the literature
(Dou et al. 2025, Gillis et al. 2021) -- check these before freezing the SESOI values in 5.

## 1. Estimand
Population-level (patient-cluster) effect of z-scored word surprisal on the N400-window
response: gain `g` (fractional amplitude change per SD), shift (ms per SD), width `omega`
(log-width change per SD), from `erp_testbed.py::decompose`.
- z-scoring: ONE global constant (mean/SD of the stimulus's GPT-2 surprisal).
- Decomposition window: **250-600 ms**, fixed. Why: N400 is classically ~300-500 ms; the extra
  margin lets a shift or a widening stay inside the window; fixed from literature, never from data.
- Unit of inference: the **patient**.
- Lag grid for ALL fits: **0-800 ms, dt = 10 ms (81 lags)**. Why: covers the N400 and its tail; cost is acceptable now.

## 2. Splits (lock first) -- UPDATED: N >= 8 patients, >= 2 h each
A patient-level split is NOT affordable at N = 8 (ceil(0.35*8) = 3 development / 5 confirmatory patients
leaves a t-interval with df = 4 and almost no power). Instead:
- **Within-patient time split**: first **30%** of each patient's speech = development; last **70%** = confirmatory.
  Contiguous blocks (never interleaved: adjacent samples share stimulus history).
- All choices (kernel, priors, covariates, lag grid, ridge strength) are made on development blocks + the testbed only.
- Confirmatory estimates and leave-one-patient-out (LOPO) use ONLY the confirmatory blocks.
- Honest limitation: the SAME patients appear in both parts, so choices are not tested on new patients; with N = 8
  that is the price of keeping power. State it.
- Confirm that "2 h" means 2 h of SPEECH. Gillis et al. have ~150 words/min (2257 words per 15.1 min), so 2 h of
  speech ~ 18,000 words per patient; much less if the 2 h includes silence or other tasks.
- Assignment of blocks written to disk before any fitting.

## 3. Covariates
Surprisal effects are always reported adjusted for: log word frequency, word duration, position in
sentence, acoustic onset/envelope; unadjusted shown alongside. Testbed: unadjusted gain 0.18 vs 0.02
adjusted when the truth was on a confound with rho = 0.6.

## 4. Models and eligibility
- M0 (no surprisal); rERP + decomposition (model-light primary analysis); Model A (A-C1, A-C2).
- **Model B is eligible only if it has a population-level mean TRF** (h_pop ~ GP, units = h_pop + deviation).
  IMPLEMENTED in `models/bayesian_pop_trf.py` (functional mixed model, non-centered, sufficient-statistic
  likelihood so cost does not depend on recording length). `kind="raised_cosine"` gives Model A in the SAME
  model, so A vs B differ only in the level prior. v1 shares one length scale per component across levels; the
  "patients differ in smoothness" hyperparameter hierarchy is deferred until v1 passes the gates.
  **UPDATE: the hyperparameter hierarchy is KEPT as Model B's defining feature** (decision: you). It is implemented as
  `hierarchy="hyperparameter"`: h_i = h_pop + d_i, d_i ~ GP(0, k(ell_i, sd_i)), log ell_i / log sd_i = global + patient + electrode
  deviation (non-centred; HalfNormal(0.3) on the deviation SDs, NOT HalfCauchy, because the heavy tail let the patient-level SD
  drift to ~10 in the ablation). Model ladder, identical likelihood/priors elsewhere:
    A-pop (basis, amplitude pooling) -> B-pop-amp (GP, amplitude pooling, one shared length scale) -> B-pop-hyp (GP, hyperparameter hierarchy).
  Each step must earn its place: B-pop-hyp is adopted over B-pop-amp only if it passes 5d AND recovers a known patient-level
  smoothness difference in the testbed (`--patient_width_sd`).
  Why: as built, B has no shared mean, so a held-out patient's prediction is just the zero-mean prior,
  and the leave-one-patient-out comparison (section 7) cannot be run. This is a required code change
  before any A-vs-B comparison.
- C2 (surprisal in the kernel) is NOT tested on real data in its current unit-mean form: with one shared
  audiobook every unit has the same mean surprisal, so delta is identified only by the intercept prior.
  Revisit only after a word-level or envelope-based definition exists.

## 5. Gates and claims

### 5a. Calibration gate (must pass BEFORE any real-data claim) -- AMENDED after the 30-rep calibration run
Finding (testbed, 6 patients x 4 electrodes, 30 reps/scenario): errors are BIAS-dominated, not interval-width-dominated.
| scenario | bias (gain / shift ms / width) | sampling SD |
|---|---|---|
| null | -0.001 / +0.1 / +0.002 | 0.008 / 0.7 / 0.005 |
| gain 0.30 | +0.016 / -0.1 / +0.033 | 0.010 / 0.8 / 0.006 |
| latency 25 ms | -0.062 / -1.6 / +0.096 | 0.009 / 1.1 / 0.007 |
| width 0.25 | +0.036 / +4.5 / -0.082 | 0.010 / 1.1 / 0.008 |
Consequences:
1. Patient-cluster bootstrap undercovers even in the null (0.63-0.80 at 90% nominal; half-width ~1.1 SD instead of ~1.6).
   The patient-level t-interval (df = P-1) is closer (0.80-1.00 in the null). Use the t-interval, not the bootstrap.
2. In effect scenarios NO interval can reach nominal coverage, because bias is 4-8x the sampling SD
   (first-order approximation + ridge shrinkage + cross-talk between gain/shift/width).
3. Cross-talk is the real hazard: a pure latency effect leaks +0.096 into "width" (SESOI is 0.10), a pure width
   effect leaks +4.5 ms into "shift". A claim about WHICH mechanism is therefore unsafe until bias is reduced.
Gate (replaces the old coverage-only rule), evaluated over >= 100 seeds:
 (i) null: patient-t 90% interval covers 0 in 85-95% of seeds, for each of gain/shift/width;
 (ii) effect scenarios: |bias| < 0.3 x SESOI for every parameter INCLUDING the off-target ones (cross-talk);
 (iii) until (ii) holds, mechanism-specific claims (gain vs latency vs width) are NOT allowed; only
       "surprisal modulates the response" (M1 vs M0) may be claimed.
Candidate bias fixes to try (not yet tested): fit h(tau|z) in z-bins and estimate shift by cross-correlation and
gain by amplitude ratio; a second-order decomposition; simulation-based bias correction (simulate at the estimate,
invert the bias map); lighter ridge on h1 than on h0.

### 5b. Smallest effects of interest (SESOI) -- **proposed values, check against literature**
gain +/-0.10 per SD; shift +/-10 ms per SD; width +/-0.10 per SD.
Why: below one sample (10 ms) a latency shift is hard to defend; 10% amplitude/width change is a
modest, defensible floor. Testbed noise is far smaller (6 min/unit: sd gain 0.01-0.02, shift ~0.5 ms),
but real data will be noisier, so do not take the testbed as proof these are detectable.

### 5c. How a result is worded (replaces "interval includes 0")
- 90% interval entirely outside the SESOI band and excluding 0 -> **effect present**.
- 90% interval entirely inside the SESOI band -> **evidence of (practical) absence**.
- Otherwise -> **inconclusive**. Say so; do not narrate it as null.

### 5d. Model B adoption criteria (all required, evaluated on the testbed)
1. Paired over seeds, B lowers leave-one-patient-out error OR decomposition RMSE vs Model A by >= **10%**,
   with the 95% interval of the paired difference excluding 0. Why 10%: large enough that it is not
   Monte-Carlo noise at the seed counts below; small enough to matter.
2. B is never worse than A by more than **5%** in any scenario (null included).
3. Convergence on the predicted signal and h(tau): R-hat < 1.01, bulk ESS > 400, zero divergences, in
   >= **95%** of cells (strict because compute is not the constraint).
4. Variance components identified at the real number of patients (posterior SD clearly below prior SD).
5. B is not worse than the simple rERP decomposition at recovering gain/shift/width.
6. False-positive rate under the null (effect claimed per 5c) <= **10%**.
If any fails: report Model A + rERP; Model B becomes future work.

## 6. Confirmatory contrasts (the only ones that may be called confirmatory)
(a) surprisal vs none (M1 vs M0); (b) gain-only vs gain+shift+width (nested); (c) A vs B (only if 5d passed).
Everything else is exploratory and labelled.

## 7. Evaluation
Primary: leave-one-patient-out. Secondary: unit-level LOO ("new electrode"). Always print the noise
ceiling sqrt(SNR/(1+SNR)) beside held-out r. Units with bad R-hat / BFMI / Pareto-k are reported
separately, never silently dropped.

## 8. Compute plan (Euler)
Stage 1 (cheap, ridge): `erp_testbed.py`, >= 100 seeds, full scenario matrix.
Stage 2 (Bayesian, screening): Model A vs B-with-mean on
{null, gain, latency, width} x {1/f + shared noise, half-responsive electrodes, shared stimulus} x 6 min/unit,
**30 seeds** (Monte-Carlo SE of a win-rate ~ 9%; adequate for screening).
Stage 3 (confirmation): **100 seeds** only for cells that pass screening (SE ~ 5%).
Sizing, from my small test (4 units, 81 lags, max-treedepth saturated, ~2.5 s/iteration): request
>= 24 h, 4 CPUs, 16 GB per Bayesian task and CALIBRATE with one task first; non-centered GP draws
should cut this. Do not run the full factorial of Bayesian fits (120 cells x 100 seeds x 2 models is
~24,000 fits).

## 9. Forking-path log
Keep a dated table of EVERY model/prior/kernel/lag/covariate variant fit, its reason and its
development-set result. Report all of it. Never delete a failed variant.

## 10. Stated limitations
GPT-2 is not the listener's predictive model; a stationary-kernel length scale is not response duration;
ridge shrinkage biases the decomposition toward 0 (testbed: shift 23 vs 25 ms, width 0.17 vs 0.25);
testbed templates and surprisal distribution are assumptions; C2 as built cannot be tested on shared-stimulus data.

## 11. Anchors from Gillis et al. 2021 (what the paper actually supports; scalp EEG, 29 participants, Dutch audiobooks)
- Word-level surprisal response: central negativity ~300-400 ms (topography window 242-484 ms; significant range 242-531 ms),
  plus an early positive frontal ~50 ms. TRF window used: 0-900 ms. => my 250-600 ms decomposition window is consistent.
- Size: adding the significant linguistic predictors raised prediction accuracy by 3.4e-4 in Pearson r (1.05% relative)
  over an acoustic + segmentation baseline of r ~ 0.03; word-level alone ~1e-4. Effect sizes (Wilcoxon z/sqrt(N)):
  word surprisal 0.49, word frequency 0.38, phoneme surprisal 0.70. => the REAL scale of the effect on prediction is
  1e-4 to 1e-3, 10-100x smaller than my testbed's LOPO gains (which used SNR 0.2, r ceiling 0.41).
  The testbed must therefore be run at SNR ~ 0.005-0.03 as well. (They are scalp EEG; iEEG SNR is unknown to me.)
- Word frequency and word surprisal each add unique variance and have different topographies => frequency is a real
  confound/co-predictor; always include it. Function words' ~200 ms positivity was contaminated by the next word's onset
  (mean ISI 239 ms) => short-ISI words need an explicit next-word-onset control.
- They claim added value via held-out prediction accuracy (cross-validated), not via TRF significance, because correlated
  predictors can give significant TRFs without adding information. => supports the plan's M1-vs-M0 LOPO contrast.
- Participants were the unit of inference (Wilcoxon across participants); generalisation to unseen stories failed in 2 of 6
  stories => expect story/segment variability; the within-patient time split should use several non-adjacent blocks per patient.
- Peak latency was taken as argmax|TRF| inside PRE-SPECIFIED windows per component (30-90, 90-180, 180-300 ms), not globally.
  => use a windowed argmax (e.g. 250-600 ms) as a robustness check next to the decomposition.
- Their surprisal is a Dutch 5-gram (log10); yours is GPT-2 (bits). Effect-size numbers do not transfer directly.

## 12. FINAL-RUN DESIGN (iEEG scale) -- Stage 1 implemented in `ieeg_stage1.py`
Design: 8 patients x 128 electrodes (iEEG typically 100-256; `--electrodes` is configurable), 2 h at 100 Hz, shared story,
1/f noise + within-patient shared noise. ASSUMPTIONS to revisit: 25% of electrodes responsive to speech, 50% of those modulated by surprisal.
- The patient remains the unit of inference (N = 8). More electrodes help electrode-level heterogeneity and selection; they do NOT add patients.
- Time split inside each patient (30% development / 70% confirmatory); electrodes are selected on DEVELOPMENT data (cross-validated
  surprisal-free R^2, top-k per patient, k=16), so selection cannot be driven by the surprisal effect. Confirmatory block untouched until then.
- Estimand = modulation among RESPONSIVE electrodes (non-responsive ones add noise to the pooled TRF, nothing to the surprisal term).
- Measured cost (my sandbox): one scenario run at full size = 72 s, 0.24 GB. A seed (3 SNR x 4 mechanisms) ~15 min; 100 seeds ~25 core-hours.
- Stage 2 (Bayesian ladder A-pop / B-pop-amp / B-pop-hyp) runs on the SELECTED electrodes only (8 x 16 = 128 units): all ~1000 units is infeasible
  (batched Cholesky + 160k latents). Extrapolated, NOT measured: amplitude ~1-4 h per chain, hyperparameter ~8-31 h per chain. Runner not yet written.
- Sensitivity grid (20 seeds each, `--tag`): electrodes 256; responsive_frac 0.10/0.40; select_k 8/32; common_noise 0.6; white noise.

### 12b. Spatial structure and the final-run split (added)
- Geometry: sEEG shafts (12 contacts, 3.5 mm) or ECoG grid. Noise covariance c*11' + (1-c)*exp(-d/L); L = 1.5 mm for high-gamma ('hg', ~0.10 correlation at
  3.5 mm, ~0 by 8 mm) and 15 mm for low-frequency/ERP-like ('lfp', 0.79 at 3.5 mm). Checked: empirical correlation matches the target to the sampling floor.
  Responsive/effect electrodes are spatially clustered (P(neighbour responsive | responsive) 0.64 vs 0.26 if independent).
- Both bands are run for the main result: the N400-like template is ERP/low-frequency, but many iEEG speech TRFs use high-gamma envelopes.
- NOT modelled: bipolar / Laplacian re-referencing or ICA (these reduce the LFP correlation); add if your pipeline uses them.
- Measured cost (sandbox, 1 core): full-size run (8 x 128 x 2 h, 1 mechanism) = 48 s, 0.95 GB; one seed (2 bands x 3 SNR x 4 mechanisms) ~ 20-30 min.
- Split: Mac = main seeds 0-29 + sensitivity variants; Euler = main seeds 30-99 (and later Stage 2, the Bayesian ladder).
- Merge + gates: `summarize_ieeg1.py` (needs >= ~100 seeds; at 6 seeds the Monte-Carlo error on a coverage is +/-0.20).

### 12c. Findings from the first 30 seeds (Mac, lfp/hg x 3 SNR; lfp snr=0.03 at 20 seeds) and the decisions they forced
- Null behaves: coverage of 0 = 0.90, false-positive rate 0.10 for gain/shift/width (MC error +/-0.11 at 20 seeds).
- Gain is recovered almost unbiased (0.133 vs 0.139; detected in 95% of seeds; LOPO R^2 gain > 0 in 100% of seeds). Global argmax calls it a
  176 ms latency shift; windowed argmax gives 4 ms.
- SPECIFICITY is the problem, not bias: in the latency scenario the (off-target) gain interval excludes 0 in 60% of seeds; width scenario 25% false shift,
  15% false gain; gain scenario 35% false width. => new gate (iii): false-mechanism rate <= 15% (summarize_ieeg1.py).
- The underestimate of shift (~72-78%) and width (~55-72%) is NOT ridge shrinkage: at SNR 30 (noiseless) the width estimate is still 60% of truth, and
  relaxing the ridge (`--ridge_c_mult 0.01`) changes nothing. It is a structural bias of the first-order decomposition, in part contamination of the window
  by the unmodulated P2 tail: narrowing 250-600 -> 300-520 ms raised width recovery 56% -> 72% and cut cross-talk. ~25% remains (second-order).
- DECISION (development, seeds 0-29): consider 300-520 ms as the primary decomposition window (N400 literature: 300-500 ms; Gillis et al.: 242-531 ms) with
  250-600 ms as sensitivity. This is a forking path: LOG IT. On REAL data the window must be set from the development block (e.g. around the estimated peak of |h0|),
  not from this simulation, because component positions differ.
- HYGIENE: seeds 0-29 are now DEVELOPMENT seeds (estimator was tuned looking at them). The final gate must use FRESH seeds (>= 100-299).

### 12d. Window comparison, lfp SNR 0.03, 20 seeds each (development seeds 0-19)
| | 250-600 ms | 300-520 ms |
|---|---|---|
| null: SD of estimates (gain / shift / width) | 0.015 / 1.3 / 0.011 | 0.027 / 2.0 / 0.020 (~1.7x noisier) |
| gain scenario: recovery; bias on shift, width | 0.96; -2.2 ms, +0.032 | 1.29; +4.0 ms, +0.049 (worse) |
| latency scenario: recovery; false gain / false width | 0.80; 60% / 50% | 0.88; 35% / 10% |
| width scenario: recovery; false gain / false shift | 0.56; 15% / 25% | 0.79; 10% / 10% |
Reading: a narrower window removes leakage from the unmodulated P2 tail and improves width/shift recovery, but costs variance and, in the gain scenario,
introduces an overestimate (+29%) and a +4 ms spurious shift (~3 SE). NO window passes every gate; latency -> false gain stays at 35% (interval ~[0.19, 0.55]).
The sign of that leakage (negative gain when the N400 is delayed) is what a second-order term of a shifted bump would give (untested). Do NOT adopt 300-520 ms as primary
on this evidence alone. Next development step: a parametric (gain/shift/dilation) estimator on z-bins = the Dou et al. (a,b,c)-style baseline the plan already calls for.
Gate verdicts are now three-state (PASS / FAIL / INCONCLUSIVE) so that a 0.15 false-positive rate at 20 seeds is not reported as a failure.

### 12e. 30-seed results (Mac, first-order estimator) and the binned parametric estimator
30-seed gates, first-order decomposition, selected electrodes (hg and lfp behave alike; SNR is what matters):
- Gain: recovery 0.93-1.01 and power 0.93-1.00 at SNR >= 0.03; essentially undetectable at SNR 0.005 (power 0.13-0.17).
- Shift: recovery 0.79-0.86 (bias -1.7 to -2.5 ms, inside the 3 ms tolerance) at SNR >= 0.03.
- Width: recovery 0.53-0.64 at SNR >= 0.03 (0.25-0.30 at 0.005); bias -0.045 to -0.057 FAILS the 0.03 tolerance with small error bars.
- Specificity tested as "interval excludes 0": false-mechanism rates 77-100% at SNR 0.2 (63% / 60% at SNR 0.03 in the latency scenario).
  The interval shrinks with SNR but the structural bias does not, so the first-order estimator is inconsistent: more data -> more false mechanism claims.
- Plan rule 5c (claim only if the whole interval is beyond the SESOI) is NOT the same test; `summarize_ieeg1.py` section (iv) now reports it.
New estimator `--estimator binned` (ieeg_stage1.py): surprisal-bin waveforms -> direct gain/shift/dilation fit to a reference waveform (nonlinear least squares,
window 300-600 ms) -> per-SD slope across bins -> patient-level t-interval. Head-to-head, 8 seeds, SNR 0.2, 8 patients x 48 electrodes x 40 min:
width recovery 0.67 -> 0.99, latency 0.90 -> 0.99, leakage into gain (width scenario) 0.022 -> 0.007, shift bias (latency) -1.5 -> -0.2 ms.
LIMITS: (a) 8 seeds, one SNR; low-SNR behaviour unknown (5 bins = noisier). (b) The simulation generates EXACTLY a gain/shift/dilation of the N400, so this tests
consistency when the family is right, NOT that the family is right on real data. A misspecified-truth scenario is the next test. (c) Seeds 0-29 are development
seeds; final gates need fresh seeds (>= 100).

### 12f. Binned estimator, 30 development seeds (Mac), and what the misspecification tests changed
Recovery (estimate / truth), selected electrodes, first-order -> binned: width 0.53-0.64 -> 0.83-0.94 (SNR >= 0.03); latency 0.79-0.86 -> 0.84-0.95; gain ~1.0 both.
Bias gate (ii) now PASSES (or is inconclusive) in every cell at SNR >= 0.03. At SNR 0.005 the estimator fails: gain recovery 0.20-0.24 (power 0.13-0.23),
+4.6 ms spurious shift in the gain scenario -- nothing about mechanisms is estimable at that SNR, and no false claims are made either.
Specificity: under "interval excludes 0" (iii) leakage of 0.01-0.02 (gain/width) is still 'significant' at high SNR (FAIL in several cells). Under the PLAN's rule 5c (iv) -- claim only
if the whole interval lies beyond the SESOI -- FALSE-MECHANISM CLAIMS ARE 0 IN ALL 18 CELLS. Power under (iv) is 0.00-0.73 and is capped by effect size vs SESOI: the simulated
population-average effect among selected electrodes (gain 0.14, shift 11.5 ms, width 0.117) is only 1.2-1.4x the SESOI, not by an estimator defect.
GATE CHANGE (post-hoc, LOG IT): (iii) is demoted to informational; (iv) is the specificity gate. Reason: (iii) was my addition and is stricter than the plan's own claim rule. This was decided
after seeing development seeds 0-29; the verdict must come from FRESH seeds (>= 100).
MISSPECIFICATION (8 seeds, SNR 0.2; truth outside the gain/shift/dilation family, so true gain/shift/width = 0):
  - all three mechanisms at once (in family): truth (0.19, 15.5 ms, 0.155) -> binned (0.20, 15.8, 0.18): separable.
  - late additive component: binned reports width -0.14 and shift -7.5 ms; width claimed beyond SESOI in 75% of seeds (first-order: 25%).
  - asymmetric N400 (slower DECAY = "recovery speed"): read as shift +6.8 ms and width +0.10 (0% beyond SESOI). A symmetric dilation cannot express recovery speed;
    add a two-sided dilation (left/right) to test that hypothesis directly.
FAMILY-ADEQUACY DIAGNOSTIC (share of the surprisal-dependent waveform change explained by the fitted transform, 6 seeds, SNR 0.2): in family 0.93-0.99 (min 0.90); late 0.71-0.80; decay 0.87-0.91.
Separates 'late' cleanly and 'decay' partly. It falls with noise, so a threshold must be calibrated per SNR by simulation before use. Proposed rule: report mechanism-specific results only if adequacy >= the
SNR-matched simulated in-family range.
OPEN: the REAL SNR. Estimate it from your own data: SNR ~ r^2 / (1 - r^2), r = held-out correlation of the speech-only TRF (surprisal-free) in the selected electrodes, development block.
  r 0.07 -> SNR 0.005 (no mechanism claims possible); r 0.17 -> 0.03; r 0.41 -> 0.2.

### 12g. Verdict rules corrected BEFORE the fresh-seed (100-199) results were examined
The null-coverage verdict required the whole 90% interval of the observed rate to lie inside [0.85, 0.95]; a perfectly calibrated procedure (true coverage 0.90) has an interval of
about [0.84, 0.94] at 100 seeds, so it could never PASS. Replaced by: coverage PASS if the interval contains 0.90 and its lower bound >= 0.80, FAIL if it lies wholly below 0.85 or above 0.95;
false-positive rate PASS if its lower bound <= 0.10 and upper bound <= 0.20, FAIL if its lower bound > 0.10. Specificity (iv) unchanged: PASS if the upper bound <= 0.15.
Fresh seeds 100-199 (binned estimator, tag _fin; 600 jobs, none failed) are summarised with `summarize_ieeg1.py --table` using these rules. No estimator change after looking at them without logging it here.

### 12h. FRESH-SEED RESULTS (seeds 100-199, binned estimator, 100 seeds, both bands, 3 SNRs) -- first verdict
Null: coverage PASS/INCONCLUSIVE (a few 0.96-0.97 = slightly conservative), false-positive rate PASS in 17/18 cells (INCONCLUSIVE 1), and ZERO null seeds beyond the SESOI in all 6 cells.
Bias (ii): SNR 0.2: PASS everywhere. SNR 0.03: PASS except gain->shift (+3.4 ms, tolerance 3) and gain->width (0.026, tolerance 0.03) INCONCLUSIVE. SNR 0.005: FAIL (gain recovery 0.16-0.18; shift -4 to -5 ms; width recovery 0.45-0.47).
Specificity (iv, plan rule 5c): false-mechanism claims = 0 in all 18 cells (PASS). Power (iv) at SNR 0.2: gain 0.54-0.63, latency 0.14-0.21, width 0.20-0.26; at 0.03: 0.09-0.21; at 0.005: ~0.
PAIRED binned vs first-order on the same data (SNR >= 0.03): width recovery 0.91-0.94 vs 0.58-0.64; latency 0.92-0.94 vs 0.78-0.85; gain equal; first-order never claims width (power 0.00).
(iii) 'excludes 0' still shows persistent small leakage (latency -> gain -0.01..-0.02, latency -> width +0.016..+0.019; false 'significant' 25-100% at SNR >= 0.03). Reporting effect sizes against the SESOI, not significance, is therefore REQUIRED.
Family adequacy (in-family truth): 0.96-0.99 at SNR 0.2, 0.85-0.87 at 0.03, 0.77-0.80 at 0.005 -> it falls with noise, so an absolute threshold is meaningless; it must be referenced to simulation at the estimated real SNR.
CAVEAT -- THE (iv) PASS IS PARTLY VACUOUS: power is low, so "no false claims" is achieved partly by rarely claiming anything. The simulated population-average effects among selected electrodes (gain 0.139,
shift 11.5 ms, width 0.117) are only 1.2-1.4x the SESOI, so even at recovery 0.93 the estimates sit at/near the SESOI and the whole 90% interval (df = 7 patients) rarely clears it. Next experiment: `--effect_scale`
(0.5, 2, 3) x SNR to map power against effect size (the "identifiability map" promised in section 9).

### 12i. Strategic re-prioritisation (decision log) and the rhythm/oddball scenario
DECISION: Stage-1 simulation work has reached diminishing returns. The estimator (binned) and gates are frozen as "Stage-1 v1" (further effect-scale power maps are optional). Priorities now:
 (1) reorganise the code into a documented package with one-model-at-a-time execution and regression tests; (2) real-data positive control on ERP CORE (MMN/P3 oddball: known rare-event surprise, p = .2);
 (3) estimate the REAL SNR; (4) Stage 2 (A-pop vs B-pop-amp vs B-pop-hyp); (5) Proix iEEG with the dev/confirm split; (6) data-efficiency ("learning") curves at each stage.
Rhythm/oddball scenario (`ieeg_stage1.py --rhythm isochronous --period_ms 300 --violation_p P --jitter_ms 90`): events on a strict grid, a fraction P displaced in time; surprisal = ideal-observer -log2 p (binary).
 - Identifiability (81 lags = 800 ms, period 300 ms = 30 samples): strictly periodic -> only 30/81 lag directions identifiable (the TRF is determined only modulo the period); 2% violations already restore full rank.
 - Recovery, single seed, 6 patients x 30 electrodes x 25 min, SNR 0.2, binned (2 bins): p=0.15 gain 0.13 (truth 0.10), shift 12.1 (8.7), width 0.12 (0.09); p=0.05 gain 0.12, shift 9.8, width 0.088.
   first-order: gain 0.16-0.18 and leakage (latency -> gain -0.07..-0.10). INDICATIVE ONLY (one seed).
CLARIFICATION: the TRF models do not detect or predict rhythm; they estimate how the response depends on a SUPPLIED surprisal regressor. The surprise computation (ideal observer / GPT-2) is a separate model.
