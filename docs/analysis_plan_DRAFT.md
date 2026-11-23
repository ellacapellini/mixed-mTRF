# Analysis plan (DRAFT v0) -- commit this BEFORE touching the real recordings

Items marked **[YOU DECIDE]** are placeholders. I proposed values only so the
document is concrete; they are not derived from anything. Change them, then
freeze the file (git tag) before any real-data fit.

## 1. Estimand
Population-level (patient-cluster) effect of **z-scored word surprisal** on the
N400-window response, expressed as three numbers per 1 SD of surprisal:
gain `g`, latency shift `shift` (ms), width `omega` (log-width), from the
first-order decomposition in `erp_testbed.py::decompose`.
- z-scoring uses ONE global constant (mean/SD of the stimulus's GPT-2 surprisal), never per-unit.
- Window for the decomposition: **[YOU DECIDE]** (proposal 250-600 ms), fixed from the literature, not from the data.
- The unit of inference is the **patient**, not the electrode.

## 2. Data splits (lock first)
- Assign patients to *development* and *confirmatory* by a hash of patient ID, saved to disk, before any model sees data.
- Proposal: ~1/3 development, ~2/3 confirmatory **[YOU DECIDE]**. All model choices (kernel, priors, lag window, covariates) are made on development patients only.
- Confirmatory patients are touched once per pre-named contrast (section 6).

## 3. Covariates (surprisal is confounded)
Always report the surprisal effect **adjusted** for: word frequency, word length/duration,
position in sentence, acoustic onset/envelope. Show unadjusted next to adjusted.
Testbed evidence (`erp_testbed.py --effect_on confound --confound_rho 0.6`):
unadjusted surprisal gain 0.18 vs 0.02 once the confound is in the model.
GPT-2 surprisal is also not the listener's predictive model; say so.

## 4. Models and their roles
- M0: no surprisal (outside baseline).
- rERP / gain-shift-width decomposition: the model-light primary analysis.
- Model A (A-C1 restricted vs A-C2 free): main hierarchical analysis.
- Model B: **only** if the criteria in section 5 are met on the testbed first.

## 5. Decision criteria for Model B (written BEFORE running it on real data)
Model B is adopted over Model A only if ALL hold on the ERP-shaped testbed
(realistic SNR, 1/f + shared noise, shared stimulus, 800 ms window, heterogeneous effects):
1. It lowers patient-level LOPO error or decomposition error vs Model A by at least **[YOU DECIDE: 10%]**, over at least **[YOU DECIDE: 20]** seeds.
2. It converges: R-hat < 1.01 and bulk ESS > 400 on the predicted signal and h(tau) (functionals), no divergences, in >= **[YOU DECIDE: 90%]** of cells.
3. Its extra structure is identified at the real number of patients: variance components' posterior SD clearly below prior SD.
4. It is not worse than the simple rERP decomposition at recovering gain/shift/width.
If any fails: report Model A + rERP, and state Model B as future work.

## 6. Confirmatory contrasts (the only ones allowed to claim "significant")
(a) surprisal vs no surprisal (M1 vs M0), leave-one-patient-out;
(b) gain-only vs gain+shift+width (nested);
(c) Model A vs Model B (only if section 5 passed).
Everything else is exploratory and labelled so.

## 7. Evaluation
- Primary: leave-one-patient-out prediction. Unit-level LOO is secondary (it answers "new electrode", not "new patient").
- Always print the noise ceiling (max attainable r given SNR) next to held-out r.
- Units with bad R-hat/BFMI/Pareto-k are **reported separately, not dropped** (dropping them biases toward electrodes with strong effects).

## 8. What counts as "no evidence"
Pre-state it: e.g. patient-cluster 90% interval for `g` includes 0 and |shift| < **[YOU DECIDE: 5 ms]**.
Pre-write the sentence you will publish if the result is amplitude-only.

## 9. Forking-path log
Keep a table (date, change, reason, development result) of EVERY model/prior/kernel/lag-window
variant fit. Report all of it. Do not delete failed variants.

## 10. Limitations to state up front
GPT-2 != listener; stationary-kernel length scale != response duration; ridge shrinkage
biases decomposition estimates toward 0 (testbed: shift 23 vs 25 ms, width 0.17 vs 0.25);
bootstrap CIs with ~6 patients under-cover (testbed: 0.4-0.9 vs 0.9 nominal).
