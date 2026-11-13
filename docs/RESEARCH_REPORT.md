# PREDICAI — Research & Feasibility Report (Phase 0)

> Date: 13 Nov 2026 · Status: **For owner approval — no implementation started**
> Scope: the 20 points required by the project brief, §17.
> Rule followed throughout: no performance, benchmark or accuracy number in
> this report is a PREDICAI result. Where numbers appear, they are *published
> literature values on the same benchmark*, cited as context for feasibility —
> PREDICAI's own numbers will exist only after Phase A4 / Phase D runs.

---

## 1. Understanding of PREDICAI

PREDICAI is a **Remaining Useful Life (RUL) regression system** for industrial
machines, built and validated on the NASA C-MAPSS turbofan simulation, and
packaged the way a real product would be:

- A **Python deep-learning lab** learns degradation patterns from multivariate
sensor sequences (LSTM) and estimates "how many operating cycles remain".
- The trained model is exported to **ONNX** and served by a **Rust edge
engine** (ONNX Runtime via the `ort` crate, Axum API) — the pattern used
when inference must run locally, close to machines, without a Python stack.
- A **maintenance-history module** (SQLite) tracks servicing per machine and
turns predictions into *reasoned, policy-based* recommendations.
- A **simulator** replays real FD001 test-engine data (there is no physical
turbine), and a **React dashboard** presents the whole system honestly —
simulated/synthetic data always labelled as such.

It is explicitly **not**: a failure classifier, a real-time system connected
to a real turbine, or a maintenance product with validated failure thresholds.
Its honest framing — and its strength in a viva / interview / portfolio — is:
*a complete, tested, explainable prognostics pipeline on the standard public
benchmark, with a production-shaped deployment path (ONNX + Rust).*

## 2. Overall technical feasibility — **FEASIBLE, with 3 flagged risks**

Every component uses mature, documented technology:

| Component | Feasibility | Basis |
|---|---|---|
| FD001 loading / preprocessing / windows | High | Trivially sized data (~20k train rows), standard pandas/numpy work |
| LSTM RUL regression | High | The most-replicated result in prognostics literature (since Zheng et al., 2017) |
| Evaluation (RMSE/MAE/NASA score) | High | Closed-form metrics; official protocol is well documented |
| ONNX export of an LSTM | High | LSTM operators are core ONNX; export is a standard PyTorch path |
| Rust inference via `ort` | High, **with a version risk** | `ort` wraps ONNX Runtime, but 2.0 is still in release candidates — pin exactly (§11) |
| Maintenance module (SQLite) | High | Ordinary CRUD + a rules engine |
| Simulator (replay) | High | Reuses the test files + the Rust API |
| Dashboard | High | Standard React/TS; the only real work is honest state handling |

Risks that shape the plan: preprocessing skew between Python and Rust,
data leakage (the classic way C-MAPSS scores get accidentally inflated),
and the `ort` RC-version situation. All three have concrete mitigations in
`plan.md` (single preprocessing artifact + parity tests; engine-wise splits;
exact version pin + `tract` fallback).

## 3. NASA C-MAPSS FD001 — dataset structure (verified)

Source: Saxena, Goebel, Simon & Eklund, *"Damage Propagation Modeling for
Aircraft Engine Run-to-Failure Simulation"*, PHM 2008; NASA Prognostics
Center of Excellence / NASA Open Data. Always obtain the data from the
official NASA source (or its Kaggle mirror, "NASA Turbofan Jet Engine
Dataset") and cite it — do not redistribute the raw files in the repo.

**The four subsets** (start with FD001 only):

| Subset | Op. conditions | Fault modes | Train engines | Test engines |
|---|---|---|---|---|
| **FD001** | 1 | 1 (HPC degradation) | 100 | 100 |
| FD002 | 6 | 1 | 260 | 259 |
| FD003 | 1 | 2 | 100 | 100 |
| FD004 | 6 | 2 | 248–249* | 249 |

\*Secondary sources disagree by one engine for FD004; irrelevant for FD001,
and a reminder to verify counts from the actual files in Phase A1.

**FD001 files:**
- `train_FD001.txt` — 20,631 rows; 100 engines, each run **to failure**.
- `test_FD001.txt` — 13,096 rows; 100 engines, each **truncated before failure**.
- `RUL_FD001.txt` — 100 numbers: true RUL at each test engine's last cycle.
- `readme.txt` — the dataset's own documentation (read it in A1, first).

**Each row = 26 whitespace-separated, header-less columns:**
1. `unit` (engine ID), 2. `cycle`, 3–5. three operational settings
(altitude, Mach number, throttle resolver angle — in FD001 the operating
condition is a single sea-level regime, so these carry little variation),
6–26. **21 sensors** (temperatures, pressures, fan/core speeds, ratios, etc.).

Known properties to *verify in EDA rather than assume*: no missing values are
expected; several sensors are constant or near-constant in FD001 — the
literature commonly drops 7 sensors (`s1, s5, s6, s10, s16, s18, s19`) leaving
**14 informative sensors**. PREDICAI will derive
its own list from measured variance in Phase A1 (Decision D-5) — if our
measured list differs, ours wins and the difference gets documented.

## 4. Correct RUL calculation

**Training engines** (run to failure). For engine *e* with life `T_e`:
```
RUL(e, t) = T_e − t
```

**Test engines** (truncated at `T_e^obs`, true remaining life `R_e` given in
`RUL_FD001.txt`). At any observed cycle *t*:
```
RUL(e, t) = R_e + (T_e^obs − t)
```
This is the single most common place C-MAPSS implementations go wrong:
applying the train formula to test data, or labelling only the last cycle.
Both formulas are unit-tested in Phase A2 on a hand-built toy engine.

**Piecewise-linear clipping:** labels are capped, `RUL ← min(RUL, R_max)`,
because early in life the sensors show no measurable degradation, so asking
the model to distinguish "180 left" from "150 left" mostly teaches noise.

## 5. Sliding-window preprocessing strategy

Pipeline order matters (leakage lives here):

1. Compute RUL labels (§4) and apply the cap (§4, §5.1 below).
2. **Split by engine ID** — train engines → train/val (e.g. 80/20, fixed
seed); the official 100 test engines are a separate file and stay untouched.
3. Select features from **train-split variance only**.
4. Fit the scaler on **train-split rows only**; persist it.
5. Transform train/val/test with that one scaler.
6. Build windows **inside each engine**: length **30**, stride 1,
`X = rows[t−29 … t]`, `y = RUL at t`. Engines shorter than 30 cycles:
left-pad by repeating the first row (documented rule; needed because some
test engines are short, and the official evaluation needs a last window
for every engine).

Resulting tensor: `(batch, 30, F)` where `F` is the selected feature count
(expected ≈ 14; confirmed in A1). Windows never cross engine boundaries —
enforced by construction *and* by an explicit test.

### 5.1 Is the 125-cycle cap right? — Researched, not assumed

125 is the **dominant literature convention** since the piecewise-linear
target was popularized (Heimes 2008; Li et al. 2018): studies applying it
report FD001 LSTM RMSE values around **14–16 cycles** (e.g. a stacked-LSTM
baseline at 14.46; a piecewise-linear LSTM study
reporting RMSE 14.03, R² 0.88 at the 125 cap).
But "everyone uses 125" is not evidence that 125 is optimal *for our model* —
it interacts with window length, capacity and loss. Some pipelines use 130
instead. **Plan:** start at 125 (comparability with literature), then run the
Phase A4 ablation (125 vs 150 vs no cap, same seed, val RMSE decides) and
record the outcome. That turns an assumption into a measured decision — a
strong viva point.

## 6–7. Recommended LSTM architecture + training methodology

**Architecture (starting point, config-driven so it can change by YAML, not
by code edits):**

```
Input (B, 30, F)
→ LSTM(F → 128, batch_first) # layer 1
→ Dropout 0.2
→ LSTM(128 → 64) # layer 2 — last hidden state
→ Dropout 0.2
→ Linear(64 → 32) → ReLU
→ Linear(32 → 1) # RUL (cycles), clip at ≥ 0 at inference
```

Reasoning: two stacked LSTMs at this size are the literature's workhorse for
FD001 (Deep LSTM, Zheng et al. 2017: RMSE 16.14; later tuned variants reach
~12.5–14) — large enough to model degradation trends, small enough to train
on CPU in minutes and to keep ONNX/Rust inference trivial. Bigger ideas
(attention, BiLSTM, CNN-LSTM, Transformers) are *stretch experiments under
the same pipeline*, not the starting point — starting simple is what makes
the baseline believable.

**Training methodology:**
- Loss: **MSE** (directly optimizes the RMSE we report). The optional
asymmetric loss (penalize over-prediction more, mirroring the NASA score)
is a flagged experiment in config — off by default.
- Optimizer Adam (lr 1e-3), batch 64, gradient clipping (norm 1.0),
LR schedule (plateau/cosine), **early stopping on val RMSE** (patience 10),
best-checkpoint restore.
- **Seeds fixed** (python/numpy/torch) and the full config snapshotted per
run — a result without its config + seed is not a result.
- Sanity gates before any real run: forward-shape test; overfit-one-batch
test (loss must collapse); checkpoint-reload identity test.

## 8. Evaluation metrics

- **RMSE** (primary) and **MAE**, in cycles.
- **NASA PHM08 score** — asymmetric: with `d = predicted − true`,
`score = Σ (exp(−d/13) − 1)` for `d < 0` and `Σ (exp(d/10) − 1)` for
`d ≥ 0`. Late (over-) predictions are punished harder — operationally
correct, since over-estimating remaining life is the dangerous error.
- **Protocol:** headline test numbers come from the **last window of each
test engine** (100 predictions) — the official comparison protocol used by
the literature. Val metrics over all windows
are reported separately and never mixed with test numbers.
- **Error analysis** (mandatory, Phase A4): error by true-RUL band, worst
engines, prediction-vs-actual scatter. Plus two baselines (mean predictor,
Ridge on flattened windows) — the LSTM must beat them to justify itself.
- Literature context for sanity (not targets): published FD001 LSTM RMSE
spans roughly **12.5–16.5** depending on pipeline.
If PREDICAI lands far outside this band either way, that is a bug signal
(too good ⇒ suspect leakage; too bad ⇒ pipeline error) and gets investigated
before anything is reported.

## 9. Explainability strategy — **with a correction to the brief**

The brief asks for SHAP. SHAP is usable, but on **raw LSTM sequences it is
methodologically weak**: KernelSHAP/DeepSHAP treat timesteps-features in ways
that break temporal dependence, need a background sample, and produce
unstable rankings on C-MAPSS — recent faithfulness studies on exactly this
dataset find gradient/occlusion attributions substantially more reliable
than attention-style or naive SHAP explanations. So the plan uses:

1. **Permutation importance** (primary, model-agnostic): shuffle one sensor
across the val set, measure RMSE increase. Simple, honest, explainable.
2. **Integrated Gradients** via Captum (primary): per-sensor attribution on
representative windows (low/mid/high true RUL).
3. **SHAP** (secondary, labelled *experimental* in the dashboard): applied to
a flattened-window view with a small background set, limitations stated
next to the chart.

Language rule: attributions show **association** ("when sensor X moves, the
prediction moves"), never causation ("sensor X causes failure"). Agreement
or disagreement between methods is reported as-is (Phase A5 gate).

## 10. ONNX export feasibility — **High**

- PyTorch → ONNX for an LSTM is a core, long-supported path:
`torch.onnx.export(..., opset_version=17, dynamic_axes={batch})`,
sequence length kept fixed at 30 to keep the serving contract simple.
- Mandatory verification chain: `onnx.checker` → Python ONNX Runtime load →
**numerical parity**: same ≥100 windows through PyTorch and ONNX,
max |diff| ≤ 1e-4. This is a Phase A6 *gate*, not a nicety.
- Artifacts frozen together: `cmapss_lstm.onnx`, `preprocessing.json`
(feature order, scaler mean/scale, seq_len, cap), `model_metadata.json`
(architecture, metrics, git commit). The model file without its
preprocessing artifact is considered an incomplete deliverable.

## 11. Rust ONNX Runtime feasibility — **High, one real risk**

- The `ort` crate is the standard Rust wrapper over ONNX Runtime and the
default recommendation for "trained in Python, served from Rust" — it
supports the LSTM operator set and all major execution providers.
- **The risk, plainly:** `ort` 2.0 has been in **release candidates since
2024 with no stable crates.io release** (latest seen: 2.0.0-rc.15,
Oct 2026); the guidance from the Rust ML community is to **pin an exact
version** (`ort = "=2.0.0-rc.…"`).
PREDICAI does exactly that, isolates all `ort` calls inside `model.rs`
(so an API change touches one file), and keeps **`tract`** (pure-Rust
ONNX runtime) as the documented fallback if a blocker appears.
- Windows note: `ort` downloads/links the ONNX Runtime binary at build time
(`download-binaries` feature) — the Phase D setup must verify this on the
owner's Windows machine early, not at demo time.
- Performance honesty: Rust is **not** assumed faster. The LSTM is tiny;
Python+ONNX Runtime may match it. Phase D *measures* p50/p95 latency over
1,000 local requests and reports whatever comes out. The honest Rust
advantages to cite are deployment ones: single binary, no Python runtime,
low/stable memory, type-safe validation — speed only if measured.

## 12. Python ↔ Rust preprocessing consistency

The #1 real-world failure mode for this exact architecture. Controls:

1. **One artifact:** both sides read the same `preprocessing.json`
(feature names in order, mean, scale, seq_len, cap, model version).
Rust hardcodes *nothing* about features or statistics.
2. **One formula:** normalization is only `z = (x − mean) / scale` per
feature — trivially identical in numpy and Rust `f32/f64` with a defined
tolerance (use `f64` in Rust preprocessing, cast once at tensor build).
3. **Two parity tests:** Python ONNX vs PyTorch (A6, ≤1e-4) and Rust
end-to-end vs Python ONNX on identical raw windows (D, ≤1e-3).
4. **Validation symmetry:** both sides reject NaN, wrong feature count and
wrong sequence length with equivalent errors.

## 13. Maintenance-history integration

Design (Phase B): SQLite with `machines`, `service_records`,
`recommendations`; a Python reference module first (fast to test), the same
schema consumed by the Rust engine — one schema file, two consumers.

Honesty constraints (from the brief, endorsed and strengthened):
- FD001 contains **no** maintenance records. All demo service data is
synthetic, carries `is_synthetic = 1` in the DB, and is labelled in the UI.
- Service history is **never** an LSTM feature (no training data exists to
justify it) and a service event **does not** reset the model's RUL —
the model knows sensor degradation only. The recommendation engine may
*say* "post-service, re-baseline monitoring", but it must not fake a
model effect.
- Recommendations = RUL band + cycles-since-service + last service type,
combined by a **config-file policy**; every output carries a human-readable
`reason`; thresholds are labelled *policy choices* in code, docs and UI —
never "failure boundaries".

## 14. Sensor simulation strategy

- **Primary: replay.** Stream the FD001 *test* files cycle-by-cycle for a
chosen engine: append the real sensor row, keep the rolling 30-window,
call `POST /predict_rul`. This is simulated *delivery* of **real benchmark
data**, and it has a bonus: the engine's true final RUL is known, so the
demo can honestly show prediction vs ground truth converging.
- **Secondary (optional): synthetic perturbation** for demo variety,
labelled `SYNTHETIC`.
- Prediction history is stored per call (timestamp, engine, cycle, RUL,
model version) — the dashboard's trend charts read real stored rows,
never front-end inventions.
- Hardware (Pi/IoT) is a documented extension point (same JSON contract),
explicitly out of the MVP.

## 15. Dashboard architecture

- React + TypeScript + Vite + Tailwind; charts with **Recharts** (simple,
React-native, sufficient for line/area/bar at this scale).
- Data flow: dashboard → Rust API only (predictions, history,
recommendations, model metadata). No direct DB access from the browser.
- Sections per the brief §11, with two additions of discipline:
every panel carries a **source label** (`FD001 REPLAY` / `SYNTHETIC DEMO`),
and the Model Info page shows **only Phase A4's measured metrics** +
metadata (training config, cap, feature list).
- Designed states: API-down banner, empty-history state, loading skeletons.
Visual identity: dark industrial slate, amber alert + cyan info accents,
dense-but-clean tables — premium through typography and spacing, not
through fake statistics or decorative gradients.

## 16. Main technical risks (consolidated)

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| 1 | Train/serve preprocessing skew | High | §12 controls + parity gates |
| 2 | Leakage (scaler/windows/split) inflating metrics | High | Engine-wise split, train-only fit, dedicated tests |
| 3 | `ort` RC API changes / Windows link issues | Medium | Exact pin, `model.rs` isolation, `tract` fallback, early Windows build check |
| 4 | Cap=125 treated as truth | Medium | A4 ablation decides on val data |
| 5 | SHAP over-interpretation | Medium | §9 method stack + association-only language |
| 6 | Test engines shorter than the 30-window | Medium | Documented padding rule + test |
| 7 | Synthetic maintenance data mistaken for real | Medium | DB flag + UI labels + docs |
| 8 | Scope creep (FD002–4, Pi, quantization) | Medium | Stretch list gated behind MVP |
| 9 | Dataset licence / redistribution | Low | Download script; raw data gitignored; cite NASA |
| 10 | "Rust is faster" claimed without evidence | Low | Benchmarks measured in Phase D or not claimed |

## 17. Hardware requirements

Deliberately modest — a strength of the project:
- **Development:** any Windows 11 laptop with ≥8 GB RAM (16 GB comfortable).
FD001 is ~20k rows; LSTM training is **CPU-feasible in minutes**; a GPU
is optional and must not become a requirement.
- **Rust build:** stable toolchain; ~2–3 GB disk for `target/` + ONNX Runtime.
- **Dashboard:** Node 20 LTS.
- **Edge story (documented, not required):** the same Rust binary + ONNX
model is sized for a Raspberry Pi-class device; deployment on Pi is a
stretch item, verified only if hardware is actually available.

## 18. Estimated development phases

Sequenced exactly as `plan.md` §3 (durations are *session estimates* for a
student schedule, not promises): A1 scaffold+EDA (1 session) → A2
preprocessing (1) → A3 training (1–2) → A4 evaluation+ablation (1) →
A5 explainability (1) → A6 ONNX (½) → B maintenance (1, parallel-able) →
D Rust engine (2) → E simulator (½) → F dashboard (2) → G integration+
presentation pack (1). Roughly **11–13 focused sessions** to MVP.

## 19. Realistic MVP scope

MVP = Phase A1–A6 + B + D + E + F as gated in `plan.md` §4: a trained,
evaluated, explainable LSTM; a parity-proven ONNX model; a Rust API serving
it; a replay simulator; a labelled synthetic maintenance demo; and a
dashboard showing the real flow end-to-end — with tests at every layer.
Explicitly **out** of MVP: FD002–FD004, classification heads, uncertainty
intervals, quantization, physical hardware, auth/multi-user, cloud anything.

## 20. Recommended improvements over the original brief

1. **Explainability method stack** (§9) instead of SHAP-only.
2. **Baselines + clipping ablation** (A4) — turns the project from "an LSTM
was trained" into "the LSTM was *justified*", the difference that matters
in evaluation.
3. **Single preprocessing artifact + two parity gates** (§12) — the
professional MLOps habit this project can genuinely demonstrate.
4. **`ort` version pinning + `tract` fallback** (§11) — names the real risk
in the Rust plan instead of discovering it mid-build.
5. **Config-driven everything** (YAML for ML, TOML/JSON for policy) —
reproducibility and clean viva answers.
6. **Replay-first simulator** (§14) — real benchmark data with known ground
truth beats invented sensor streams for an honest demo.
7. **Model-info page fed from `model_metadata.json`** — the dashboard can
never drift from the actually-deployed model.
8. **What I deliberately did *not* add:** no Kafka/Kubernetes/cloud, no
Transformer-first architecture, no FD004 "for completeness", no fake
real-time claims. Restraint is part of the engineering quality here.

---

### Corrections to assumptions in the brief (summary)

| Assumption | Finding |
|---|---|
| Cap 125 is the value to use | It is the literature *convention*; optimality is unproven → ablation in A4 |
| SHAP will explain the LSTM | Weak on raw sequences → permutation + IG primary, SHAP secondary (§9) |
| Rust will be faster | Unproven for a tiny LSTM → measure in D; cite deployment benefits, not speed, unless measured |
| 14 sensors is the feature set | Likely, but must be *measured* in A1 EDA, not copied |
| Maintenance can inform the model | No training data justifies it → kept as a separate, labelled policy layer |

### Primary references
- Saxena et al., PHM 2008 (dataset + NASA score) — NASA Prognostics Center.
- Zheng et al., IEEE PHM 2017 (Deep LSTM baseline, FD001 RMSE 16.14).
- Li et al., 2018 (piecewise-linear RUL + DCNN line of work; cap convention).
- PyTorch docs: `torch.onnx.export` (opset 17, dynamic axes).
- ONNX Runtime + `ort` crate docs (pin exact RC; see §11).
- Published FD001 context values cited inline via search results above.

**Next step:** owner approval of this report + `plan.md` → Phase A1 begins.
No code is written before that approval.
