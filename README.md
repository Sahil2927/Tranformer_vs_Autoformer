# Transformer vs Autoformer on the Weather LTSF Benchmark

A controlled comparison of the **vanilla Transformer** and **Autoformer** — both as published in
[thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library), neither modified — on the
Weather long-term forecasting benchmark.

The point of this repo is not the leaderboard number. It is the *control*: the upstream benchmark
scripts contain asymmetries that quietly favour one model, and this study documents and fixes them
before comparing anything.

---

## Result

Autoformer wins **6 out of 6** matched runs. Test MSE in standardised units, mean over two seeds:

| Horizon | Transformer | Autoformer | MSE reduction |
|--------:|------------:|-----------:|--------------:|
| 96      | 0.3902      | **0.2615** | 33.0%         |
| 192     | 0.4904      | **0.3167** | 35.4%         |
| 336     | 0.6959      | **0.3658** | 47.4%         |

MAE improves by less (20.3% / 23.7% / 31.6%). Because MSE penalises large errors quadratically,
that divergence indicates the Transformer's error distribution is **heavy-tailed** — occasional
large misses rather than uniformly worse predictions.

Wilcoxon signed-rank over the 6 pairs: `W = 0, p = 0.0312`. That is the *smallest attainable*
p-value at n = 6 — the test is saturated, so the effect sizes carry the evidence, not the p-value.

Autoformer costs roughly **1.8×** more per training iteration.

---

## Three asymmetries in the upstream benchmark

These are the reason this repo exists. All were found by reading the library source, and all are
verifiable against commit `4e938a17`.

### 1. The shipped Weather scripts use unequal epoch budgets

`scripts/long_term_forecast/Weather_script/Transformer.sh` passes `--train_epochs 3` at horizon 96.
`Autoformer.sh` passes `--train_epochs 2`. Both fall back to the default of 10 elsewhere.

Copying the shipped scripts hands the two models different training budgets at one horizon. This
study sets a ceiling of 10 for both, at every horizon, and **verifies the ceiling never bound** —
every run stopped on its own validation curve via early stopping.

### 2. `run.py` hardcodes the seed and ignores its own `--seed` flag

```python
fix_seed = 2021          # line 10 — args.seed is parsed but never read
random.seed(fix_seed)
torch.manual_seed(fix_seed)
```

Multi-seed experiments are impossible from the CLI until this is redirected.

### 3. The same seed does not give the two models the same batch order

This is the subtle one. `Exp_Basic.__init__` constructs the model **before** `_get_data` builds the
DataLoader. Weight initialisation draws from the global RNG in proportion to parameter count, so by
the time the loader is created the two architectures have advanced the global stream by different
amounts — identical seed, different shuffle order.

Fixed by attaching an explicit generator to the forecasting DataLoader:

```python
_g = torch.Generator()
_g.manual_seed(int(os.environ.get('TSLIB_SEED', '2021')))
data_loader = DataLoader(..., generator=_g)
```

> **Note when applying this patch yourself:** the bare `DataLoader(...)` block appears **twice** in
> `data_provider/data_factory.py` — the anomaly-detection branch has a byte-identical one. Anchor the
> replacement on the preceding `seasonal_patterns=args.seasonal_patterns` lines or you will patch the
> wrong branch.

Two further patches (removing the unused per-epoch test evaluation, disabling plot rendering during
testing) cut ~9% of runtime and cannot affect results — validation runs in eval mode, where dropout
draws no random numbers, and each loader holds its own generator.

**Total footprint: 9 lines across 3 files.** `models/` and `layers/` are untouched, verified with
`git diff`.

---

## Quickstart

Designed to run end-to-end in Google Colab on a single T4. Roughly 4–5 hours for the full grid.

```bash
git clone https://github.com/thuml/Time-Series-Library.git
cd Time-Series-Library
git checkout 4e938a1767106324dd753b2a44832bf870a0252e   # pinned; main moves

# module-level imports on the paths this experiment touches.
# None are called here — they only have to resolve.
pip install einops==0.8.1 reformer-pytorch==1.4.4 sktime datasets patool nltk

mkdir -p dataset/weather && cp /path/to/weather.csv dataset/weather/
```

Then open `transformer_vs_autoformer_weather.ipynb`, set `DRIVE_CSV`, and run top to bottom.
The notebook applies the four patches (idempotently, with assertions that each anchor matched
exactly once), audits parameter counts, micro-benchmarks both models to estimate runtime *before*
committing to the grid, then runs 12 experiments.

Results are checkpointed to disk after every run and completed runs are skipped, so a Colab
disconnect costs one run rather than all twelve.

---

## Experimental setup

**Data.** Weather (Max Planck Institute for Biogeochemistry, Jena): 52,696 timesteps × 21
meteorological variables at 10-minute resolution. Chronological 7:1:2 split, standardisation fitted
on the training split only, multivariate setting.

**Shared configuration**, identical for both models at every horizon:

```
seq_len 96 · label_len 48 · e_layers 2 · d_layers 1
d_model 512 · d_ff 2048 · n_heads 8 · dropout 0.1
batch_size 32 · lr 1e-4 · lradj type1 · patience 3 · train_epochs 10
features M · freq h · enc_in/dec_in/c_out 21
```

**Not shared:** `factor 3` and `moving_avg 25` are read only by Autoformer and are inert for the
vanilla Transformer's full attention. They are reported as model-specific rather than presented as
shared hyperparameters.

**Grid:** 2 models × 3 horizons {96, 192, 336} × 2 seeds {2021, 2022} = 12 runs.

**On `freq='h'`:** Weather is 10-minute data, but `h` is the repo default and what published Weather
numbers were produced with. It gives the time-feature embedding 4 channels instead of 5. Kept for
comparability, and stated rather than silently corrected.

---

## Findings beyond the headline

### The advantage is concentrated, not broad

Contribution to the total absolute error gap by physical variable family, horizon 96:

| Family        | Channels | Mean advantage | Share of gap |
|---------------|---------:|---------------:|-------------:|
| Temperature   | 4        | +0.4121        | 61.0%        |
| Humidity      | 7        | +0.1822        | 47.2%        |
| Radiation     | 3        | +0.0498        | 5.5%         |
| Pressure      | 1        | −0.0662        | −2.4%        |
| Wind          | 3        | −0.0285        | −3.2%        |
| Precipitation | 2        | −0.0452        | −3.3%        |
| Target (OT)   | 1        | −0.1295        | −4.8%        |

Temperature and humidity — 11 of 21 channels — account for more than the entire gap. A single
channel, `Tlog (degC)`, contributes **28.6%**; on it the Transformer scores 0.93, roughly what
predicting the training mean would give.

### Two explanations tested and rejected

**Periodicity.** Weather's diurnal cycle is 144 timesteps. Spearman ρ between diurnal spectral power
and per-channel advantage: **ρ = +0.05, p = 0.84**. The two most diurnal channels (PAR, SWDR) show
among the *smallest* advantages. Consistent with the mechanism — at `seq_len=96` the input window is
shorter than the period, so Auto-Correlation cannot locate the daily cycle at all.

**Smoothness.** Lag-1 autocorrelation gives ρ = +0.60, p = 0.004 across 21 channels — but those are
not independent observations. Collapsing to 7 physical families removes the effect entirely
(ρ ≈ +0.14, p ≈ 0.76). Direct counterexample: `p (mbar)` has the highest autocorrelation in the
dataset (0.9999) and Autoformer is 27% *worse* on it.

### A candidate mechanism

Both models receive a decoder input that is **zeros across the forecast horizon**. The Transformer
uses it directly. Autoformer discards it for its trend path:

```python
mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
```

So Autoformer's forecast begins at the local mean of the lookback window, with accumulated trend
added on top, while the Transformer must reconstruct the signal's level through attention alone.

This predicts exactly the observed pattern: large gains on smooth, level-dominated channels where
the local mean is already most of the answer, and none on zero-mean intermittent channels where it
carries no information.

**Consistent with the evidence, not demonstrated.** Separating it from Auto-Correlation requires a
Transformer-plus-decomposition variant, which a two-model design does not contain.

### A data-quality note

Three channels contain `-9999` sentinel values for missing readings: `OT` (50), `max. PAR` (30),
`wv` (1). These inflate the standard deviation, so standardisation compresses the real signal and
the per-channel MSE for those columns is unreliable — they are the source of the extreme percentage
deltas (−444%, −373%).

**Left uncorrected**, because this is a property of the benchmark file used throughout the
literature and fixing it would break comparability. Excluding those three raises the mean per-channel
advantage from 0.129 to 0.155, so their presence makes the reported result slightly conservative.

---

## Repository contents

```
transformer_vs_autoformer_weather.ipynb   end-to-end experiment notebook
report.md                                 full write-up
presentation.pptx                         slide deck
speaker_notes.md                          plain-language talk notes
results/
├── results.json          per-run metrics + full validation curves (12 runs)
├── raw_runs.csv          flat table of every run
├── param_counts.csv      measured trainable parameters per model per horizon
├── cost_benchmark.csv    timing and peak memory, controlled micro-benchmark
├── per_channel_full.csv  per-channel errors and signal properties
├── patches.diff          the complete 9-line source diff
├── run_config.json       full configuration and pinned commit
└── summary.png           error, cost, and convergence figures
```

---

## Limitations

- **Two seeds.** Seed spread is a range, not a variance estimate, and the paired test is saturated
  at its minimum attainable p.
- **No component attribution.** A two-model design cannot isolate decomposition from
  Auto-Correlation.
- **No linear baseline.** DLinear and NLinear are known to be competitive on Weather. This study
  establishes that Autoformer beats the vanilla Transformer, not that either is strong for this
  data. Adding them is the highest-value next step.
- **One dataset, one lookback.** All results are specific to Weather at `seq_len=96`.
- **Horizon 720 excluded** for compute reasons. Conservative with respect to the conclusion: the
  margin is largest at 336, so the omitted horizon is where Autoformer's lead would most plausibly
  have extended.
- **Cost measurement varies the horizon at fixed lookback**, so it measures decoder-length scaling
  and does *not* test the O(L log L) vs O(L²) complexity claim.

---

## Reproducibility

Every number in this repo traces to a pinned commit and a recorded configuration. Each run stores
the epoch it stopped at and its full validation curve, so the claim that no run was truncated by the
epoch ceiling is auditable rather than asserted.

```
Upstream:  thuml/Time-Series-Library @ 4e938a1767106324dd753b2a44832bf870a0252e
Hardware:  NVIDIA T4 (Google Colab)
Changes:   4 patches, 9 lines, 3 files — see results/patches.diff
           models/ and layers/ unmodified
```

---

## Acknowledgements and citation

This work uses [thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library)
(MIT License, © 2021 THUML @ Tsinghua University) as its experimental harness, with the two model
implementations used unmodified.

```bibtex
@inproceedings{wu2021autoformer,
  title     = {Autoformer: Decomposition Transformers with Auto-Correlation
               for Long-Term Series Forecasting},
  author    = {Wu, Haixu and Xu, Jiehui and Wang, Jianmin and Long, Mingsheng},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2021}
}

@inproceedings{vaswani2017attention,
  title     = {Attention Is All You Need},
  author    = {Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob
               and Jones, Llion and Gomez, Aidan N and Kaiser, Lukasz and Polosukhin, Illia},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2017}
}
```

The Weather dataset is recorded by the Max Planck Institute for Biogeochemistry, Jena, and is
distributed with the Time-Series-Library benchmark suite.

---

## Author

**Nisha Kumari** — Birla Institute of Technology, Mesra
