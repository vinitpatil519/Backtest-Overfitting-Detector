# Backtest Overfitting Detector

Detect whether an impressive Sharpe ratio is real or statistical luck.

Generate a thousand trading strategies, backtest them all on the same price
history, and the best one will look excellent — even when the price history is a
driftless random walk and nothing in the grid has any edge at all. This tool
measures exactly how much of that performance the search itself manufactured,
using the Deflated Sharpe Ratio, the Probability of Backtest Overfitting, and
Monte Carlo validation.

Everything is computed locally in NumPy and SciPy. No data leaves the machine.

<img width="1902" height="967" alt="image" src="https://github.com/user-attachments/assets/c3aabc18-9d35-48ab-9d18-a606ab8d65f3" />


<img width="1917" height="971" alt="image" src="https://github.com/user-attachments/assets/5654b130-2962-4550-aa85-010c185e3a19" />


<img width="1917" height="973" alt="image" src="https://github.com/user-attachments/assets/5125003c-a206-4690-b34f-96f615c52f94" />


---

## What it computes

| Metric | Question it answers |
| --- | --- |
| **Sharpe ratio** | How good does this look? |
| **Probabilistic Sharpe Ratio (PSR)** | What is the probability the true Sharpe beats a benchmark, given sample length, skew and kurtosis? |
| **Deflated Sharpe Ratio (DSR)** | Same question, but against the Sharpe you would expect from the *best* of N trials with no edge. |
| **Probability of Backtest Overfitting (PBO)** | How often does the in-sample winner land in the bottom half out-of-sample? |
| **Performance degradation** | Does in-sample Sharpe predict out-of-sample Sharpe at all? |
| **Monte Carlo p-value** | How often does a zero-edge grid of the same size produce a best Sharpe this good? |
| **Block bootstrap CI** | How wide are the error bars on the selected strategy's Sharpe? |
| **Minimum Track Record Length** | How much history would you need before this edge is significant? |
| **Skew / kurtosis** | How much do fat tails and asymmetry inflate the naive Sharpe? |
| **Robustness score** | All of the above, weighted into one 0–100 number and a grade. |

### Workflow

```mermaid
flowchart LR
A[Generate 1000 Strategies] --> B[Backtest]
B --> C[Sharpe Distribution]
C --> D[Deflated Sharpe]
D --> E[Robustness Score]
```

---

## Quick start

Requires Node 18.18+ and Python 3.9+.

```bash
npm install
pip install -r requirements-dev.txt
npm run dev
```

`npm run dev` starts both processes: Next.js on <http://localhost:3000> and the
Python API on <http://localhost:5328>. Open the first one.

To run them separately:

```bash
npm run dev:next     # UI only
npm run dev:api      # uvicorn api.index:app --reload --port 5328
```

### Command line

The engine works without the web UI:

```bash
# Default synthetic lab, writes reports/report.json and reports/report.png
python -m engine.cli --out reports

# Positive control: serial correlation makes momentum genuinely predictive
python -m engine.cli --autocorr 0.25 --n-bars 2000

# Your own returns, one column per strategy
python -m engine.cli --csv my_returns.csv --declared-trials 250

# Use the scikit-learn clustering estimate of the effective trial count
python -m engine.cli --cluster-trials
```

### Tests

```bash
python -m pytest
```

71 tests cover closed-form identities (PSR reducing to Lo's variance factor,
MinTRL inverting PSR exactly, the analytic expected maximum matching simulation),
CSCV behaviour under known regimes, absence of look-ahead bias in the
backtester, and end-to-end reproducibility.

---

## The mathematics

All Sharpe ratios inside the engine are **per observation**; annualisation
happens only at the presentation layer. Mixing the two conventions is the most
common bug in DSR implementations, so the conversions are explicit
(`annualize_sharpe` / `deannualize_sharpe`).

### Probabilistic Sharpe Ratio

```
                    (SR − SR*) · √(T − 1)
PSR(SR*) = Φ( ────────────────────────────────── )
              √(1 − γ₃·SR + (γ₄ − 1)/4 · SR²)
```

γ₃ is skewness and γ₄ is **non-excess** kurtosis (3 for a Gaussian). The
denominator is the variance factor of the Sharpe estimator; for Gaussian returns
it collapses to `1 + SR²/2`, the classic Lo (2002) result.

### Deflated Sharpe Ratio

Run N strategies on data with no edge and the best still looks good. The
expected maximum of N Gaussian Sharpe estimates is the bar a real result must
clear:

```
SR*₀ = √V[SR_n] · [ (1 − γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e)) ]

DSR  = PSR(SR*₀)
```

γ ≈ 0.5772 is the Euler–Mascheroni constant and `V[SR_n]` is the variance of the
Sharpe ratios across trials.

**Effective trials.** A grid of correlated variants is not N independent
experiments. Two estimators are available:

- `engine.stats.effective_number_of_trials` (default, NumPy only) —
  `N_eff = 1 + (N − 1)(1 − ρ̄)` from the mean absolute pairwise correlation.
- `engine.cluster.effective_trials_by_clustering` (opt-in, scikit-learn) —
  hierarchical clustering on the correlation distance `√((1−ρ)/2)`, cutting the
  dendrogram where ρ > 0.9 so only near-duplicates merge.

The cluster count is deliberately *not* chosen by silhouette score. Silhouette
prefers the coarsest split on financial correlation matrices; it reported two
effective trials for a 500-strategy sweep, which shrank the benchmark enough to
let a pure-noise grid score a B. A fixed, conservative correlation cutoff is both
more defensible and more stable.

### Probability of Backtest Overfitting

CSCV splits the `T × N` performance matrix into S equal blocks in time and
evaluates every way of using half of them in-sample. For each split it selects
the in-sample winner and records its rank ω among the N out-of-sample Sharpes:

```
ω   = rank / (N + 1)
λ   = ln( ω / (1 − ω) )
PBO = P[ λ ≤ 0 ]
```

Above 0.5, selecting on the backtest is actively worse than choosing at random.

Sharpe ratios for every union of blocks are reconstructed from per-block sums and
sums of squares, so all C(16, 8) = 12 870 combinations cost one matrix multiply
each rather than a re-scan of the returns — the whole sweep runs in about two
seconds for 1000 strategies.

### Minimum Track Record Length

```
MinTRL = 1 + [1 − γ₃·SR + (γ₄ − 1)/4 · SR²] · ( Φ⁻¹(p) / (SR − SR*) )²
```

Infinite when the Sharpe sits below the benchmark: no amount of extra data makes
a non-edge significant.

### Robustness score

| Weight | Component |
| --- | --- |
| 0.30 | Deflated Sharpe |
| 0.25 | 1 − PBO |
| 0.15 | 1 − P(out-of-sample Sharpe ≤ 0) |
| 0.15 | 1 − Monte Carlo p-value |
| 0.15 | Estimation stability (bootstrap, tail penalty, MinTRL coverage) |

Components that cannot be computed — PBO needs at least two candidate strategies
— are dropped and the remaining weights renormalised, so the score always stays
on the same scale. Grades: A ≥ 80, B ≥ 65, C ≥ 50, D ≥ 35, otherwise F.

---

## Calibration

The tool is only useful if it separates luck from edge in both directions. Two
controls, both running the full 993-strategy grid at the default seed
(`python -m engine.cli` and `--autocorr 0.25 --n-bars 2000`):

| Market | Best Sharpe | Naive PSR | SR\* | DSR | PBO | MC p | Score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Driftless random walk | 1.24 | 99.4% | 1.63 | **21.7%** | 0.54 | 0.99 | **41 · D** |
| AR(1) ρ = 0.25 (real edge) | 3.61 | 100.0% | 3.41 | **70.4%** | 0.00 | 0.27 | **86 · A** |

The first row is the whole point: a Sharpe of 1.24 with a 99.4% Probabilistic
Sharpe Ratio, from data that contains nothing. The second shows the detector does
not simply condemn everything — a genuine edge clears the deflated bar and PBO
collapses to zero.

Note that raising `--drift-annual` is a *poor* positive control even though it
creates a genuine edge — it splits the grid into long-biased winners and
short-biased losers, which widens `V[SR_n]` and therefore raises SR\* faster than
it raises the best Sharpe. Serial correlation creates predictability without
handing out free beta, which is why the presets use it.

---

## Using your own returns

Switch the UI to **My returns**, or pass `--csv` on the command line. One column
per strategy, one row per period; a header row is optional, and comma, semicolon,
tab or whitespace all work. Values may be decimals (`0.012`) or percentages
(`1.2`) — the convention is inferred from magnitude and reported back.

- **Two or more columns** unlock the full analysis including PBO.
- **A single column** still gives PSR, DSR, Monte Carlo and the bootstrap, but
  PBO is skipped: CSCV needs the candidates you rejected, not just the one you
  kept. Set "trials you ran" so the deflation knows how hard you searched — that
  number is the difference between a 99% PSR and a 20% DSR.

---

## Architecture

```
app/                Next.js App Router UI (client-side, no server state)
components/         Hand-rolled SVG charts and form controls
lib/                Typed API client, response types, formatting
api/index.py        FastAPI ASGI app → Vercel serverless function
engine/
  stats.py          Sharpe, PSR, DSR, MinTRL, moments, drawdowns
  backtest.py       Market simulation + vectorised multi-strategy backtester
  cscv.py           Combinatorially Symmetric Cross-Validation / PBO
  montecarlo.py     Null max-Sharpe distribution, circular block bootstrap
  score.py          Robustness score, grades, flags
  pipeline.py       Orchestration → JSON-safe payload
  frames.py         pandas views            (optional)
  cluster.py        scikit-learn N_eff      (optional)
  report.py         matplotlib report       (optional)
  cli.py            Command line entry point
tests/              71 pytest tests
```

**Why the split.** `engine.stats`, `backtest`, `cscv`, `montecarlo`, `score` and
`pipeline` depend only on NumPy and SciPy, which keeps the deployed function well
inside Vercel's bundle size limit. pandas, scikit-learn and matplotlib are used
by modules that the serverless path never imports, so they stay in
`requirements-dev.txt`.

The charts are plain inline SVG rather than a charting library: no runtime
dependency, a 101 kB first-load bundle, and identical visual language across
every panel.

---

## Deploying to Vercel

The repository is ready to deploy as-is.

1. Push to GitHub.
2. Import the repository at [vercel.com/new](https://vercel.com/new). The
   Next.js preset is detected automatically; no environment variables are needed.
3. Deploy.

`vercel.json` rewrites `/api/py/*` to the Python function, gives it 1769 MB and a
60-second ceiling, and ships the `engine/` package alongside the entry point via
`includeFiles`. `next.config.mjs` performs the same rewrite in development,
pointing at the local uvicorn process instead.

Defaults are sized to finish in well under a second of compute; the request
schema in `api/index.py` caps every parameter so a crafted request cannot exhaust
the function's memory or wall clock. Cold starts take a few seconds while SciPy
loads.

---

## References

- Bailey, D. and López de Prado, M. (2012). *The Sharpe Ratio Efficient
  Frontier*. Journal of Risk 15(2).
- Bailey, D. and López de Prado, M. (2014). *The Deflated Sharpe Ratio:
  Correcting for Selection Bias, Backtest Overfitting and Non-Normality*.
  Journal of Portfolio Management 40(5).
- Bailey, D., Borwein, J., López de Prado, M. and Zhu, Q. (2017). *The
  Probability of Backtest Overfitting*. Journal of Computational Finance 20(4).
- Lo, A. (2002). *The Statistics of Sharpe Ratios*. Financial Analysts Journal
  58(4).

---

## Outcome

Only statistically significant strategies pass.

Research and education only. Not investment advice.
 
 
