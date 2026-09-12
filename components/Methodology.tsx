"use client";

import { useState } from "react";
import { Formula, Panel } from "@/components/ui";

const SECTIONS: { title: string; body: React.ReactNode }[] = [
  {
    title: "Probabilistic Sharpe Ratio",
    body: (
      <>
        <p>
          The Sharpe ratio is an estimate, and its error bars widen with negative skew and fat
          tails. The PSR gives the probability that the true Sharpe exceeds a benchmark, given
          the sample length and the higher moments.
        </p>
        <Formula>
          {`PSR(SR*) = Φ( (SR − SR*) · √(T − 1)
               / √(1 − γ₃·SR + (γ₄ − 1)/4 · SR²) )`}
        </Formula>
        <p>
          γ₃ is skewness, γ₄ non-excess kurtosis (3 for a Gaussian), T the number of
          observations. Sharpe here is per observation, not annualised — mixing the two is the
          most common error in implementations of this formula.
        </p>
      </>
    ),
  },
  {
    title: "Deflated Sharpe Ratio",
    body: (
      <>
        <p>
          Run N strategies on data with no edge and the best of them still looks good. The
          expected maximum of N Gaussian Sharpe estimates is the benchmark any real result has
          to clear:
        </p>
        <Formula>
          {`SR*₀ = √V[SR_n] · [ (1 − γ)·Φ⁻¹(1 − 1/N)
                    +      γ ·Φ⁻¹(1 − 1/(N·e)) ]

DSR  = PSR(SR*₀)`}
        </Formula>
        <p>
          γ ≈ 0.5772 is the Euler–Mascheroni constant and V[SR_n] is the variance of the Sharpe
          ratios across the trials. A grid of correlated variants is not N independent
          experiments, so N is replaced by an effective count N_eff = 1 + (N − 1)(1 − ρ̄), using
          the mean absolute pairwise correlation ρ̄.
        </p>
      </>
    ),
  },
  {
    title: "Probability of Backtest Overfitting",
    body: (
      <>
        <p>
          CSCV splits the T×N performance matrix into S equal blocks in time and evaluates every
          way of using half of them as in-sample. For each split it selects the in-sample winner
          and records its rank ω among the N out-of-sample Sharpes:
        </p>
        <Formula>
          {`ω  = rank / (N + 1)
λ  = ln( ω / (1 − ω) )
PBO = P[ λ ≤ 0 ]`}
        </Formula>
        <p>
          PBO is the probability that the strategy you would have picked lands in the bottom half
          out-of-sample. Above 0.5, backtest selection is actively harmful. Sharpe ratios for
          every block union are reconstructed from per-block sums and sums of squares, so all
          C(16, 8) = 12 870 combinations cost one matrix multiply each.
        </p>
      </>
    ),
  },
  {
    title: "Monte Carlo and bootstrap",
    body: (
      <>
        <p>
          Two separate questions. The <em>selection</em> test simulates N zero-edge Sharpe
          estimates per path and keeps the maximum, which gives an empirical null for the best
          result of a search — and a direct check on the analytic SR*₀ above.
        </p>
        <p>
          The <em>estimation</em> test resamples the selected strategy with a circular block
          bootstrap, preserving short-horizon autocorrelation that an i.i.d. bootstrap would
          destroy, and reports a confidence interval for its Sharpe.
        </p>
      </>
    ),
  },
  {
    title: "Minimum Track Record Length",
    body: (
      <>
        <p>How much history you would need before the observed edge is significant:</p>
        <Formula>
          {`MinTRL = 1 + [1 − γ₃·SR + (γ₄ − 1)/4 · SR²]
              · ( Φ⁻¹(p) / (SR − SR*) )²`}
        </Formula>
        <p>
          It is infinite when the Sharpe sits below the benchmark: no amount of additional data
          makes a non-edge significant.
        </p>
      </>
    ),
  },
  {
    title: "Robustness score",
    body: (
      <>
        <p>The 0–100 score is a weighted average of five probabilities:</p>
        <Formula>
          {`0.30  Deflated Sharpe
0.25  1 − PBO
0.15  1 − P(OOS Sharpe ≤ 0)
0.15  1 − Monte Carlo p-value
0.15  estimation stability (bootstrap,
      tail penalty, MinTRL coverage)`}
        </Formula>
        <p>
          Components that cannot be computed — PBO needs at least two candidate strategies — are
          dropped and the remaining weights renormalised, so the score always stays on the same
          scale.
        </p>
      </>
    ),
  },
  {
    title: "References",
    body: (
      <ul className="list-inside list-disc space-y-1">
        <li>
          Bailey &amp; López de Prado (2012), <em>The Sharpe Ratio Efficient Frontier</em>,
          Journal of Risk 15(2).
        </li>
        <li>
          Bailey &amp; López de Prado (2014), <em>The Deflated Sharpe Ratio</em>, Journal of
          Portfolio Management 40(5).
        </li>
        <li>
          Bailey, Borwein, López de Prado &amp; Zhu (2017),{" "}
          <em>The Probability of Backtest Overfitting</em>, Journal of Computational Finance
          20(4).
        </li>
        <li>Lo (2002), <em>The Statistics of Sharpe Ratios</em>, Financial Analysts Journal 58(4).</li>
      </ul>
    ),
  },
];

export function Methodology() {
  const [open, setOpen] = useState(false);

  return (
    <Panel
      title="Methodology"
      subtitle="Every number on this page, and where it comes from"
      right={
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          className="underline underline-offset-2 hover:text-ink"
        >
          {open ? "collapse" : "expand"}
        </button>
      }
    >
      {open ? (
        <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
          {SECTIONS.map((section) => (
            <div key={section.title}>
              <h3 className="mb-2 text-[12.5px] font-semibold text-ink">
                {section.title}
              </h3>
              <div className="space-y-2 text-[11.5px] leading-relaxed text-muted [&_em]:text-ink/80">
                {section.body}
              </div>
            </div>
          ))}
        </div>
      ) : (
        <p className="text-[11.5px] leading-relaxed text-muted">
          Deflated Sharpe Ratio, Probability of Backtest Overfitting via CSCV, Monte Carlo null
          for the maximum Sharpe, circular block bootstrap and Minimum Track Record Length — all
          computed locally in NumPy and SciPy. Expand for the formulas and references.
        </p>
      )}
    </Panel>
  );
}
