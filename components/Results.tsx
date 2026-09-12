"use client";

import {
  EquityCurve,
  Histogram,
  ProbabilityMeter,
  Scatter,
  ScoreDial,
  type Marker,
} from "@/components/charts";
import { Badge, FlagRow, Panel, Stat } from "@/components/ui";
import {
  COLORS,
  compact,
  durationMs,
  int,
  isNum,
  levelFor,
  levelForLow,
  num,
  pct,
  pval,
  scoreColor,
  signed,
  trl,
} from "@/lib/format";
import type { AnalysisResult } from "@/lib/types";

export function Results({ result }: { result: AnalysisResult }) {
  const { meta, universe, best, deflated, pbo, montecarlo, bootstrap, score } = result;
  const color = scoreColor(score.total);

  return (
    <div className="flex flex-col gap-4">
      <Verdict result={result} color={color} />
      <Kpis result={result} />

      <Panel
        title="Sharpe distribution across the search"
        subtitle={`${int(meta.n_strategies)} strategies backtested on the same ${int(
          meta.n_observations
        )} bars`}
        right={
          <span>
            spread σ = {num(universe.sharpe_std)} · median {num(universe.sharpe_mean)}
          </span>
        }
      >
        <Histogram
          edges={universe.sharpe_histogram.edges}
          counts={universe.sharpe_histogram.counts}
          xLabel="annualised Sharpe ratio"
          yLabel="strategies"
          markers={
            [
              isNum(best.sharpe_annual) && {
                x: best.sharpe_annual,
                color: COLORS.fail,
                label: `best ${num(best.sharpe_annual)}`,
              },
              isNum(deflated.sr_star_annual) && {
                x: deflated.sr_star_annual,
                color: COLORS.warn,
                label: `SR* ${num(deflated.sr_star_annual)}`,
              },
              { x: 0, color: COLORS.muted, label: "0" },
            ].filter(Boolean) as Marker[]
          }
        />
        <p className="mt-2 text-[11.5px] leading-relaxed text-muted">
          The spread of this distribution is what makes the best result unremarkable. SR* is the
          Sharpe you should expect to reach by luck alone after {int(deflated.n_eff)} effective
          trials — a winner to the left of it has earned nothing.
        </p>
      </Panel>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Deflation result={result} />
        <MonteCarloPanel result={result} />
      </div>

      {pbo ? (
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          <Panel
            title="Probability of Backtest Overfitting"
            subtitle={`CSCV · ${pbo.n_splits} splits · ${compact(pbo.n_combos)} of ${compact(
              pbo.n_combos_total
            )} combinations`}
            right={
              <Badge level={levelForLow(pbo.pbo, 0.25, 0.5)}>PBO {pct(pbo.pbo)}</Badge>
            }
          >
            <Histogram
              edges={pbo.logit_histogram.edges}
              counts={pbo.logit_histogram.counts}
              xLabel="logit of out-of-sample rank"
              yLabel="combinations"
              colorFor={(start) => (start < 0 ? COLORS.fail : COLORS.pass)}
              markers={[{ x: 0, color: COLORS.text, label: "", dash: false }]}
            />
            <p className="mt-2 text-[11.5px] leading-relaxed text-muted">
              Each combination picks the in-sample winner, then measures where it ranks
              out-of-sample. Mass to the left of zero (red) is the winner landing in the bottom
              half; PBO is that share. Median rank sits at{" "}
              {pct(pbo.median_relative_rank)} of the field.
            </p>
          </Panel>

          <Panel
            title="Performance degradation"
            subtitle="In-sample Sharpe against the same strategy out-of-sample"
            right={
              <span>
                slope {num(pbo.slope)} · R² {num(pbo.r_squared)}
              </span>
            }
          >
            <Scatter
              xs={pbo.is_sharpe}
              ys={pbo.oos_sharpe}
              slope={pbo.slope}
              intercept={pbo.intercept}
              xLabel="in-sample Sharpe (annualised)"
              yLabel="out-of-sample Sharpe"
            />
            <div className="mt-2 grid grid-cols-3 gap-2">
              <Mini label="Mean IS" value={num(pbo.is_sharpe_mean)} />
              <Mini
                label="Mean OOS"
                value={num(pbo.oos_sharpe_mean)}
                level={levelFor(pbo.oos_sharpe_mean, 0.2, 0)}
              />
              <Mini
                label="P(OOS ≤ 0)"
                value={pct(pbo.prob_oos_loss)}
                level={levelForLow(pbo.prob_oos_loss, 0.25, 0.5)}
              />
            </div>
            <p className="mt-2 text-[11.5px] leading-relaxed text-muted">
              A slope near or below zero means in-sample success carries no information about
              future performance — selecting on the backtest is then worse than useless. The
              average selected strategy loses {num(pbo.degradation)} Sharpe points out-of-sample.
            </p>
          </Panel>
        </div>
      ) : (
        <Panel title="Probability of Backtest Overfitting" subtitle="Unavailable for this input">
          <p className="text-[12px] leading-relaxed text-muted">
            {result.pbo_error ??
              "CSCV needs the full matrix of candidate strategies, not just the winner."}{" "}
            Paste two or more return columns to run it.
          </p>
        </Panel>
      )}

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-5">
        <Panel
          className="lg:col-span-3"
          title="Selected strategy"
          subtitle={best.label}
          right={
            <span>
              {pct(best.hit_rate)} winning bars
              {isNum(best.turnover_annual) ? ` · ${num(best.turnover_annual, 0)} turns/yr` : ""}
            </span>
          }
        >
          <EquityCurve t={best.equity.t} v={best.equity.v} color={color} />
          <div className="mt-2 grid grid-cols-2 gap-2 sm:grid-cols-4">
            <Mini label="Total return" value={pct(best.total_return, 0)} />
            <Mini label="Max drawdown" value={pct(best.max_drawdown, 1)} level="fail" />
            <Mini label="Calmar" value={num(best.calmar)} />
            <Mini label="Volatility" value={pct(best.vol_annual, 1)} />
          </div>
        </Panel>

        <Panel
          className="lg:col-span-2"
          title="Top of the search"
          subtitle="Best in-sample Sharpes — the shortlist a researcher would keep"
          bodyClassName="p-0"
        >
          <ul className="divide-y divide-line">
            {universe.leaderboard.map((row) => (
              <li
                key={row.index}
                className={`flex items-center justify-between gap-3 px-4 py-2 text-[11.5px] ${
                  row.index === best.index ? "bg-beam/8" : ""
                }`}
              >
                <span className="flex min-w-0 items-center gap-2">
                  <span className="w-4 shrink-0 font-mono text-[10px] text-muted">
                    {row.rank}
                  </span>
                  <span className="truncate text-ink/85">{row.label}</span>
                </span>
                <span className="shrink-0 font-mono" style={{ color: COLORS.text }}>
                  {num(row.sharpe_annual)}
                </span>
              </li>
            ))}
          </ul>
        </Panel>
      </div>

      <Panel title="Run details" subtitle="Reproduce this exact result">
        <div className="grid grid-cols-2 gap-x-6 gap-y-2 text-[11.5px] sm:grid-cols-4">
          <Detail label="Mode" value={meta.mode === "upload" ? "uploaded returns" : "synthetic lab"} />
          <Detail label="Seed" value={String(meta.seed)} />
          <Detail label="Observations" value={int(meta.n_observations)} />
          <Detail label="Warm-up trimmed" value={`${int(meta.warmup_bars)} bars`} />
          <Detail label="Periods / year" value={int(meta.periods_per_year)} />
          <Detail label="Cost" value={`${num(meta.cost_bps, 1)} bps`} />
          <Detail label="True drift" value={pct(meta.drift_annual, 0)} />
          <Detail label="Serial correlation" value={num(meta.autocorr)} />
          <Detail label="Engine" value={`v${meta.engine_version}`} />
          <Detail label="Compute time" value={durationMs(meta.runtime_ms)} />
        </div>
        {meta.notes.length > 0 && (
          <ul className="mt-3 space-y-1 border-t border-line pt-3">
            {meta.notes.map((note, i) => (
              <li key={i} className="text-[11.5px] leading-relaxed text-muted">
                — {note}
              </li>
            ))}
          </ul>
        )}
      </Panel>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Verdict                                                                     */
/* -------------------------------------------------------------------------- */
function Verdict({ result, color }: { result: AnalysisResult; color: string }) {
  const { score } = result;
  return (
    <section className="panel overflow-hidden">
      <div className="grid grid-cols-1 gap-6 p-5 lg:grid-cols-[auto_1fr]">
        <div className="flex flex-col items-center justify-center gap-1">
          <ScoreDial total={score.total} grade={score.grade} color={color} />
        </div>

        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <h2 className="text-[20px] font-semibold tracking-tight" style={{ color }}>
              {score.verdict}
            </h2>
            <Badge level={score.total >= 65 ? "pass" : score.total >= 50 ? "warn" : "fail"}>
              robustness {score.total.toFixed(1)}
            </Badge>
          </div>
          <p className="mt-1.5 max-w-2xl text-[13px] leading-relaxed text-ink/80">
            {score.headline}
          </p>

          <div className="mt-4 space-y-2">
            {score.components.map((component) => (
              <div key={component.key} className="flex items-center gap-3">
                <span className="w-[190px] shrink-0 text-[11px] text-muted">
                  {component.label}
                </span>
                <span className="relative h-[6px] flex-1 overflow-hidden rounded-full bg-line">
                  {component.available && isNum(component.value) && (
                    <span
                      className="absolute inset-y-0 left-0 rounded-full transition-all"
                      style={{
                        width: `${Math.min(Math.max(component.value * 100, 0), 100)}%`,
                        backgroundColor: scoreColor(component.value * 100),
                      }}
                    />
                  )}
                </span>
                <span className="w-[92px] shrink-0 text-right font-mono text-[11px] text-ink/80">
                  {component.available ? pct(component.value) : "n/a"}
                  <span className="ml-1 text-[10px] text-muted">
                    ×{component.weight.toFixed(2)}
                  </span>
                </span>
              </div>
            ))}
          </div>

          <ul className="mt-4 border-t border-line pt-3">
            {score.flags.map((flag, i) => (
              <FlagRow key={i} level={flag.level} message={flag.message} />
            ))}
          </ul>
        </div>
      </div>
    </section>
  );
}

/* -------------------------------------------------------------------------- */
/* KPI grid                                                                    */
/* -------------------------------------------------------------------------- */
function Kpis({ result }: { result: AnalysisResult }) {
  const { best, deflated, pbo, montecarlo, meta } = result;
  return (
    <div className="grid grid-cols-2 gap-2.5 sm:grid-cols-3 xl:grid-cols-6">
      <Stat
        label="Best Sharpe"
        value={num(best.sharpe_annual)}
        hint={`${pct(meta.confidence, 0)} CI ${num(best.sharpe_ci_low)} to ${num(
          best.sharpe_ci_high
        )}`}
        level="beam"
        emphasis
      />
      <Stat
        label="Deflated Sharpe"
        value={pct(deflated.dsr)}
        hint="P(true SR > selection benchmark)"
        level={levelFor(deflated.dsr, 0.95, 0.5)}
        emphasis
      />
      <Stat
        label="PBO"
        value={pbo ? pct(pbo.pbo) : "n/a"}
        hint="P(winner ranks bottom half OOS)"
        level={pbo ? levelForLow(pbo.pbo, 0.25, 0.5) : "none"}
        emphasis
      />
      <Stat
        label="Naive PSR"
        value={pct(deflated.psr_benchmark)}
        hint="Before correcting for the search"
      />
      <Stat
        label="MC p-value"
        value={pval(montecarlo.p_value)}
        hint="vs a zero-edge grid of the same size"
        level={levelForLow(montecarlo.p_value, 0.05, 0.2)}
      />
      <Stat
        label="MinTRL"
        value={trl(deflated.min_trl)}
        hint={
          isNum(deflated.min_trl_years)
            ? `${num(deflated.min_trl_years, 1)} years needed · have ${num(
                meta.n_observations / meta.periods_per_year,
                1
              )}`
            : "edge is below the benchmark"
        }
        level={deflated.track_record_sufficient ? "pass" : "warn"}
      />
      <Stat label="Skew" value={signed(best.skew)} hint="third moment, bias-corrected" />
      <Stat
        label="Kurtosis"
        value={num(best.kurtosis)}
        hint={`excess ${signed(best.excess_kurtosis)} · Gaussian = 3`}
      />
      <Stat label="Max drawdown" value={pct(best.max_drawdown, 1)} hint={`${int(best.max_drawdown_length)} bars underwater`} />
      <Stat label="Trials" value={int(deflated.n_trials)} hint="candidate strategies searched" />
      <Stat
        label="Effective trials"
        value={int(deflated.n_eff)}
        hint={
          deflated.n_eff_method?.startsWith("clustering")
            ? deflated.n_eff_method
            : `mean |ρ| = ${num(deflated.mean_abs_correlation)}`
        }
      />
      <Stat
        label="Selection benchmark"
        value={num(deflated.sr_star_annual)}
        hint="SR* — free Sharpe from searching"
        level="warn"
      />
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Deflation panel                                                             */
/* -------------------------------------------------------------------------- */
function Deflation({ result }: { result: AnalysisResult }) {
  const { deflated, best, meta } = result;
  const dsrColor =
    isNum(deflated.dsr) && deflated.dsr >= 0.95
      ? COLORS.pass
      : isNum(deflated.dsr) && deflated.dsr >= 0.5
      ? COLORS.warn
      : COLORS.fail;

  return (
    <Panel
      title="Deflated Sharpe Ratio"
      subtitle="The same Sharpe, priced for how hard you looked"
      right={<Badge level={levelFor(deflated.dsr, 0.95, 0.5)}>{pct(deflated.dsr)}</Badge>}
    >
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        <ProbabilityMeter
          value={deflated.psr_benchmark}
          label="Probabilistic Sharpe (naive)"
          color={COLORS.beam}
        />
        <ProbabilityMeter value={deflated.dsr} label="Deflated Sharpe" color={dsrColor} />
      </div>

      <div className="mt-3 space-y-1.5 border-t border-line pt-3 text-[11.5px]">
        <Row
          label="Observed Sharpe"
          value={`${num(best.sharpe_annual)} annual · ${num(best.sharpe_period, 4)} per bar`}
        />
        <Row label="Trials searched" value={`${int(deflated.n_trials)} → ${int(deflated.n_eff)} effective`} />
        <Row
          label="Spread of trial Sharpes"
          value={`V[SR] = ${num(deflated.variance_of_sharpes, 5)}`}
        />
        <Row
          label="Selection benchmark SR*"
          value={`${num(deflated.sr_star_annual)} annual`}
          highlight
        />
        <Row
          label="Margin over benchmark"
          value={signed(
            isNum(best.sharpe_annual) && isNum(deflated.sr_star_annual)
              ? best.sharpe_annual - deflated.sr_star_annual
              : null
          )}
          highlight
        />
        <Row
          label="Track record"
          value={`${int(meta.n_observations)} bars vs MinTRL ${trl(deflated.min_trl)}`}
        />
      </div>

      <p className="mt-3 text-[11.5px] leading-relaxed text-muted">
        The naive PSR asks whether the Sharpe beats zero. The deflated version asks whether it
        beats the best you would expect from {int(deflated.n_eff)} independent attempts on data
        with no edge at all — a far higher bar, and the only one that survives contact with a
        parameter sweep.
      </p>
    </Panel>
  );
}

/* -------------------------------------------------------------------------- */
/* Monte Carlo panel                                                           */
/* -------------------------------------------------------------------------- */
function MonteCarloPanel({ result }: { result: AnalysisResult }) {
  const { montecarlo, bootstrap, best } = result;
  return (
    <Panel
      title="Monte Carlo validation"
      subtitle={`${compact(montecarlo.n_paths)} simulated searches · ${compact(
        bootstrap.n_samples
      )} block-bootstrap resamples`}
      right={
        <Badge level={levelForLow(montecarlo.p_value, 0.05, 0.2)}>p = {pval(montecarlo.p_value)}</Badge>
      }
    >
      <div className="text-[11px] font-medium uppercase tracking-wider text-muted">
        Best Sharpe when nothing has an edge
      </div>
      <Histogram
        edges={montecarlo.histogram.edges}
        counts={montecarlo.histogram.counts}
        color={COLORS.muted}
        height={190}
        xLabel="annualised Sharpe of the best trial"
        yLabel="paths"
        markers={
          [
            isNum(montecarlo.observed_annual) && {
              x: montecarlo.observed_annual,
              color: COLORS.fail,
              label: `observed ${num(montecarlo.observed_annual)}`,
              dash: false,
            },
            isNum(montecarlo.q95) && {
              x: montecarlo.q95,
              color: COLORS.warn,
              label: `null 95% ${num(montecarlo.q95)}`,
            },
          ].filter(Boolean) as Marker[]
        }
      />

      <div className="mt-3 text-[11px] font-medium uppercase tracking-wider text-muted">
        Bootstrap distribution of the selected strategy
      </div>
      <Histogram
        edges={bootstrap.histogram.edges}
        counts={bootstrap.histogram.counts}
        color={COLORS.beam}
        height={190}
        xLabel="annualised Sharpe"
        yLabel="resamples"
        markers={
          [
            { x: 0, color: COLORS.muted, label: "0" },
            isNum(bootstrap.ci_low) && {
              x: bootstrap.ci_low,
              color: COLORS.warn,
              label: `${num(bootstrap.ci_low)}`,
            },
            isNum(bootstrap.ci_high) && {
              x: bootstrap.ci_high,
              color: COLORS.warn,
              label: `${num(bootstrap.ci_high)}`,
            },
          ].filter(Boolean) as Marker[]
        }
      />

      <div className="mt-3 grid grid-cols-3 gap-2">
        <Mini label="Null E[max]" value={num(montecarlo.mean)} />
        <Mini label="Observed" value={num(best.sharpe_annual)} level="beam" />
        <Mini
          label="P(SR > 0)"
          value={pct(bootstrap.prob_positive)}
          level={levelFor(bootstrap.prob_positive, 0.95, 0.75)}
        />
      </div>
      <p className="mt-2 text-[11.5px] leading-relaxed text-muted">
        The grey distribution is what {int(montecarlo.n_trials_effective)} worthless strategies
        produce when you keep only the best one. If the red line sits inside that cloud, the
        backtest has told you nothing the search would not have produced on its own.
      </p>
    </Panel>
  );
}

/* -------------------------------------------------------------------------- */
/* Small pieces                                                                */
/* -------------------------------------------------------------------------- */
function Mini({
  label,
  value,
  level = "none",
}: {
  label: string;
  value: string;
  level?: "pass" | "warn" | "fail" | "none" | "beam";
}) {
  const color =
    level === "none"
      ? COLORS.text
      : level === "beam"
      ? COLORS.beam
      : level === "pass"
      ? COLORS.pass
      : level === "warn"
      ? COLORS.warn
      : COLORS.fail;
  return (
    <div className="rounded-md border border-line bg-panel2 px-2.5 py-1.5">
      <div className="text-[10px] uppercase tracking-wider text-muted">{label}</div>
      <div className="mt-0.5 font-mono text-[13px]" style={{ color }}>
        {value}
      </div>
    </div>
  );
}

function Row({
  label,
  value,
  highlight = false,
}: {
  label: string;
  value: string;
  highlight?: boolean;
}) {
  return (
    <div className="flex items-baseline justify-between gap-4">
      <span className="text-muted">{label}</span>
      <span
        className="shrink-0 font-mono"
        style={{ color: highlight ? COLORS.warn : "rgb(var(--ink))" }}
      >
        {value}
      </span>
    </div>
  );
}

function Detail({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-[10px] uppercase tracking-wider text-muted">{label}</div>
      <div className="mt-0.5 font-mono text-[12px] text-ink/85">{value}</div>
    </div>
  );
}
