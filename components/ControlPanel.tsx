"use client";

import { useState } from "react";
import {
  Button,
  ChipGroup,
  NumberField,
  Panel,
  Slider,
  Toggle,
} from "@/components/ui";
import { FAMILY_LABELS, type AnalyzeRequest, type Family } from "@/lib/types";

const FAMILY_OPTIONS = (Object.keys(FAMILY_LABELS) as Family[]).map((value) => ({
  value,
  label: FAMILY_LABELS[value],
}));

/** Named starting points that demonstrate each regime in one click. */
export const PRESETS: { name: string; hint: string; patch: Partial<AnalyzeRequest> }[] = [
  {
    name: "Pure noise",
    hint: "Driftless random walk — every winner is luck",
    patch: { mode: "synthetic", drift_annual: 0, autocorr: 0, max_strategies: 1000, n_bars: 1250 },
  },
  {
    name: "Real edge",
    hint: "Trending market that momentum can genuinely capture",
    patch: { mode: "synthetic", drift_annual: 0, autocorr: 0.25, max_strategies: 1000, n_bars: 2000 },
  },
  {
    name: "Short sample",
    hint: "Two years of data, wide grid — the classic trap",
    patch: { mode: "synthetic", drift_annual: 0, autocorr: 0, max_strategies: 1500, n_bars: 500 },
  },
  {
    name: "Crisis tails",
    hint: "Fat-tailed returns that inflate the naive Sharpe",
    patch: { mode: "synthetic", drift_annual: 0, autocorr: 0, fat_tails: true, n_bars: 1500 },
  },
];

export function ControlPanel({
  request,
  onChange,
  onRun,
  running,
}: {
  request: AnalyzeRequest;
  onChange: (patch: Partial<AnalyzeRequest>) => void;
  onRun: () => void;
  running: boolean;
}) {
  const [advanced, setAdvanced] = useState(false);
  const upload = request.mode === "upload";

  return (
    <div className="flex flex-col gap-4">
      <Panel title="Experiment" subtitle="Where the returns come from">
        <div className="flex flex-col gap-4">
          <div className="grid grid-cols-2 gap-1.5">
            {(["synthetic", "upload"] as const).map((mode) => (
              <button
                key={mode}
                type="button"
                onClick={() => onChange({ mode })}
                className={`rounded-md border px-3 py-2 text-[12px] font-medium transition ${
                  request.mode === mode
                    ? "border-beam/60 bg-beam/10 text-ink"
                    : "border-line text-muted hover:border-beam/40"
                }`}
              >
                {mode === "synthetic" ? "Synthetic lab" : "My returns"}
              </button>
            ))}
          </div>

          {upload ? (
            <div>
              <div className="field-label">
                <span>Returns (one column per strategy)</span>
                <span className="normal-case tracking-normal">CSV / TSV</span>
              </div>
              <textarea
                className="input h-40 resize-y leading-relaxed"
                placeholder={"strategy_a,strategy_b\n0.0041,-0.0012\n-0.0008,0.0033\n..."}
                value={request.csv}
                onChange={(e) => onChange({ csv: e.target.value })}
                spellCheck={false}
              />
              <p className="mt-1.5 text-[10.5px] leading-relaxed text-muted">
                Per-period returns, header optional. Decimals or percentages both work. Two or
                more columns unlock PBO; a single column needs the trial count below.
              </p>
              <div className="mt-3">
                <NumberField
                  label="Trials you ran"
                  value={request.declared_trials}
                  min={0}
                  step={1}
                  suffix="for deflation"
                  onChange={(declared_trials) => onChange({ declared_trials })}
                />
              </div>
            </div>
          ) : (
            <>
              <div className="grid grid-cols-2 gap-2">
                {PRESETS.map((preset) => (
                  <button
                    key={preset.name}
                    type="button"
                    title={preset.hint}
                    onClick={() => onChange(preset.patch)}
                    className="rounded-md border border-line px-2.5 py-2 text-left transition hover:border-beam/50"
                  >
                    <span className="block text-[11.5px] font-medium text-ink">
                      {preset.name}
                    </span>
                    <span className="mt-0.5 block text-[10px] leading-tight text-muted">
                      {preset.hint}
                    </span>
                  </button>
                ))}
              </div>

              <Slider
                label="Strategies tested"
                value={request.max_strategies}
                min={50}
                max={2000}
                step={50}
                display={request.max_strategies.toLocaleString()}
                onChange={(max_strategies) => onChange({ max_strategies })}
              />
              <Slider
                label="Bars of history"
                value={request.n_bars}
                min={250}
                max={5000}
                step={50}
                display={`${request.n_bars.toLocaleString()} (${(
                  request.n_bars / request.periods_per_year
                ).toFixed(1)}y)`}
                onChange={(n_bars) => onChange({ n_bars })}
              />
              <ChipGroup
                label="Strategy families"
                options={FAMILY_OPTIONS}
                selected={request.families}
                onChange={(families) => onChange({ families })}
              />
            </>
          )}
        </div>
      </Panel>

      {!upload && (
        <Panel title="Market" subtitle="Truth the strategies are searching in">
          <div className="flex flex-col gap-4">
            <Slider
              label="True drift"
              value={Math.round(request.drift_annual * 100)}
              min={-30}
              max={50}
              step={1}
              display={`${(request.drift_annual * 100).toFixed(0)}% / yr`}
              onChange={(v) => onChange({ drift_annual: v / 100 })}
            />
            <Slider
              label="Serial correlation"
              value={Math.round(request.autocorr * 100)}
              min={-50}
              max={50}
              step={1}
              display={request.autocorr.toFixed(2)}
              onChange={(v) => onChange({ autocorr: v / 100 })}
            />
            <p className="-mt-1 text-[10.5px] leading-relaxed text-muted">
              Both controls are zero by default, so nothing in the grid has an edge. Positive
              serial correlation is the honest way to create one: it makes momentum genuinely
              predictive instead of just handing out free beta.
            </p>
            <Slider
              label="Volatility"
              value={Math.round(request.vol_annual * 100)}
              min={5}
              max={80}
              step={1}
              display={`${(request.vol_annual * 100).toFixed(0)}% / yr`}
              onChange={(v) => onChange({ vol_annual: v / 100 })}
            />
            <Slider
              label="Trading cost"
              value={request.cost_bps}
              min={0}
              max={25}
              step={0.5}
              display={`${request.cost_bps} bps / turn`}
              onChange={(cost_bps) => onChange({ cost_bps })}
            />
            <Toggle
              label="Fat tails"
              hint="Student-t innovations (df = 4)"
              checked={request.fat_tails}
              onChange={(fat_tails) => onChange({ fat_tails })}
            />
          </div>
        </Panel>
      )}

      <Panel
        title="Statistics"
        subtitle="CSCV, Monte Carlo and inference settings"
        right={
          <button
            type="button"
            onClick={() => setAdvanced((v) => !v)}
            className="underline underline-offset-2 hover:text-ink"
          >
            {advanced ? "hide" : "show"}
          </button>
        }
      >
        {advanced ? (
          <div className="flex flex-col gap-4">
            <Slider
              label="CSCV splits (S)"
              value={request.n_splits}
              min={4}
              max={16}
              step={2}
              display={`${request.n_splits} → ${binomial(request.n_splits).toLocaleString()} combos`}
              onChange={(n_splits) => onChange({ n_splits })}
            />
            <Slider
              label="Combination cap"
              value={request.max_combos}
              min={100}
              max={12870}
              step={100}
              display={request.max_combos.toLocaleString()}
              onChange={(max_combos) => onChange({ max_combos })}
            />
            <Slider
              label="Monte Carlo paths"
              value={request.mc_paths}
              min={1000}
              max={100000}
              step={1000}
              display={request.mc_paths.toLocaleString()}
              onChange={(mc_paths) => onChange({ mc_paths })}
            />
            <Slider
              label="Bootstrap resamples"
              value={request.bootstrap_samples}
              min={500}
              max={10000}
              step={500}
              display={request.bootstrap_samples.toLocaleString()}
              onChange={(bootstrap_samples) => onChange({ bootstrap_samples })}
            />
            <Slider
              label="Bootstrap block"
              value={request.block_size}
              min={1}
              max={100}
              step={1}
              display={`${request.block_size} bars`}
              onChange={(block_size) => onChange({ block_size })}
            />
            <Slider
              label="Confidence"
              value={Math.round(request.confidence * 100)}
              min={50}
              max={99}
              step={1}
              display={`${(request.confidence * 100).toFixed(0)}%`}
              onChange={(v) => onChange({ confidence: v / 100 })}
            />
            <div className="grid grid-cols-2 gap-3">
              <NumberField
                label="Benchmark SR"
                value={request.benchmark_sr_annual}
                step={0.1}
                suffix="annual"
                onChange={(benchmark_sr_annual) => onChange({ benchmark_sr_annual })}
              />
              <NumberField
                label="Periods / year"
                value={request.periods_per_year}
                min={1}
                step={1}
                onChange={(periods_per_year) => onChange({ periods_per_year })}
              />
            </div>
            <NumberField
              label="Random seed"
              value={request.seed}
              min={0}
              step={1}
              suffix="reproducible"
              onChange={(seed) => onChange({ seed })}
            />
          </div>
        ) : (
          <p className="text-[11.5px] leading-relaxed text-muted">
            {request.n_splits} CSCV splits, {request.mc_paths.toLocaleString()} Monte Carlo
            paths, {request.bootstrap_samples.toLocaleString()} block-bootstrap resamples,
            seed {request.seed}.
          </p>
        )}
      </Panel>

      <div className="sticky bottom-4 z-10 flex gap-2">
        <Button onClick={onRun} disabled={running} className="flex-1 shadow-lg">
          {running ? "Running…" : "Run analysis"}
        </Button>
        <Button
          variant="ghost"
          disabled={running}
          onClick={() => onChange({ seed: Math.floor(Math.random() * 100000) })}
        >
          New seed
        </Button>
      </div>
    </div>
  );
}

function binomial(s: number): number {
  const k = Math.floor(s / 2);
  let result = 1;
  for (let i = 1; i <= k; i += 1) {
    result = (result * (s - k + i)) / i;
  }
  return Math.round(result);
}
