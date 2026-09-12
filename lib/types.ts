/** Shapes returned by `POST /api/py/analyze` — mirrors `engine/pipeline.py`. */

export type Level = "pass" | "warn" | "fail";

export interface Histogram {
  counts: number[];
  edges: number[];
}

export interface Flag {
  level: Level;
  message: string;
}

export interface ScoreComponent {
  key: string;
  label: string;
  weight: number;
  value: number | null;
  available: boolean;
  contribution: number | null;
}

export interface Score {
  total: number;
  grade: string;
  verdict: string;
  headline: string;
  components: ScoreComponent[];
  flags: Flag[];
}

export interface Meta {
  mode: string;
  periods_per_year: number;
  seed: number;
  n_observations: number;
  n_strategies: number;
  warmup_bars: number;
  cost_bps: number | null;
  drift_annual: number | null;
  vol_annual: number | null;
  autocorr: number | null;
  fat_tails: boolean;
  confidence: number | null;
  benchmark_sr_annual: number | null;
  notes: string[];
  runtime_ms: number;
  engine_version: string;
}

export interface LeaderboardRow {
  rank: number;
  index: number;
  label: string;
  family: string;
  sharpe_annual: number | null;
}

export interface Universe {
  sharpes_annual: number[];
  sharpe_histogram: Histogram;
  sharpe_mean: number | null;
  sharpe_std: number | null;
  sharpe_max: number | null;
  sharpe_min: number | null;
  family_counts: Record<string, number>;
  leaderboard: LeaderboardRow[];
}

export interface Best {
  index: number;
  label: string;
  family: string;
  config: Record<string, string | number>;
  sharpe_period: number | null;
  sharpe_annual: number | null;
  sharpe_se_annual: number | null;
  sharpe_ci_low: number | null;
  sharpe_ci_high: number | null;
  mean_period: number | null;
  vol_annual: number | null;
  skew: number | null;
  kurtosis: number | null;
  excess_kurtosis: number | null;
  hit_rate: number | null;
  turnover_annual: number | null;
  max_drawdown: number | null;
  max_drawdown_length: number | null;
  total_return: number | null;
  calmar: number | null;
  equity: { t: number[]; v: number[] };
}

export interface Deflated {
  n_trials: number | null;
  n_eff: number | null;
  /** How N_eff was estimated: "correlation" or "clustering (k=...)". */
  n_eff_method: string;
  mean_abs_correlation: number | null;
  variance_of_sharpes: number | null;
  sr_star_period: number | null;
  sr_star_annual: number | null;
  sr_star_naive_annual: number | null;
  dsr: number | null;
  psr_benchmark: number | null;
  min_trl: number | null;
  min_trl_years: number | null;
  track_record_sufficient: boolean;
}

export interface Pbo {
  pbo: number | null;
  n_splits: number;
  block_length: number;
  n_combos: number;
  n_combos_total: number;
  observations_used: number;
  median_logit: number | null;
  median_relative_rank: number | null;
  logit_histogram: Histogram;
  is_sharpe: number[];
  oos_sharpe: number[];
  is_sharpe_mean: number | null;
  oos_sharpe_mean: number | null;
  degradation: number | null;
  slope: number | null;
  intercept: number | null;
  r_squared: number | null;
  prob_oos_loss: number | null;
  most_selected: { index: number; count: number; label: string }[];
}

export interface MonteCarlo {
  n_paths: number;
  n_trials_effective: number;
  sigma_sharpe_annual: number | null;
  histogram: Histogram;
  mean: number | null;
  median: number | null;
  q95: number | null;
  q99: number | null;
  observed_annual: number | null;
  p_value: number | null;
}

export interface Bootstrap {
  n_samples: number;
  block_size: number;
  histogram: Histogram;
  mean: number | null;
  median: number | null;
  std: number | null;
  ci_low: number | null;
  ci_high: number | null;
  prob_positive: number | null;
  confidence: number | null;
}

export interface AnalysisResult {
  meta: Meta;
  universe: Universe;
  best: Best;
  deflated: Deflated;
  pbo: Pbo | null;
  pbo_error: string | null;
  montecarlo: MonteCarlo;
  bootstrap: Bootstrap;
  score: Score;
}

export type Family = "ma_cross" | "momentum" | "mean_reversion" | "breakout";

export interface AnalyzeRequest {
  mode: "synthetic" | "upload";
  n_bars: number;
  periods_per_year: number;
  seed: number;
  drift_annual: number;
  vol_annual: number;
  fat_tails: boolean;
  autocorr: number;
  cost_bps: number;
  max_strategies: number;
  families: Family[];
  n_splits: number;
  max_combos: number;
  mc_paths: number;
  bootstrap_samples: number;
  block_size: number;
  benchmark_sr_annual: number;
  confidence: number;
  csv: string;
  declared_trials: number;
}

export const FAMILY_LABELS: Record<Family, string> = {
  ma_cross: "MA crossover",
  momentum: "Momentum",
  mean_reversion: "Mean reversion",
  breakout: "Breakout",
};

export const DEFAULT_REQUEST: AnalyzeRequest = {
  mode: "synthetic",
  n_bars: 1250,
  periods_per_year: 252,
  seed: 7,
  drift_annual: 0,
  vol_annual: 0.2,
  fat_tails: false,
  autocorr: 0,
  cost_bps: 1,
  max_strategies: 1000,
  families: ["ma_cross", "momentum", "mean_reversion", "breakout"],
  n_splits: 10,
  max_combos: 4000,
  mc_paths: 20000,
  bootstrap_samples: 2000,
  block_size: 20,
  benchmark_sr_annual: 0,
  confidence: 0.95,
  csv: "",
  declared_trials: 0,
};
