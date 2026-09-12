/** Number formatting helpers shared by every panel. */

const DASH = "—";

export function isNum(x: unknown): x is number {
  return typeof x === "number" && Number.isFinite(x);
}

/** Fixed-decimal number, or an em dash when the value is missing. */
export function num(x: number | null | undefined, digits = 2): string {
  return isNum(x) ? x.toFixed(digits) : DASH;
}

/** Signed fixed-decimal number (`+1.24`). */
export function signed(x: number | null | undefined, digits = 2): string {
  if (!isNum(x)) return DASH;
  return `${x >= 0 ? "+" : ""}${x.toFixed(digits)}`;
}

/** Percentage from a 0–1 probability. */
export function pct(x: number | null | undefined, digits = 1): string {
  return isNum(x) ? `${(x * 100).toFixed(digits)}%` : DASH;
}

/** Thousands-separated integer. */
export function int(x: number | null | undefined): string {
  return isNum(x) ? Math.round(x).toLocaleString("en-US") : DASH;
}

/** Compact magnitude for large counts (12 870 → 12.9k). */
export function compact(x: number | null | undefined): string {
  if (!isNum(x)) return DASH;
  const abs = Math.abs(x);
  if (abs >= 1e9) return `${(x / 1e9).toFixed(1)}B`;
  if (abs >= 1e6) return `${(x / 1e6).toFixed(1)}M`;
  if (abs >= 1e4) return `${(x / 1e3).toFixed(1)}k`;
  return Math.round(x).toLocaleString("en-US");
}

/** p-values: small ones get scientific notation rather than a row of zeros. */
export function pval(x: number | null | undefined): string {
  if (!isNum(x)) return DASH;
  if (x === 0) return "<0.0001";
  if (x < 0.0001) return x.toExponential(1);
  return x.toFixed(4);
}

/** MinTRL can legitimately be infinite when the edge is below the benchmark. */
export function trl(x: number | null | undefined): string {
  return isNum(x) ? int(x) : "unreachable";
}

export function durationMs(x: number | null | undefined): string {
  if (!isNum(x)) return DASH;
  return x >= 1000 ? `${(x / 1000).toFixed(2)} s` : `${Math.round(x)} ms`;
}

/** Colour token for a probability-style metric against pass/warn cutoffs. */
export function levelFor(
  value: number | null | undefined,
  passAbove: number,
  warnAbove: number
): "pass" | "warn" | "fail" | "none" {
  if (!isNum(value)) return "none";
  if (value >= passAbove) return "pass";
  if (value >= warnAbove) return "warn";
  return "fail";
}

/** Same idea, inverted: lower is better (PBO, p-values). */
export function levelForLow(
  value: number | null | undefined,
  passBelow: number,
  warnBelow: number
): "pass" | "warn" | "fail" | "none" {
  if (!isNum(value)) return "none";
  if (value <= passBelow) return "pass";
  if (value <= warnBelow) return "warn";
  return "fail";
}

export const COLORS = {
  pass: "#34d399",
  warn: "#fbbf24",
  fail: "#f87171",
  beam: "#60a5fa",
  violet: "#a78bfa",
  muted: "#8b94a8",
  grid: "#2a3142",
  text: "#e6eaf2",
} as const;

export function scoreColor(total: number): string {
  if (total >= 65) return COLORS.pass;
  if (total >= 50) return COLORS.warn;
  return COLORS.fail;
}
