"use client";

/**
 * Hand-rolled SVG charts.
 *
 * No charting library: every mark here is a few dozen lines of geometry, the
 * bundle stays tiny, and the visual language (colours, axes, marker labels) is
 * identical across panels — which matters more than features when the whole page
 * is one statistical argument.
 */

import { COLORS, isNum, num } from "@/lib/format";

const PAD = { top: 18, right: 16, bottom: 30, left: 46 };

export interface Marker {
  x: number;
  color: string;
  label: string;
  dash?: boolean;
}

function niceTicks(min: number, max: number, count = 5): number[] {
  if (!isNum(min) || !isNum(max) || min === max) return [min];
  const span = max - min;
  const raw = span / count;
  const magnitude = Math.pow(10, Math.floor(Math.log10(raw)));
  const normalized = raw / magnitude;
  const step =
    (normalized >= 5 ? 10 : normalized >= 2 ? 5 : normalized >= 1 ? 2 : 1) * magnitude;
  const start = Math.ceil(min / step) * step;
  const ticks: number[] = [];
  for (let v = start; v <= max + step * 1e-9; v += step) {
    ticks.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  }
  return ticks;
}

function Axes({
  width,
  height,
  xTicks,
  yTicks,
  xScale,
  yScale,
  xLabel,
  yLabel,
  xFormat = (v: number) => num(v, Math.abs(v) >= 100 ? 0 : 2),
  yFormat = (v: number) => num(v, 0),
}: {
  width: number;
  height: number;
  xTicks: number[];
  yTicks: number[];
  xScale: (v: number) => number;
  yScale: (v: number) => number;
  xLabel?: string;
  yLabel?: string;
  xFormat?: (v: number) => string;
  yFormat?: (v: number) => string;
}) {
  return (
    <g>
      {yTicks.map((t) => (
        <g key={`y${t}`}>
          <line
            x1={PAD.left}
            x2={width - PAD.right}
            y1={yScale(t)}
            y2={yScale(t)}
            stroke={COLORS.grid}
            strokeWidth={0.7}
            opacity={0.7}
          />
          <text
            x={PAD.left - 7}
            y={yScale(t) + 3.5}
            textAnchor="end"
            fontSize={10}
            fill={COLORS.muted}
            fontFamily="ui-monospace, monospace"
          >
            {yFormat(t)}
          </text>
        </g>
      ))}
      {xTicks.map((t) => (
        <text
          key={`x${t}`}
          x={xScale(t)}
          y={height - PAD.bottom + 15}
          textAnchor="middle"
          fontSize={10}
          fill={COLORS.muted}
          fontFamily="ui-monospace, monospace"
        >
          {xFormat(t)}
        </text>
      ))}
      <line
        x1={PAD.left}
        x2={width - PAD.right}
        y1={height - PAD.bottom}
        y2={height - PAD.bottom}
        stroke={COLORS.grid}
        strokeWidth={1}
      />
      {xLabel && (
        <text
          x={(PAD.left + width - PAD.right) / 2}
          y={height - 2}
          textAnchor="middle"
          fontSize={10}
          fill={COLORS.muted}
        >
          {xLabel}
        </text>
      )}
      {yLabel && (
        <text
          x={10}
          y={(PAD.top + height - PAD.bottom) / 2}
          textAnchor="middle"
          fontSize={10}
          fill={COLORS.muted}
          transform={`rotate(-90 10 ${(PAD.top + height - PAD.bottom) / 2})`}
        >
          {yLabel}
        </text>
      )}
    </g>
  );
}

function MarkerLines({
  markers,
  xScale,
  height,
  domain,
}: {
  markers: Marker[];
  xScale: (v: number) => number;
  height: number;
  domain: [number, number];
}) {
  const visible = markers.filter(
    (m) => isNum(m.x) && m.x >= domain[0] && m.x <= domain[1]
  );
  return (
    <g>
      {visible.map((m, i) => {
        const x = xScale(m.x);
        const flip = x > (domain[0] + domain[1]) / 2 ? -1 : 1;
        return (
          <g key={`${m.label}-${i}`}>
            <line
              x1={x}
              x2={x}
              y1={PAD.top - 6}
              y2={height - PAD.bottom}
              stroke={m.color}
              strokeWidth={1.3}
              strokeDasharray={m.dash === false ? undefined : "4 3"}
            />
            <text
              x={x + flip * 5}
              y={PAD.top - 8 + (i % 2) * 11}
              textAnchor={flip > 0 ? "start" : "end"}
              fontSize={9.5}
              fill={m.color}
              fontFamily="ui-monospace, monospace"
            >
              {m.label}
            </text>
          </g>
        );
      })}
    </g>
  );
}

/* -------------------------------------------------------------------------- */
/* Histogram                                                                   */
/* -------------------------------------------------------------------------- */
export function Histogram({
  edges,
  counts,
  markers = [],
  color = COLORS.beam,
  colorFor,
  xLabel,
  yLabel = "count",
  height = 230,
  width = 660,
}: {
  edges: number[];
  counts: number[];
  markers?: Marker[];
  color?: string;
  colorFor?: (binStart: number, binEnd: number) => string;
  xLabel?: string;
  yLabel?: string;
  height?: number;
  width?: number;
}) {
  const clean = edges.map((e) => (isNum(e) ? e : 0));
  if (clean.length < 2 || counts.length === 0) {
    return <Empty height={height} width={width} message="no data" />;
  }

  const markerXs = markers.filter((m) => isNum(m.x)).map((m) => m.x);
  const lo = Math.min(clean[0], ...markerXs);
  const hi = Math.max(clean[clean.length - 1], ...markerXs);
  const span = hi - lo || 1;
  const pad = span * 0.04;
  const domain: [number, number] = [lo - pad, hi + pad];
  const maxCount = Math.max(...counts, 1);

  const xScale = (v: number) =>
    PAD.left +
    ((v - domain[0]) / (domain[1] - domain[0])) * (width - PAD.left - PAD.right);
  const yScale = (v: number) =>
    height - PAD.bottom - (v / maxCount) * (height - PAD.top - PAD.bottom);

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="h-auto w-full" role="img">
      <Axes
        width={width}
        height={height}
        xTicks={niceTicks(domain[0], domain[1], 6)}
        yTicks={niceTicks(0, maxCount, 4)}
        xScale={xScale}
        yScale={yScale}
        xLabel={xLabel}
        yLabel={yLabel}
      />
      {counts.map((c, i) => {
        const x0 = xScale(clean[i]);
        const x1 = xScale(clean[i + 1] ?? clean[i]);
        const y = yScale(c);
        const w = Math.max(x1 - x0 - 0.6, 0.6);
        return (
          <rect
            key={i}
            x={x0}
            y={y}
            width={w}
            height={Math.max(height - PAD.bottom - y, 0)}
            fill={colorFor ? colorFor(clean[i], clean[i + 1] ?? clean[i]) : color}
            opacity={0.8}
          />
        );
      })}
      <MarkerLines markers={markers} xScale={xScale} height={height} domain={domain} />
    </svg>
  );
}

/* -------------------------------------------------------------------------- */
/* Scatter with an optional regression line                                    */
/* -------------------------------------------------------------------------- */
export function Scatter({
  xs,
  ys,
  slope,
  intercept,
  xLabel,
  yLabel,
  height = 250,
  width = 660,
  pointColor = COLORS.beam,
}: {
  xs: number[];
  ys: number[];
  slope?: number | null;
  intercept?: number | null;
  xLabel?: string;
  yLabel?: string;
  height?: number;
  width?: number;
  pointColor?: string;
}) {
  const pairs = xs
    .map((x, i) => [x, ys[i]] as [number, number])
    .filter(([x, y]) => isNum(x) && isNum(y));
  if (!pairs.length) return <Empty height={height} width={width} message="no data" />;

  const xValues = pairs.map((p) => p[0]);
  const yValues = pairs.map((p) => p[1]);
  const xDomain = padDomain(Math.min(...xValues), Math.max(...xValues));
  const yDomain = padDomain(Math.min(...yValues, 0), Math.max(...yValues, 0));

  const xScale = (v: number) =>
    PAD.left +
    ((v - xDomain[0]) / (xDomain[1] - xDomain[0])) * (width - PAD.left - PAD.right);
  const yScale = (v: number) =>
    height -
    PAD.bottom -
    ((v - yDomain[0]) / (yDomain[1] - yDomain[0])) * (height - PAD.top - PAD.bottom);

  // Thin very dense clouds so the SVG stays light without changing the shape.
  const step = Math.max(1, Math.ceil(pairs.length / 1600));
  const shown = pairs.filter((_, i) => i % step === 0);

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="h-auto w-full" role="img">
      <Axes
        width={width}
        height={height}
        xTicks={niceTicks(xDomain[0], xDomain[1], 6)}
        yTicks={niceTicks(yDomain[0], yDomain[1], 5)}
        xScale={xScale}
        yScale={yScale}
        xLabel={xLabel}
        yLabel={yLabel}
        yFormat={(v) => num(v, 1)}
      />
      {yDomain[0] < 0 && yDomain[1] > 0 && (
        <line
          x1={PAD.left}
          x2={width - PAD.right}
          y1={yScale(0)}
          y2={yScale(0)}
          stroke={COLORS.muted}
          strokeWidth={0.9}
          strokeDasharray="3 3"
          opacity={0.8}
        />
      )}
      {shown.map(([x, y], i) => (
        <circle key={i} cx={xScale(x)} cy={yScale(y)} r={1.9} fill={pointColor} opacity={0.32} />
      ))}
      {isNum(slope) && isNum(intercept) && (
        <line
          x1={xScale(xDomain[0])}
          y1={yScale(slope * xDomain[0] + intercept)}
          x2={xScale(xDomain[1])}
          y2={yScale(slope * xDomain[1] + intercept)}
          stroke={COLORS.warn}
          strokeWidth={1.8}
        />
      )}
    </svg>
  );
}

/* -------------------------------------------------------------------------- */
/* Equity curve                                                                */
/* -------------------------------------------------------------------------- */
export function EquityCurve({
  t,
  v,
  height = 210,
  width = 660,
  color = COLORS.pass,
}: {
  t: number[];
  v: number[];
  height?: number;
  width?: number;
  color?: string;
}) {
  const points = t
    .map((x, i) => [x, v[i]] as [number, number])
    .filter(([x, y]) => isNum(x) && isNum(y));
  if (points.length < 2) return <Empty height={height} width={width} message="no data" />;

  const xValues = points.map((p) => p[0]);
  const yValues = points.map((p) => p[1]);
  const xDomain: [number, number] = [Math.min(...xValues), Math.max(...xValues)];
  const yDomain = padDomain(Math.min(...yValues, 1), Math.max(...yValues, 1), 0.08);

  const xScale = (x: number) =>
    PAD.left +
    ((x - xDomain[0]) / (xDomain[1] - xDomain[0] || 1)) * (width - PAD.left - PAD.right);
  const yScale = (y: number) =>
    height -
    PAD.bottom -
    ((y - yDomain[0]) / (yDomain[1] - yDomain[0] || 1)) * (height - PAD.top - PAD.bottom);

  const path = points
    .map(([x, y], i) => `${i === 0 ? "M" : "L"}${xScale(x).toFixed(2)} ${yScale(y).toFixed(2)}`)
    .join(" ");
  const area = `${path} L${xScale(xDomain[1]).toFixed(2)} ${yScale(yDomain[0]).toFixed(
    2
  )} L${xScale(xDomain[0]).toFixed(2)} ${yScale(yDomain[0]).toFixed(2)} Z`;

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="h-auto w-full" role="img">
      <defs>
        <linearGradient id="equityFill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity={0.28} />
          <stop offset="100%" stopColor={color} stopOpacity={0} />
        </linearGradient>
      </defs>
      <Axes
        width={width}
        height={height}
        xTicks={niceTicks(xDomain[0], xDomain[1], 6)}
        yTicks={niceTicks(yDomain[0], yDomain[1], 4)}
        xScale={xScale}
        yScale={yScale}
        xLabel="bar"
        yLabel="equity (×)"
        xFormat={(x) => num(x, 0)}
        yFormat={(y) => num(y, 2)}
      />
      <line
        x1={PAD.left}
        x2={width - PAD.right}
        y1={yScale(1)}
        y2={yScale(1)}
        stroke={COLORS.muted}
        strokeDasharray="3 3"
        strokeWidth={0.9}
      />
      <path d={area} fill="url(#equityFill)" />
      <path d={path} fill="none" stroke={color} strokeWidth={1.5} />
    </svg>
  );
}

/* -------------------------------------------------------------------------- */
/* Probability meter (DSR / PSR)                                               */
/* -------------------------------------------------------------------------- */
export function ProbabilityMeter({
  value,
  threshold = 0.95,
  label,
  sublabel,
  color,
}: {
  value: number | null;
  threshold?: number;
  label: string;
  sublabel?: string;
  color: string;
}) {
  const width = 320;
  const height = 108;
  const cx = width / 2;
  const cy = height - 14;
  const radius = 78;
  const stroke = 12;

  const clamped = isNum(value) ? Math.min(Math.max(value, 0), 1) : 0;
  const arc = (fraction: number) => {
    const angle = Math.PI * (1 - fraction);
    return [cx + radius * Math.cos(angle), cy - radius * Math.sin(angle)];
  };
  const [ex, ey] = arc(clamped);
  const [tx, ty] = arc(threshold);
  const largeArc = clamped > 0.5 ? 1 : 0;

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="h-auto w-full" role="img">
      <path
        d={`M ${cx - radius} ${cy} A ${radius} ${radius} 0 0 1 ${cx + radius} ${cy}`}
        fill="none"
        stroke={COLORS.grid}
        strokeWidth={stroke}
        strokeLinecap="round"
      />
      {clamped > 0.001 && (
        <path
          d={`M ${cx - radius} ${cy} A ${radius} ${radius} 0 ${largeArc} 1 ${ex} ${ey}`}
          fill="none"
          stroke={color}
          strokeWidth={stroke}
          strokeLinecap="round"
        />
      )}
      <line
        x1={cx + (radius - stroke) * Math.cos(Math.PI * (1 - threshold))}
        y1={cy - (radius - stroke) * Math.sin(Math.PI * (1 - threshold))}
        x2={tx + (stroke / 2) * Math.cos(Math.PI * (1 - threshold))}
        y2={ty - (stroke / 2) * Math.sin(Math.PI * (1 - threshold))}
        stroke={COLORS.text}
        strokeWidth={1.6}
      />
      <text
        x={cx}
        y={cy - 24}
        textAnchor="middle"
        fontSize={27}
        fill={color}
        fontFamily="ui-monospace, monospace"
      >
        {isNum(value) ? `${(value * 100).toFixed(1)}%` : "—"}
      </text>
      <text x={cx} y={cy - 8} textAnchor="middle" fontSize={10.5} fill={COLORS.muted}>
        {label}
      </text>
      <text x={cx - radius} y={cy + 12} textAnchor="middle" fontSize={9} fill={COLORS.muted}>
        0%
      </text>
      <text x={cx + radius} y={cy + 12} textAnchor="middle" fontSize={9} fill={COLORS.muted}>
        100%
      </text>
      {sublabel && (
        <text x={cx} y={cy + 12} textAnchor="middle" fontSize={9} fill={COLORS.muted}>
          {sublabel}
        </text>
      )}
    </svg>
  );
}

/* -------------------------------------------------------------------------- */
/* Score dial                                                                  */
/* -------------------------------------------------------------------------- */
export function ScoreDial({
  total,
  grade,
  color,
  size = 168,
}: {
  total: number;
  grade: string;
  color: string;
  size?: number;
}) {
  const radius = 68;
  const stroke = 13;
  const circumference = 2 * Math.PI * radius;
  const fraction = Math.min(Math.max(total / 100, 0), 1);

  return (
    <svg viewBox="0 0 168 168" width={size} height={size} role="img">
      <circle
        cx={84}
        cy={84}
        r={radius}
        fill="none"
        stroke={COLORS.grid}
        strokeWidth={stroke}
      />
      <circle
        cx={84}
        cy={84}
        r={radius}
        fill="none"
        stroke={color}
        strokeWidth={stroke}
        strokeLinecap="round"
        strokeDasharray={`${circumference * fraction} ${circumference}`}
        transform="rotate(-90 84 84)"
      />
      <text
        x={84}
        y={80}
        textAnchor="middle"
        fontSize={40}
        fill={COLORS.text}
        fontFamily="ui-monospace, monospace"
      >
        {total.toFixed(0)}
      </text>
      <text x={84} y={99} textAnchor="middle" fontSize={11} fill={COLORS.muted}>
        / 100
      </text>
      <text x={84} y={122} textAnchor="middle" fontSize={15} fill={color} fontWeight={700}>
        GRADE {grade}
      </text>
    </svg>
  );
}

/* -------------------------------------------------------------------------- */
/* Helpers                                                                     */
/* -------------------------------------------------------------------------- */
function padDomain(lo: number, hi: number, fraction = 0.06): [number, number] {
  if (!isNum(lo) || !isNum(hi)) return [0, 1];
  if (lo === hi) return [lo - 0.5, hi + 0.5];
  const pad = (hi - lo) * fraction;
  return [lo - pad, hi + pad];
}

function Empty({
  height,
  width,
  message,
}: {
  height: number;
  width: number;
  message: string;
}) {
  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="h-auto w-full" role="img">
      <text
        x={width / 2}
        y={height / 2}
        textAnchor="middle"
        fontSize={12}
        fill={COLORS.muted}
      >
        {message}
      </text>
    </svg>
  );
}
