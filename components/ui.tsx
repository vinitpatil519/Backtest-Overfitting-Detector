"use client";

import type { ReactNode } from "react";
import { COLORS, isNum } from "@/lib/format";

/* -------------------------------------------------------------------------- */
/* Layout                                                                      */
/* -------------------------------------------------------------------------- */
export function Panel({
  title,
  subtitle,
  right,
  children,
  className = "",
  bodyClassName = "p-4",
}: {
  title?: string;
  subtitle?: string;
  right?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClassName?: string;
}) {
  return (
    <section className={`panel ${className}`}>
      {(title || right) && (
        <header className="panel-head">
          <div>
            {title && <h2 className="panel-title">{title}</h2>}
            {subtitle && <p className="panel-sub mt-0.5">{subtitle}</p>}
          </div>
          {right && <div className="shrink-0 text-[11px] text-muted">{right}</div>}
        </header>
      )}
      <div className={bodyClassName}>{children}</div>
    </section>
  );
}

/* -------------------------------------------------------------------------- */
/* Stats                                                                       */
/* -------------------------------------------------------------------------- */
const LEVEL_COLOR: Record<string, string> = {
  pass: COLORS.pass,
  warn: COLORS.warn,
  fail: COLORS.fail,
  none: COLORS.text,
  beam: COLORS.beam,
};

export function Stat({
  label,
  value,
  hint,
  level = "none",
  emphasis = false,
}: {
  label: string;
  value: string;
  hint?: string;
  level?: "pass" | "warn" | "fail" | "none" | "beam";
  emphasis?: boolean;
}) {
  return (
    <div
      className={`rounded-lg border border-line bg-panel2 px-3 py-2.5 ${
        emphasis ? "ring-1 ring-beam/25" : ""
      }`}
    >
      <div className="stat-label">{label}</div>
      <div className="stat-value mt-1.5" style={{ color: LEVEL_COLOR[level] }}>
        {value}
      </div>
      {hint && (
        <div className="mt-1.5 text-[10.5px] leading-tight text-muted">{hint}</div>
      )}
    </div>
  );
}

export function Badge({
  children,
  level = "none",
}: {
  children: ReactNode;
  level?: "pass" | "warn" | "fail" | "none";
}) {
  const color = LEVEL_COLOR[level];
  return (
    <span
      className="inline-flex items-center gap-1.5 rounded-md border px-2 py-0.5 text-[11px] font-medium"
      style={{ color, borderColor: `${color}55`, backgroundColor: `${color}14` }}
    >
      {children}
    </span>
  );
}

export function FlagRow({ level, message }: { level: string; message: string }) {
  const color = LEVEL_COLOR[level] ?? COLORS.muted;
  const symbol = level === "pass" ? "✓" : level === "warn" ? "!" : "✕";
  return (
    <li className="flex items-start gap-2.5 py-1.5 text-[12px] leading-relaxed">
      <span
        className="mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center rounded-full text-[10px] font-bold"
        style={{ color, backgroundColor: `${color}1f` }}
      >
        {symbol}
      </span>
      <span className="text-ink/85">{message}</span>
    </li>
  );
}

/* -------------------------------------------------------------------------- */
/* Form controls                                                               */
/* -------------------------------------------------------------------------- */
export function Slider({
  label,
  value,
  min,
  max,
  step = 1,
  display,
  onChange,
  disabled = false,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step?: number;
  display?: string;
  onChange: (value: number) => void;
  disabled?: boolean;
}) {
  return (
    <div className={disabled ? "opacity-40" : ""}>
      <div className="field-label">
        <span>{label}</span>
        <span className="font-mono text-[11px] normal-case tracking-normal text-ink">
          {display ?? value}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
        className="cursor-pointer"
      />
    </div>
  );
}

export function NumberField({
  label,
  value,
  min,
  max,
  step = 1,
  suffix,
  onChange,
  disabled = false,
}: {
  label: string;
  value: number;
  min?: number;
  max?: number;
  step?: number;
  suffix?: string;
  onChange: (value: number) => void;
  disabled?: boolean;
}) {
  return (
    <div className={disabled ? "opacity-40" : ""}>
      <div className="field-label">
        <span>{label}</span>
        {suffix && <span className="normal-case tracking-normal">{suffix}</span>}
      </div>
      <input
        type="number"
        className="input"
        value={Number.isFinite(value) ? value : ""}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        onChange={(e) => {
          const next = Number(e.target.value);
          if (isNum(next)) onChange(next);
        }}
      />
    </div>
  );
}

export function Toggle({
  label,
  checked,
  onChange,
  hint,
}: {
  label: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
  hint?: string;
}) {
  return (
    <button
      type="button"
      onClick={() => onChange(!checked)}
      className="flex w-full items-center justify-between gap-3 rounded-md border border-line bg-panel2 px-2.5 py-2 text-left transition hover:border-beam/50"
      aria-pressed={checked}
    >
      <span>
        <span className="block text-[12px] text-ink">{label}</span>
        {hint && <span className="block text-[10.5px] text-muted">{hint}</span>}
      </span>
      <span
        className="relative h-[18px] w-[32px] shrink-0 rounded-full transition"
        style={{ backgroundColor: checked ? COLORS.beam : "#2a3142" }}
      >
        <span
          className="absolute top-[2px] h-[14px] w-[14px] rounded-full bg-white transition-all"
          style={{ left: checked ? 16 : 2 }}
        />
      </span>
    </button>
  );
}

export function ChipGroup<T extends string>({
  label,
  options,
  selected,
  onChange,
  multi = true,
}: {
  label: string;
  options: { value: T; label: string }[];
  selected: T[];
  onChange: (selected: T[]) => void;
  multi?: boolean;
}) {
  const toggle = (value: T) => {
    if (!multi) {
      onChange([value]);
      return;
    }
    const next = selected.includes(value)
      ? selected.filter((v) => v !== value)
      : [...selected, value];
    // Never allow an empty selection: the engine needs at least one family.
    onChange(next.length ? next : selected);
  };

  return (
    <div>
      <div className="field-label">
        <span>{label}</span>
      </div>
      <div className="flex flex-wrap gap-1.5">
        {options.map((option) => {
          const active = selected.includes(option.value);
          return (
            <button
              key={option.value}
              type="button"
              onClick={() => toggle(option.value)}
              className="chip"
              style={{
                color: active ? COLORS.text : COLORS.muted,
                borderColor: active ? `${COLORS.beam}77` : "rgb(var(--line))",
                backgroundColor: active ? `${COLORS.beam}1a` : "transparent",
              }}
            >
              {option.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}

export function Button({
  children,
  onClick,
  variant = "primary",
  disabled = false,
  type = "button",
  className = "",
}: {
  children: ReactNode;
  onClick?: () => void;
  variant?: "primary" | "ghost";
  disabled?: boolean;
  type?: "button" | "submit";
  className?: string;
}) {
  const base =
    "inline-flex items-center justify-center gap-2 rounded-md px-3.5 py-2 text-[12.5px] font-semibold transition disabled:cursor-not-allowed disabled:opacity-50";
  const styles =
    variant === "primary"
      ? "bg-beam text-[#06101f] hover:brightness-110"
      : "border border-line text-ink hover:border-beam/60";
  return (
    <button type={type} onClick={onClick} disabled={disabled} className={`${base} ${styles} ${className}`}>
      {children}
    </button>
  );
}

/* -------------------------------------------------------------------------- */
/* Feedback                                                                    */
/* -------------------------------------------------------------------------- */
export function Skeleton({ className = "" }: { className?: string }) {
  return (
    <div
      className={`sweep relative overflow-hidden rounded-lg border border-line bg-panel2 ${className}`}
    />
  );
}

export function ErrorNote({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div
      className="rounded-lg border px-4 py-3 text-[12.5px]"
      style={{ borderColor: `${COLORS.fail}55`, backgroundColor: `${COLORS.fail}12`, color: COLORS.fail }}
    >
      <div className="font-semibold">Analysis failed</div>
      <p className="mt-1 text-ink/80">{message}</p>
      {onRetry && (
        <button
          type="button"
          onClick={onRetry}
          className="mt-2 underline underline-offset-2 hover:no-underline"
        >
          Try again
        </button>
      )}
    </div>
  );
}

export function Formula({ children }: { children: ReactNode }) {
  return (
    <pre className="overflow-x-auto rounded-md border border-line bg-panel2 px-3 py-2.5 font-mono text-[11.5px] leading-relaxed text-ink/90">
      {children}
    </pre>
  );
}
