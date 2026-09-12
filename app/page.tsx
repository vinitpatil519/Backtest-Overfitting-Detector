"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ControlPanel } from "@/components/ControlPanel";
import { Methodology } from "@/components/Methodology";
import { Results } from "@/components/Results";
import { ErrorNote, Skeleton } from "@/components/ui";
import { analyze } from "@/lib/api";
import { DEFAULT_REQUEST, type AnalysisResult, type AnalyzeRequest } from "@/lib/types";

export default function Page() {
  const [request, setRequest] = useState<AnalyzeRequest>(DEFAULT_REQUEST);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const inFlight = useRef<AbortController | null>(null);

  const run = useCallback(
    async (payload: AnalyzeRequest) => {
      inFlight.current?.abort();
      const controller = new AbortController();
      inFlight.current = controller;

      setRunning(true);
      setError(null);
      try {
        setResult(await analyze(payload, controller.signal));
      } catch (err) {
        if ((err as Error)?.name === "AbortError") return;
        setError((err as Error)?.message ?? "Unexpected error.");
      } finally {
        if (inFlight.current === controller) {
          inFlight.current = null;
          setRunning(false);
        }
      }
    },
    []
  );

  // Kick off the default synthetic experiment so the page is never empty.
  useEffect(() => {
    void run(DEFAULT_REQUEST);
    return () => inFlight.current?.abort();
  }, [run]);

  const patch = (next: Partial<AnalyzeRequest>) =>
    setRequest((current) => ({ ...current, ...next }));

  return (
    <main className="mx-auto min-h-screen w-full max-w-[1500px] px-4 py-6 lg:px-7">
      <Header running={running} meta={result?.meta.runtime_ms ?? null} />

      <div className="mt-6 grid grid-cols-1 gap-5 xl:grid-cols-[340px_1fr]">
        <aside className="xl:sticky xl:top-6 xl:h-fit">
          <ControlPanel
            request={request}
            onChange={patch}
            onRun={() => void run(request)}
            running={running}
          />
        </aside>

        <div className="min-w-0 space-y-4">
          {error && <ErrorNote message={error} onRetry={() => void run(request)} />}
          {running && !result && <LoadingState />}
          {result && (
            <div className={running ? "opacity-60 transition-opacity" : "transition-opacity"}>
              <Results result={result} />
            </div>
          )}
          <Methodology />
        </div>
      </div>

      <footer className="mt-10 border-t border-line pt-5 text-[11px] leading-relaxed text-muted">
        <p>
          Deflated Sharpe Ratio, PBO, skew, kurtosis and bootstrap are computed locally in NumPy
          and SciPy — nothing is sent anywhere else. Research and education only; not investment
          advice.
        </p>
      </footer>
    </main>
  );
}

function Header({ running, meta }: { running: boolean; meta: number | null }) {
  return (
    <header className="flex flex-wrap items-end justify-between gap-4">
      <div>
        <div className="flex items-center gap-2.5">
          <span className="relative flex h-2 w-2">
            <span
              className={`absolute inline-flex h-full w-full rounded-full ${
                running ? "animate-ping bg-beam" : "bg-pass"
              } opacity-75`}
            />
            <span
              className={`relative inline-flex h-2 w-2 rounded-full ${
                running ? "bg-beam" : "bg-pass"
              }`}
            />
          </span>
          <h1 className="text-[22px] font-semibold tracking-tight">
            Backtest Overfitting Detector
          </h1>
        </div>
        <p className="mt-1.5 max-w-2xl text-[13px] leading-relaxed text-muted">
          Generate a thousand strategies, backtest them on the same data, and find out whether
          the winner has an edge — or whether you simply searched hard enough to find noise that
          looked like one.
        </p>
      </div>
      <div className="text-right text-[11px] text-muted">
        <div className="font-mono">
          {running ? "computing…" : meta !== null ? `${meta.toFixed(0)} ms engine time` : "idle"}
        </div>
        <div className="mt-0.5">NumPy · SciPy · CSCV · Monte Carlo</div>
      </div>
    </header>
  );
}

function LoadingState() {
  return (
    <div className="flex flex-col gap-4">
      <Skeleton className="h-[230px]" />
      <div className="grid grid-cols-2 gap-2.5 sm:grid-cols-3 xl:grid-cols-6">
        {Array.from({ length: 12 }).map((_, i) => (
          <Skeleton key={i} className="h-[78px]" />
        ))}
      </div>
      <Skeleton className="h-[300px]" />
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <Skeleton className="h-[340px]" />
        <Skeleton className="h-[340px]" />
      </div>
    </div>
  );
}
