import type { AnalysisResult, AnalyzeRequest } from "./types";

/**
 * Call the Python engine.
 *
 * In development Next.js rewrites `/api/py/*` to the local uvicorn process; in
 * production Vercel routes it to the `api/index.py` serverless function.
 */
export async function analyze(
  request: AnalyzeRequest,
  signal?: AbortSignal
): Promise<AnalysisResult> {
  const response = await fetch("/api/py/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
    signal,
  });

  if (!response.ok) {
    let message = `Analysis failed (HTTP ${response.status}).`;
    try {
      const body = await response.json();
      if (typeof body?.error === "string") {
        message = body.error;
      } else if (Array.isArray(body?.detail) && body.detail.length) {
        // FastAPI validation errors arrive as a list of field problems.
        const first = body.detail[0];
        const field = Array.isArray(first?.loc) ? first.loc.slice(1).join(".") : "input";
        message = `${field}: ${first?.msg ?? "invalid value"}`;
      }
    } catch {
      // Response body was not JSON — keep the status-based message.
    }
    throw new Error(message);
  }

  return (await response.json()) as AnalysisResult;
}

export interface HealthInfo {
  status: string;
  engine: string;
  python: string;
  numpy: string;
  scipy: string;
}

export async function health(signal?: AbortSignal): Promise<HealthInfo> {
  const response = await fetch("/api/py/health", { signal });
  if (!response.ok) throw new Error(`Engine unreachable (HTTP ${response.status}).`);
  return (await response.json()) as HealthInfo;
}
