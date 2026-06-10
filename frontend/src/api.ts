import type {
  Bootstrap,
  ActivityResponse,
  FactoryRunConfig,
  FactoryRunSummary,
  Generation,
  KVBenchmark,
  PreferencePair,
  RunSummary,
  TrainingConfig,
} from "./types";

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail ?? `요청에 실패했습니다 (${response.status})`);
  }
  return response.json() as Promise<T>;
}

export const api = {
  bootstrap: () => request<Bootstrap>("/api/bootstrap"),
  activity: () => request<ActivityResponse>("/api/activity"),
  createRun: (config: TrainingConfig) =>
    request<RunSummary>("/api/runs", {
      method: "POST",
      body: JSON.stringify(config),
    }),
  controlRun: (runId: string, action: string) =>
    request<RunSummary>(`/api/runs/${runId}/control`, {
      method: "POST",
      body: JSON.stringify({ action }),
    }),
  generate: (payload: {
    prompt: string;
    max_new_tokens: number;
    temperature: number;
    top_k: number;
    seed: number;
    run_id?: string;
    layer?: number;
    head?: number;
  }) =>
    request<Generation>("/api/generate", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  inspect: (payload: {
    token_ids: number[];
    run_id?: string;
    layer: number;
    head: number;
    token_index: number;
  }) =>
    request<import("./types").Trace>("/api/inspect", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
  system: () => request<import("./types").SystemInfo>("/api/system"),
  createFactoryRun: (config: FactoryRunConfig) =>
    request<FactoryRunSummary>("/api/factory/runs", {
      method: "POST",
      body: JSON.stringify(config),
    }),
  controlFactoryRun: (runId: string, action: string) =>
    request<FactoryRunSummary>(`/api/factory/runs/${runId}/control`, {
      method: "POST",
      body: JSON.stringify({ action }),
    }),
  nextPreference: () => request<PreferencePair>("/api/preferences/next"),
  submitPreference: (pairId: string, choice: "a" | "b" | "tie") =>
    request<{ saved: boolean; user_vote_count: number }>("/api/preferences", {
      method: "POST",
      body: JSON.stringify({ pair_id: pairId, choice }),
    }),
  kvBenchmark: (prompt: string, maxNewTokens: number) =>
    request<KVBenchmark>("/api/inference/kv-benchmark", {
      method: "POST",
      body: JSON.stringify({
        prompt,
        max_new_tokens: maxNewTokens,
        seed: 42,
      }),
    }),
  scaleEstimate: (payload: Record<string, number>) =>
    request<Record<string, unknown>>("/api/scale/estimate", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
};

export function openRunSocket(
  runId: string,
  after: number,
  onMessage: (event: Record<string, unknown>) => void,
  onOpen?: () => void,
  onClose?: () => void,
): WebSocket {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const socket = new WebSocket(
    `${protocol}//${window.location.host}/ws/runs/${runId}?after=${after}`,
  );
  socket.onmessage = (event) => onMessage(JSON.parse(event.data));
  socket.onopen = () => onOpen?.();
  socket.onclose = () => onClose?.();
  return socket;
}

export function openFactorySocket(
  runId: string,
  after: number,
  onMessage: (event: Record<string, unknown>) => void,
  onOpen?: () => void,
  onClose?: () => void,
): WebSocket {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const socket = new WebSocket(
    `${protocol}//${window.location.host}/ws/factory/${runId}?after=${after}`,
  );
  socket.onmessage = (event) => onMessage(JSON.parse(event.data));
  socket.onopen = () => onOpen?.();
  socket.onclose = () => onClose?.();
  return socket;
}
