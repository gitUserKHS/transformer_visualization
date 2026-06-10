import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { useActivity } from "./useActivity";
import type { ActivityResponse, FactoryRunSummary } from "./types";

const factoryRun = {
  id: "active-factory",
  name: "active",
  status: "running",
  stage: "pretrain",
  stage_step: 1,
  stage_total: 10,
  overall_progress: 0.1,
  config: {},
  metrics: {},
  checkpoints: {},
  comparisons: {},
  peak_vram_bytes: 0,
  created_at: 1,
} as unknown as FactoryRunSummary;

class FakeWebSocket {
  static instances: FakeWebSocket[] = [];
  url: string;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onopen: (() => void) | null = null;
  onclose: (() => void) | null = null;

  constructor(url: string) {
    this.url = url;
    FakeWebSocket.instances.push(this);
    queueMicrotask(() => this.onopen?.());
  }

  close() {}
}

describe("useActivity", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
    FakeWebSocket.instances = [];
    document.title = "Transformer Learning Studio";
  });

  it("recovers an active run and reconnects after the last event cursor", async () => {
    const activity: ActivityResponse = {
      server_time: 1,
      factory: factoryRun,
      microscope: null,
    };
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => activity,
      }),
    );
    vi.stubGlobal("WebSocket", FakeWebSocket);

    const { result, unmount } = renderHook(() => useActivity());

    await waitFor(() => expect(result.current.factory.run?.id).toBe("active-factory"));
    await waitFor(() => expect(FakeWebSocket.instances).toHaveLength(1));
    expect(FakeWebSocket.instances[0].url).toContain("after=-1");

    act(() => {
      FakeWebSocket.instances[0].onmessage?.({
        data: JSON.stringify({
          index: 4,
          type: "pretrain",
          run_id: "active-factory",
          step: 2,
          loss: 2.4,
        }),
      } as MessageEvent);
      FakeWebSocket.instances[0].onclose?.();
    });

    await waitFor(
      () => expect(FakeWebSocket.instances).toHaveLength(2),
      { timeout: 1200 },
    );
    expect(FakeWebSocket.instances[1].url).toContain("after=4");
    unmount();
  });

  it("clears stale activity after a server restart", async () => {
    const activity: ActivityResponse = {
      server_time: 1,
      factory: factoryRun,
      microscope: null,
    };
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue({
        ok: true,
        json: async () => activity,
      }),
    );
    vi.stubGlobal("WebSocket", FakeWebSocket);
    const { result, unmount } = renderHook(() => useActivity());
    await waitFor(() => expect(FakeWebSocket.instances).toHaveLength(1));

    act(() => {
      FakeWebSocket.instances[0].onmessage?.({
        data: JSON.stringify({
          type: "error",
          reason: "not_found",
          message: "실행을 찾을 수 없습니다.",
        }),
      } as MessageEvent);
    });

    await waitFor(() => expect(result.current.factory.run).toBeNull());
    expect(result.current.active).toHaveLength(0);
    unmount();
  });
});
