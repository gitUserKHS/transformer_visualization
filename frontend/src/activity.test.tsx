import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ActivityBar } from "./components/ActivityBar";
import { reduceActivityEvent } from "./useActivity";
import type { FactoryRunSummary } from "./types";

const run = {
  id: "factory-1",
  name: "공장",
  status: "running",
  stage: "pretrain",
  stage_step: 42,
  stage_total: 100,
  overall_progress: 0.42,
  eta_seconds: 73,
  config: { stages: [] },
  metrics: {},
  checkpoints: {},
  comparisons: {},
  peak_vram_bytes: 0,
  created_at: 1,
} as unknown as FactoryRunSummary;

describe("activity monitoring", () => {
  it("does not replace lifecycle status with a stage event status", () => {
    const next = reduceActivityEvent(run, {
      type: "stage",
      stage: "sft",
      stage_status: "running",
    });

    expect(next?.status).toBe("running");
    expect((next as FactoryRunSummary).stage).toBe("sft");
  });

  it("renders the global progress, ETA and connection state", () => {
    render(
      <ActivityBar
        items={[{ track: "factory", run, connection: "connected" }]}
        onNavigate={() => undefined}
      />,
    );

    expect(screen.getByText("모델 공장")).toBeInTheDocument();
    expect(screen.getByText("사전학습 중")).toBeInTheDocument();
    expect(screen.getByText("42%")).toBeInTheDocument();
    expect(screen.getByText("ETA 1분 13초")).toBeInTheDocument();
  });
});
